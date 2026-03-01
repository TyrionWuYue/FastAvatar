import torch
import torch.nn as nn
from typing import Optional, Callable, List
from torch import Tensor
import timm
from safetensors.torch import load_file

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FineFLAMEHead(nn.Module):
    def __init__(
        self,
        features=512,
        # Output dimensions
        expr_dim=100,
        jaw_dim=3,
        eye_pose_dim=6,
        # Backbone
        backbone_path="./model_zoo/mobilenet/model.safetensors",
        # Conditioning
        coarse_feat_dim=0,  # If >0, enable coarse-to-fine conditioning
        **kwargs
    ):
        super().__init__()

        # 1. Backbone
        print(f"Creating MobileNetV3 Large Minimal backbone...")
        self.backbone = timm.create_model(
            'tf_mobilenetv3_large_minimal_100', 
            pretrained=False, 
            features_only=True
        )
        
        if backbone_path:
            print(f"Loading backbone weights from {backbone_path}")
            try:
                state_dict = load_file(backbone_path)
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            except Exception as e:
                print(f"Error loading backbone weights: {e}")
                raise e

        # Get the feature dimension automatically
        self.out_channels = self.backbone.feature_info[-1]['num_chs']
        
        # 2. Prediction Heads
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Adaptive Fusion Module (if coarse conditioning enabled)
        self.coarse_feat_dim = coarse_feat_dim
        
        if coarse_feat_dim > 0:
            self.coarse_adapter = nn.Linear(coarse_feat_dim, self.out_channels)
            
            gate_input_dim = 2 * self.out_channels
            self.fusion_gate = nn.Sequential(
                nn.Linear(gate_input_dim, features),
                nn.ReLU(),
                nn.Linear(features, 1),
            )
            nn.init.constant_(self.fusion_gate[-1].bias, -3.0)  # Start conservative
            
            # Layer normalization for stable fusion
            self.cnn_norm = nn.LayerNorm(self.out_channels)
            self.coarse_norm = nn.LayerNorm(self.out_channels)
            
            mlp_in_dim = self.out_channels
        else:
            mlp_in_dim = self.out_channels

        out_dim = expr_dim + jaw_dim + eye_pose_dim 
        self.param_head = Mlp(
            in_features=mlp_in_dim,
            hidden_features=features,
            out_features=out_dim,
        )

        self.init_weights()

        # Output dims
        self.expr_dim = expr_dim
        self.jaw_dim = jaw_dim
        self.eye_pose_dim = eye_pose_dim

    def init_weights(self):
        last_layer = self.param_head.fc2 
        nn.init.constant_(last_layer.weight, 0)
        nn.init.constant_(last_layer.bias, 0)

    def forward(self, images, coarse_features=None):
        """
        Args:
            images: [B, S, 3, H, W]
            coarse_features: Optional [B, S, D] - refined tokens from CoarseFLAMEHead
        """
        B, S, C, H, W = images.shape
        
        # Flatten B and S: [B*S, 3, H, W]
        x = images.view(B * S, C, H, W)
        
        # Backbone features
        features = self.backbone(x)
        # timm features_only returns a list of feature maps
        x_features = features[-1] # [B*S, C_out, H_f, W_f]
        
        x_global = self.global_pool(x_features).flatten(1)  # [B*S, out_channels]
        
        # Adaptive fusion: gate-based selection between CNN and coarse
        if coarse_features is not None and self.coarse_feat_dim > 0:
            # coarse_features: [B, S, D] → [B*S, D]
            coarse_flat = coarse_features.view(B * S, -1)
            
            # Project both to same dimension and normalize
            x_cnn = self.cnn_norm(x_global)  # [B*S, out_channels]
            x_coarse = self.coarse_norm(self.coarse_adapter(coarse_flat))  # [B*S, out_channels]
            
            # Gate: learns fusion weight based on BOTH projected features
            gate_input = torch.cat([x_cnn, x_coarse], dim=-1)  # [B*S, 2*out_channels]
            gate = torch.sigmoid(self.fusion_gate(gate_input))  # [B*S, 1]
            
            # Linear interpolation fusion
            x_fused = (1 - gate) * x_cnn + gate * x_coarse  # [B*S, out_channels]
        else:
            x_fused = x_global
        
        params = self.param_head(x_fused)
        expr = params[:, :self.expr_dim]
        jaw_pose = params[:, self.expr_dim : self.expr_dim + self.jaw_dim]
        eye_pose = params[:, self.expr_dim + self.jaw_dim : ]

        return expr, jaw_pose, eye_pose


class CoarseFLAMEHead(nn.Module):
    def __init__(self,
                 dim_in: int = 2048,
                 hidden_dim: int = 1024,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 shape_dim: int = 300,
                 rotation_dim: int = 3,
                 neck_pose_dim: int = 3,
                 translation_dim: int = 3,
                 eps: float = 1e-6):
        super().__init__()
        
        from FastAvatar.models.utils.block import MemEffBasicBlock
        
        self.shape_dim = shape_dim
        self.rotation_dim = rotation_dim
        self.neck_pose_dim = neck_pose_dim
        self.translation_dim = translation_dim
        
        self.attn_block = MemEffBasicBlock(
            inner_dim=dim_in, 
            num_heads=num_heads, 
            eps=eps, 
            mlp_ratio=mlp_ratio
        )
        
        # Shape prediction: Sequence-level (identity feature, should be constant)
        # Use mean pooling (permutation-invariant) across frames
        self.shape_mlp = Mlp(
            in_features=dim_in,
            hidden_features=hidden_dim,
            out_features=shape_dim,
            drop=0.0
        )
        
        # Per-frame predictions (pose can vary per frame)
        self.rotation_mlp = Mlp(
            in_features=dim_in,
            hidden_features=hidden_dim // 2,
            out_features=rotation_dim,
            drop=0.0
        )
        
        self.neck_mlp = Mlp(
            in_features=dim_in,
            hidden_features=hidden_dim // 2,
            out_features=neck_pose_dim,
            drop=0.0
        )
        
        self.translation_mlp = Mlp(
            in_features=dim_in,
            hidden_features=hidden_dim // 2,
            out_features=translation_dim,
            drop=0.0
        )
        
        # Initialize to zero for stable training
        for mlp in [self.shape_mlp, self.rotation_mlp, self.neck_mlp, self.translation_mlp]:
            nn.init.constant_(mlp.fc2.weight, 0)
            nn.init.constant_(mlp.fc2.bias, 0)

    def forward(self, coarse_tokens, return_features=False):
        """
        Args:
            coarse_tokens: [B, S, C] - Contextualized multi-frame tokens (unordered)
            return_features: If True, also return refined_tokens for conditioning
            
        Returns:
            If return_features=False:
                Tuple of (shape, rotation, neck_pose, translation)
            If return_features=True:
                Tuple of (shape, rotation, neck_pose, translation, refined_tokens)
                
            Where:
                - shape: [B, S, 300] - Same value across all S frames (identity constant)
                - rotation: [B, S, 3] - Per-frame rotation
                - neck_pose: [B, S, 3] - Per-frame neck pose  
                - translation: [B, S, 3] - Per-frame translation
                - refined_tokens: [B, S, C] - Task-adapted features (for conditioning)
        """
        B, S, C = coarse_tokens.shape
        refined_tokens = self.attn_block(coarse_tokens)  # [B, S, C]
        
        # Shape: Per-frame prediction then average (multi-view voting)
        shape_per_frame = self.shape_mlp(refined_tokens)  # [B, S, 300]
        shape = shape_per_frame.mean(dim=1, keepdim=True)  # [B, 1, 300]
        shape = shape.expand(-1, S, -1)  # [B, S, 300]
        
        # Per-frame predictions for pose parameters
        rotation = self.rotation_mlp(refined_tokens)      # [B, S, 3]
        neck_pose = self.neck_mlp(refined_tokens)         # [B, S, 3]
        translation = self.translation_mlp(refined_tokens) # [B, S, 3]
        
        if return_features:
            return shape, rotation, neck_pose, translation, refined_tokens
        else:
            return shape, rotation, neck_pose, translation


class FlameHead(nn.Module):
    """
    - FineFLAMEHead (CNN): Low-level details → expr, jaw_pose, eyes_pose
    - CoarseFLAMEHead (Transformer): 3D global features → shape, rotation, neck_pose, translation
    """
    def __init__(
        self,
        # Fine Head (CNN-based) config
        features=512,
        expr_dim=100,
        jaw_dim=3,
        eyes_pose_dim=6,
        backbone_path="./model_zoo/mobilenet/model.safetensors",
        # Coarse Head config
        coarse_dim_in: int = 2048,  # Should match 2 * inner_dim from AlternatingAttn
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        shape_dim: int = 300,
        rotation_dim: int = 3,
        neck_pose_dim: int = 3,
        translation_dim: int = 3,
        **kwargs
    ):
        super().__init__()
        
        print("Creating FlameHead with dual-path architecture (Fine + Coarse)...")
        
        # Fine Branch: CNN for low-level features
        # Enable coarse-to-fine conditioning by passing coarse_feat_dim
        self.fine_head = FineFLAMEHead(
            features=features,
            expr_dim=expr_dim,
            jaw_dim=jaw_dim,
            eye_pose_dim=eyes_pose_dim,
            backbone_path=backbone_path,
            coarse_feat_dim=coarse_dim_in,  # Enable conditioning with refined tokens
            **kwargs
        )
        
        # Coarse Branch: Single attention layer for task-specific adaptation
        coarse_hidden_dim = kwargs.get('coarse_hidden_dim', coarse_dim_in // 2)
        self.coarse_head = CoarseFLAMEHead(
            dim_in=coarse_dim_in,
            hidden_dim=coarse_hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            shape_dim=shape_dim,
            rotation_dim=rotation_dim,
            neck_pose_dim=neck_pose_dim,
            translation_dim=translation_dim,
            eps=eps
        )
        
        # Store dimensions for reference
        self.expr_dim = expr_dim
        self.jaw_dim = jaw_dim
        self.eyes_pose_dim = eyes_pose_dim
        self.shape_dim = shape_dim
        self.rotation_dim = rotation_dim
        self.neck_pose_dim = neck_pose_dim
        self.translation_dim = translation_dim

    def forward(self, images, coarse_tokens, enable_conditioning=True):
        """
        Forward pass integrating fine and coarse predictions.
        
        Args:
            images: [B, S, 3, H, W] - Input images for CNN-based fine prediction
            coarse_tokens: [B, S, C] - Contextualized tokens for transformer-based coarse prediction
            enable_conditioning: If True, use coarse features to guide fine prediction
            
        Returns:
            Dictionary containing:
                - 'shape': [B, S, 300] - from Coarse Head
                - 'rotation': [B, S, 3] - from Coarse Head
                - 'neck_pose': [B, S, 3] - from Coarse Head
                - 'translation': [B, S, 3] - from Coarse Head
                - 'expr': [B, S, 100] - from Fine Head (conditioned on coarse)
                - 'jaw_pose': [B, S, 3] - from Fine Head (conditioned on coarse)
                - 'eyes_pose': [B, S, 6] - from Fine Head (conditioned on coarse)
        """
        B, S = images.shape[:2]
        
        # 1. Coarse Head: Get pose/shape features first
        if enable_conditioning and self.fine_head.coarse_feat_dim > 0:
            # Get both predictions and features for conditioning
            shape, rotation, neck_pose, translation, coarse_features = self.coarse_head(
                coarse_tokens, return_features=True
            )
        else:
            # No conditioning, just get predictions
            shape, rotation, neck_pose, translation = self.coarse_head(coarse_tokens)
            coarse_features = None
        
        # 2. Fine Head: CNN-based prediction conditioned on coarse features
        expr, jaw_pose, eyes_pose = self.fine_head(images, coarse_features=coarse_features)
        
        # Reshape from [B*S, D] to [B, S, D]
        expr = expr.view(B, S, -1)
        jaw_pose = jaw_pose.view(B, S, -1)
        eyes_pose = eyes_pose.view(B, S, -1)

        # 3. Combine both branches
        result = {
            # Coarse (3D global features)
            'shape': shape,              # All frames share same shape (identity constant)
            'rotation': rotation,
            'neck_pose': neck_pose,
            'translation': translation,
            # Fine (low-level details)
            'expr': expr,
            'jaw_pose': jaw_pose,
            'eyes_pose': eyes_pose,
        }
        
        return result
