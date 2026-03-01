import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from torch import Tensor
from FastAvatar.models.utils.block import MemEffBasicBlock

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate the input tensor using scaling and shifting parameters."""
    return x * (1 + scale) + shift

def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))

def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")

def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: Activation type for focal length component

    Returns:
        Activated pose parameters tensor
    """
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]  # or fov

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # or fov

    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

    return pred_pose_enc

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

class CameraHead(nn.Module):
    """
    Iterative Camera Head inspired by VGGT.
    Predicts camera parameters using a transformer trunk and AdaLN modulation.
    """
    def __init__(self,
                 dim_in: int = 1024,
                 trunk_depth: int = 4,
                 target_dim: int = 9,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 eps: float = 1e-6,
                 trans_act: str = "linear",
                 quat_act: str = "linear",
                 fl_act: str = "relu"):
        super().__init__()
        
        self.target_dim = target_dim
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Trunk: Sequence of transformer blocks
        self.trunk = nn.Sequential(*[
            MemEffBasicBlock(inner_dim=dim_in, num_heads=num_heads, eps=eps, mlp_ratio=mlp_ratio)
            for _ in range(trunk_depth)
        ])

        # Normalizations
        self.token_norm = nn.LayerNorm(dim_in, eps=eps)
        self.trunk_norm = nn.LayerNorm(dim_in, eps=eps)
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=eps)

        # Learnable empty pose and its embedding
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Modulation parameters generator
        self.poseLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_in, 3 * dim_in)
        )

        # Branch to predict the delta of pose encoding
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, pose_tokens, num_iterations: int = 4):
        """
        Iteratively refine camera pose predictions from normalized camera tokens [B, S, C].
        """
        B, S, C = pose_tokens.shape
        pose_tokens = self.token_norm(pose_tokens)
        
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # 1. Prepare module input from previous prediction or empty pose
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach previous prediction for stability if needed (typical for iterative heads)
                prev_pred = pred_pose_enc.detach()
                module_input = self.embed_pose(prev_pred)

            # 2. Modulation with AdaLN
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            
            # [B, S, C]
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            # 3. Process through Trunk
            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            
            # 4. Predict Delta
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # 5. Activate and store
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list
