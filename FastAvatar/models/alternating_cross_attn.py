# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Any, Dict, Tuple, List
from diffusers.utils import is_torch_version
from FastAvatar.models.block import BasicBlock, MemEffBasicBlock
from FastAvatar.models.pos_embed import RoPE2D, get_1d_sincos_pos_embed_from_grid
from FastAvatar.models.pose_encoder import FLAMEPoseEncoder, CameraPoseEncoder
from FastAvatar.models.cross_attn import SD3JointTransformerBlock


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameAttn(nn.Module):
    """
    Frame attention module based on SD3JointTransformerBlock.
    """

    def __init__(self, num_layers: int = 10, num_heads: int = 16,
                 inner_dim: int = 1024, cond_dim: int = None,
                 gradient_checkpointing=True,
                 eps: float = 1e-6):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        
        # Create layers using SD3JointTransformerBlock
        self.layers = nn.ModuleList([
            SD3JointTransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                qk_norm="rms_norm"
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        
        if cond_dim is not None:
            self.linear_cond_proj = nn.Linear(cond_dim, inner_dim)
        else:
            self.linear_cond_proj = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond] or None
        
        if cond is not None and self.linear_cond_proj is not None:
            cond = self.linear_cond_proj(cond)
        
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x, cond = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    cond,
                    **ckpt_kwargs,
                )
            else:
                x, cond = layer(
                    hidden_states=x,
                    encoder_hidden_states=cond,
                    temb=None,
                )
        
        x = self.norm(x)
        return x


class AlternatingCrossAttn(nn.Module):
    """
    Alternating attention mechanism that switches between frame and global attention.
    Frame attention uses FrameAttn with SD3JointTransformerBlock.
    Global attention uses basic transformer block for image features only.
    x: image features [B, N_frames, H*W, C]
    cond: point features [B, N_frames, N_points, C]
    """
    def __init__(self, 
                 num_layers: int, 
                 num_heads: int,
                 inner_dim: int, 
                 cond_dim: int = None,
                 gradient_checkpointing: bool = True,
                 eps: float = 1e-6,
                 aa_order: list = ["frame", "global"],
                 aa_block_size: int = 1,
                 intermediate_layer_idx: List[int] = [],
                 use_flame_tokens: bool = False,
                 flame_encoder_config: dict = None,
                 use_camera_tokens: bool = False,
                 camera_encoder_config: dict = None,
                 patch_start_idx: int = 0):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_flame_tokens = use_flame_tokens
        self.rope2d = RoPE2D(freq=300.0)
        self.patch_start_idx = patch_start_idx
        # Register frame-level index embedding
        self.register_buffer(
            "frame_idx_emb",
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(inner_dim, np.arange(300))).float(),
            persistent=False,
        )
        
        # Calculate number of global attention layers needed
        # (num_layers - 1) // aa_block_size layers
        self.num_global_layers = (num_layers - 1) // aa_block_size
        
        # Frame attention (using FrameAttn with SD3JointTransformerBlock)
        self.frame_attn = FrameAttn(
            num_layers=num_layers,
            num_heads=num_heads,
            inner_dim=inner_dim,
            cond_dim=cond_dim,
            gradient_checkpointing=gradient_checkpointing,
            eps=eps,
        )

        # Global attention
        self.global_attn_layers = nn.ModuleList([
            MemEffBasicBlock(
                inner_dim=inner_dim,
                num_heads=num_heads,
                eps=eps,
                rope=self.rope2d
            )
            for _ in range(self.num_global_layers)
        ])
        
        # FLAME pose encoder
        if self.use_flame_tokens:
            if flame_encoder_config is None:
                flame_encoder_config = {
                    'expr_dim': 10,
                    'rotation_dim': 3,
                    'neck_pose_dim': 3,
                    'jaw_pose_dim': 3,
                    'eyes_pose_dim': 6,
                    'translation_dim': 3,
                    'hidden_dim': 256,
                    'output_dim': inner_dim,
                    'dropout': 0.1
                }
            self.flame_encoder = FLAMEPoseEncoder(**flame_encoder_config)
            print("[FLAME] FLAME pose encoder initialized")

        # Camera pose encoder
        self.use_camera_tokens = use_camera_tokens
        if self.use_camera_tokens:
            if camera_encoder_config is None:
                camera_encoder_config = {
                    'hidden_dim': 256,
                    'output_dim': inner_dim,
                    'dropout': 0.1
                }
            self.camera_encoder = CameraPoseEncoder(**camera_encoder_config)
            print("[CAMERA] Camera pose encoder initialized")
    
    def forward_frame_attn(self, layer_idx: int, x: torch.Tensor, cond: torch.Tensor = None, mod: torch.Tensor = None):
        """
        Forward pass for frame attention with FLAME token fusion.
        
        Args:
            layer_idx: Index of the transformer layer
            x: Point features [B, N_input, N_points, C]
            cond: Image features [B, N_input, HW, C] (may include FLAME tokens)
            mod: Modulation tensor (optional)
        """
        B, N_input, N_points, C = x.shape
        B2, N_input2, HW, C2 = cond.shape
        assert B == B2 and C == C2 and N_input == N_input2, "Batch, channel and frame dimensions must match"
        
        x_reshaped = x.reshape(B * N_input, N_points, C)  # [B*N_input, N_points, C]
        cond_reshaped = cond.reshape(B * N_input, HW, C)  # [B*N_input, HW, C]
        
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x_out, cond_out = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.frame_attn.layers[layer_idx]),
                x_reshaped,  # [B*N_input, N_points, C]
                cond_reshaped,  # [B*N_input, HW, C]
                **ckpt_kwargs,
            )
        else:
            x_out, cond_out = self.frame_attn.layers[layer_idx](
                hidden_states=x_reshaped,  # [B*N_input, N_points, C]
                encoder_hidden_states=cond_reshaped,  # [B*N_input, HW, C]
                temb=None,
            )
        
        # Reshape back to original dimensions
        x_out = x_out.reshape(B, N_input, N_points, C)  # [B, N_input, N_points, C]
        cond_out = cond_out.reshape(B, N_input, HW, C)  # [B, N_input, HW, C]
        
        return x_out, cond_out

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, mod: torch.Tensor = None, 
                flame_params: Dict[str, torch.Tensor] = None, 
                camera_params: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Alternating cross attention with FLAME and camera token integration.
        Args:
            x: [B, N_input, N_points, C] - point features
            cond: [B, N_input, H*W, C] - image features
            flame_params: Dictionary containing FLAME parameters for encoding into tokens
            camera_params: Dictionary containing camera parameters for encoding into tokens
                - c2w: [B, N_input, 4, 4] - Camera-to-world matrices
                - intrinsic: [B, N_input, 4, 4] - Intrinsic matrices
        Returns:
            torch.Tensor: Processed point features [B, N_input, N_points, C]
            torch.Tensor: Last pre-attn image features [B, N_input, H*W, C]
            List[torch.Tensor]: Intermediate layer outputs, concatenated frame and global features
        """
        B, N_input, N_points, C = x.shape
        B2, N_input2, HW, C2 = cond.shape
        assert B == B2 and C == C2 and N_input == N_input2, "Batch, channel and frame dimensions must match"

        cond = self.frame_attn.linear_cond_proj(cond) if hasattr(self.frame_attn, 'linear_cond_proj') else cond

        frame_indices = torch.arange(N_input, device=cond.device)
        frame_emb = self.frame_idx_emb[frame_indices]  # [N_input, C]
        cond = cond + frame_emb[None, :, None, :]      # [B, N_input, HW, C]

        # Encode FLAME parameters into tokens if enabled
        flame_tokens = None
        if self.use_flame_tokens and flame_params is not None:
            flame_tokens = self.flame_encoder(flame_params)  # [B, N_input, C]
            
            # Add FLAME tokens to image features
            flame_tokens_expanded = flame_tokens.unsqueeze(2)  # [B, N_input, 1, C]
            cond = torch.cat([flame_tokens_expanded, cond], dim=2)  # [B, N_input, 1+HW, C]

        # Encode camera parameters into tokens if enabled
        camera_tokens = None
        if self.use_camera_tokens and camera_params is not None:
            camera_tokens = self.camera_encoder(camera_params['c2w'], camera_params['intrinsic'])  # [B, N_input, C]
            
            # Add camera tokens to image features
            camera_tokens_expanded = camera_tokens.unsqueeze(2)  # [B, N_input, 1, C]
            cond = torch.cat([camera_tokens_expanded, cond], dim=2)  # [B, N_input, 1+HW, C]

        H = W = int(HW ** 0.5)
        patch_positions = torch.stack(torch.meshgrid(
            torch.arange(H, device=cond.device),
            torch.arange(W, device=cond.device),
            indexing='ij'), dim=-1).reshape(-1, 2)  # [H*W, 2]
        
        # Repeat patch positions for each frame
        patch_positions = patch_positions.repeat(N_input, 1)  # [N_input*HW, 2]
        positions = patch_positions[None, :, :].expand(B, N_input * HW, 2)  # [B, N_input*HW, 2]

        intermediate_outputs = []

        global_layer_idx = 0
        for i in range(len(self.frame_attn.layers)):
            # Save cond before last frame attn
            if i == len(self.frame_attn.layers) - 1:
                last_cond = cond
            
            # Frame attention
            x, cond = self.forward_frame_attn(i, x, cond, mod)
            frame_feat = cond
            
            # Global attention
            if (i + 1) % self.aa_block_size == 0 and i < len(self.frame_attn.layers) - 1:
                cond_reshape = cond.reshape(B, N_input * (self.patch_start_idx + HW), C)  # Account for FLAME tokens
                
                # Calculate total number of additional tokens (FLAME + Camera)
                num_extra_tokens = 0
                if self.use_flame_tokens and flame_tokens is not None:
                    num_extra_tokens += 1
                if self.use_camera_tokens and camera_tokens is not None:
                    num_extra_tokens += 1
                
                if num_extra_tokens > 0:
                    # Create zero positions for FLAME and camera tokens
                    extra_positions = torch.zeros(B, N_input * num_extra_tokens, 2, device=positions.device, dtype=torch.long)
                    combined_positions = torch.cat([extra_positions, positions], dim=1)  # [B, N_input*(num_extra_tokens+HW), 2]
                else:
                    combined_positions = positions
                
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    cond_reshape = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.global_attn_layers[global_layer_idx]),
                        cond_reshape,
                        combined_positions,
                        **ckpt_kwargs,
                    )
                else:
                    cond_reshape = self.global_attn_layers[global_layer_idx](cond_reshape, pos=combined_positions)
                
                cond = cond_reshape.reshape(B, N_input, self.patch_start_idx + HW, C)
                global_layer_idx += 1
                global_feat = cond

            if i in self.intermediate_layer_idx:
                combined_feat = torch.cat([frame_feat, global_feat], dim=-1)
                intermediate_outputs.append(combined_feat)

        # Remove all additional tokens (FLAME + Camera) from final output
        if self.patch_start_idx > 0:
            last_cond = last_cond[:, :, self.patch_start_idx:, :]  # Remove all additional tokens: [B, N_input, HW, C]

        x = self.frame_attn.norm(x)

        return x, last_cond, intermediate_outputs
