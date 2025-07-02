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


from functools import partial
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Any, Dict, Optional, Tuple, Union, List
from diffusers.utils import is_torch_version
from safetensors.torch import load_file
from VGGTAvatar.models.block import BasicBlock, MemEffBasicBlock
from VGGTAvatar.models.pos_embed import RoPE2D, get_1d_sincos_pos_embed_from_grid
from VGGTAvatar.models.pose_encoder import FLAMEPoseEncoder

# === LoRA (peft) import ===
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerDecoder(nn.Module):

    """
    Transformer blocks that process the input and optionally use condition and modulation.
    """

    def __init__(self, block_type: str = 'sd3_cond',
                 num_layers: int = 10, num_heads: int = 16,
                 inner_dim: int = 1024, cond_dim: int = None, mod_dim: int = None,
                 gradient_checkpointing=True,
                 eps: float = 1e-6,
                 use_dual_attention: bool = False,
                 pretrained_attn_dict: dict = None):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.block_type = block_type
        dual_attention_layers = []
        self.layers = nn.ModuleList([
            self._block_fn(inner_dim, cond_dim, mod_dim)(
                num_heads=num_heads,
                eps=eps,
                context_pre_only = i == num_layers - 1,
                use_dual_attention=use_dual_attention,
            )
            for i in range(num_layers)
        ])
        
        
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        
        if self.block_type in ["cogvideo_cond", "sd3_cond"]:
            self.linear_cond_proj = nn.Linear(cond_dim, inner_dim)

        # Load pretrained weights if provided
        if pretrained_attn_dict:
            self.load_state_dict(pretrained_attn_dict, strict=False)
            logger.info("Successfully loaded pretrained weights for transformer")
                
    @property
    def block_type(self):
        return self._block_type

    @block_type.setter
    def block_type(self, block_type):
        assert block_type in ['basic', 'cond', 'mod', 'cond_mod', 'sd3_cond', 'cogvideo_cond'], \
            f"Unsupported block type: {block_type}"
        self._block_type = block_type

    def _block_fn(self, inner_dim, cond_dim, mod_dim):
        assert inner_dim is not None, f"inner_dim must always be specified"
        if self.block_type == 'basic':
            assert cond_dim is None and mod_dim is None, \
                f"Condition and modulation are not supported for BasicBlock"
            from .block import BasicBlock
            # logger.debug(f"Using BasicBlock")
            return partial(BasicBlock, inner_dim=inner_dim)
        elif self.block_type == 'cond':
            assert cond_dim is not None, f"Condition dimension must be specified for ConditionBlock"
            assert mod_dim is None, f"Modulation dimension is not supported for ConditionBlock"
            from .block import ConditionBlock
            # logger.debug(f"Using ConditionBlock")
            return partial(ConditionBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        elif self.block_type == 'mod':
            # logger.error(f"modulation without condition is not implemented")
            raise NotImplementedError(f"modulation without condition is not implemented")
        elif self.block_type == 'cond_mod':
            assert cond_dim is not None and mod_dim is not None, \
                f"Condition and modulation dimensions must be specified for ConditionModulationBlock"
            from .block import ConditionModulationBlock
            # logger.debug(f"Using ConditionModulationBlock")
            return partial(ConditionModulationBlock, inner_dim=inner_dim, cond_dim=cond_dim, mod_dim=mod_dim)
        elif self.block_type == 'cogvideo_cond':
            # logger.debug(f"Using CogVideoXBlock")
            from VGGTAvatar.models.transformer_dit import CogVideoXBlock
            # assert inner_dim == cond_dim, f"inner_dim:{inner_dim}, cond_dim:{cond_dim}"
            return partial(CogVideoXBlock, dim=inner_dim, attention_bias=True)
        elif self.block_type == 'sd3_cond':
            # logger.debug(f"Using SD3JointTransformerBlock")
            from VGGTAvatar.models.transformer_dit import SD3JointTransformerBlock
            return partial(SD3JointTransformerBlock, dim=inner_dim, qk_norm="rms_norm") 
        else:
            raise ValueError(f"Unsupported block type during runtime: {self.block_type}")

    def assert_runtime_integrity(self, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor):
        assert x is not None, f"Input tensor must be specified"
        if self.block_type == 'basic':
            assert cond is None and mod is None, \
                f"Condition and modulation are not supported for BasicBlock"
        elif 'cond' in self.block_type:
            assert cond is not None and mod is None, \
                f"Condition must be specified and modulation is not supported for ConditionBlock"
        elif self.block_type == 'mod':
            raise NotImplementedError(f"modulation without condition is not implemented")
        else:
            assert cond is not None and mod is not None, \
                f"Condition and modulation must be specified for ConditionModulationBlock"

    def forward_layer(self, layer: nn.Module, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor):
        if self.block_type == 'basic':
            return layer(x)
        elif self.block_type == 'cond':
            return layer(x, cond)
        elif self.block_type == 'mod':
            return layer(x, mod)
        else:
            return layer(x, cond, mod)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, mod: torch.Tensor = None):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond] or None
        # mod: [N, D_mod] or None
        self.assert_runtime_integrity(x, cond, mod)
        
        if self.block_type in ["cogvideo_cond", "sd3_cond"]:
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
                        # image_rotary_emb=None,
                    )
            x = self.norm(x)
        else:
            for layer in self.layers:
                x = self.forward_layer(layer, x, cond, mod)
            x = self.norm(x)
        return x


class AlternatingCrossAttn(nn.Module):
    """
    Alternating attention mechanism that switches between frame and global attention.
    Frame attention uses the pretrained TransformerDecoder with sd3_cond.
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
                 pretrained_attn_dict: dict = None,
                 aa_order: list = ["frame", "global"],
                 aa_block_size: int = 1,
                 enable_lora: bool = True,
                 lora_cfg: dict = None,
                 intermediate_layer_idx: List[int] = [],
                 use_flame_tokens: bool = False,
                 flame_encoder_config: dict = None,
                 post_fusion: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_flame_tokens = use_flame_tokens
        self.post_fusion = post_fusion
        self.rope2d = RoPE2D(freq=300.0)
        # Register frame-level index embedding
        self.register_buffer(
            "frame_idx_emb",
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(inner_dim, np.arange(300))).float(),
            persistent=False,
        )
        
        # Calculate number of global attention layers needed
        # (num_layers - 1) // aa_block_size layers
        self.num_global_layers = (num_layers - 1) // aa_block_size
        
        # Frame attention (using pretrained TransformerDecoder with sd3_cond)
        self.frame_attn = TransformerDecoder(
            block_type="sd3_cond",
            num_layers=num_layers,
            num_heads=num_heads,
            inner_dim=inner_dim,
            cond_dim=cond_dim,
            pretrained_attn_dict=pretrained_attn_dict,
            gradient_checkpointing=gradient_checkpointing,
            eps=eps,
        )

        # === LoRA ===
        if enable_lora and lora_cfg is not None and PEFT_AVAILABLE:
            lora_config = LoraConfig(
                r=lora_cfg.get('r', 4),
                lora_alpha=lora_cfg.get('lora_alpha', 32),
                target_modules=lora_cfg.get('target_modules', ['to_q', 'to_k', 'to_v', 'to_out.0']),
                lora_dropout=lora_cfg.get('lora_dropout', 0.0),
                bias=lora_cfg.get('bias', 'none'),
                task_type=lora_cfg.get('task_type', TaskType.FEATURE_EXTRACTION)
            )
            self.frame_attn = get_peft_model(self.frame_attn, lora_config)
            print("[LoRA] frame_attn has been wrapped with LoRA (peft)")
        elif enable_lora and lora_cfg is not None and not PEFT_AVAILABLE:
            print("[LoRA] LoRA is enabled but PEFT is not available. Please install peft package.")
        elif not enable_lora:
            print("[LoRA] LoRA is disabled")
        
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

        # post-fusion layers
        if self.post_fusion:
            self.post_fusion_attn = MemEffBasicBlock(
                inner_dim=inner_dim,
                num_heads=num_heads,
                eps=eps,
                rope=self.rope2d
            )
        
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
        
        x_list, cond_list = [], []
        for i in range(N_input):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x_out, cond_out = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.frame_attn.layers[layer_idx]),
                    x[:, i],  # [B, N_points, C]
                    cond[:, i],  # [B, HW, C]
                    **ckpt_kwargs,
                )
            else:
                x_out, cond_out = self.frame_attn.layers[layer_idx](
                    hidden_states=x[:, i],  # [B, N_points, C]
                    encoder_hidden_states=cond[:, i],  # [B, HW, C]
                    temb=None,
                )
            x_list.append(x_out)
            cond_list.append(cond_out)
        
        x = torch.stack(x_list, dim=1)  # [B, N_input, N_points, C]
        cond = torch.stack(cond_list, dim=1) if cond_list[0] is not None else None  # [B, N_input, HW, C]
        
        # Final shape checks
        assert x.shape == (B, N_input, N_points, C), f"Expected shape {(B, N_input, N_points, C)}, got {x.shape}"
        if cond is not None:
            assert cond.shape == (B, N_input, HW, C), f"Expected shape {(B, N_input, HW, C)}, got {cond.shape}"
        
        return x, cond

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, mod: torch.Tensor = None, 
                flame_params: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Alternating cross attention with FLAME token integration.
        Args:
            x: [B, N_input, N_points, C] - point features
            cond: [B, N_input, H*W, C] - image features
            flame_params: Dictionary containing FLAME parameters for encoding into tokens
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
                cond_reshape = cond.reshape(B, N_input * (1 + HW), C)  # Account for FLAME tokens
                
                if self.use_flame_tokens and flame_tokens is not None:
                    flame_positions = torch.zeros(B, N_input, 2, device=positions.device, dtype=torch.long)  # [B, N_input, 2]
                    combined_positions = torch.cat([flame_positions, positions], dim=1)  # [B, N_input*(1+HW), 2] (FLAME tokens + image patches)
                else:
                    combined_positions = positions
                
                cond_reshape = self.global_attn_layers[global_layer_idx](cond_reshape, pos=combined_positions)
                cond = cond_reshape.reshape(B, N_input, 1 + HW, C)
                global_layer_idx += 1
                global_feat = cond

            if i in self.intermediate_layer_idx:
                combined_feat = torch.cat([frame_feat, global_feat], dim=-1)
                intermediate_outputs.append(combined_feat)

        # Remove FLAME tokens from final output
        if self.use_flame_tokens and flame_tokens is not None:
            last_cond = last_cond[:, :, 1:, :]  # Remove FLAME tokens: [B, N_input, HW, C]

        x = self.frame_attn.norm(x)

        if self.post_fusion:
            x = x.reshape(B, -1, C)
            
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.post_fusion_attn),
                    x,
                    **ckpt_kwargs,
                )
            else:
                x = self.post_fusion_attn(x)
            
            x = x.reshape(B, N_input, N_points, C)

        return x, last_cond, intermediate_outputs
