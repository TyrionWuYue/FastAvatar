import numpy as np
import torch
import torch.nn as nn
import logging
from torch.utils.checkpoint import checkpoint
from FastAvatar.models.utils.block import MemEffBasicBlock
from FastAvatar.models.utils.pos_embed import RoPE2D, get_1d_sincos_pos_embed_from_grid
from FastAvatar.models.utils.cross_attn import SD3JointTransformerBlock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameAttn(nn.Module):
    """
    Frame attention module based on SD3JointTransformerBlock.
    """

    def __init__(self, num_layers: int = 10, num_heads: int = 16,
                 inner_dim: int = 1024, cond_dim: int = None,
                 eps: float = 1e-6):
        super().__init__()
        
        # Create layers using SD3JointTransformerBlock
        self.layers = nn.ModuleList([
            SD3JointTransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                qk_norm="rms_norm",
                eps=eps
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
                 aa_order: list = ["global", "frame"],
                 patch_start_idx: int = 0,
                 if_framepack: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.aa_order = aa_order
        self.if_framepack = if_framepack
        self.rope2d = RoPE2D(freq=100.0)
        self.patch_start_idx = patch_start_idx

        # Register frame-level index embedding
        self.register_buffer(
            "frame_idx_emb",
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(inner_dim, np.arange(1000))).float(),
            persistent=False,
        )

        
        # Calculate layer distribution based on aa_order
        if aa_order == ["frame", "global"]:
            frame_layers = num_layers
            global_layers = num_layers - 1
        else:  # aa_order == ["global", "frame"]
            frame_layers = num_layers
            global_layers = num_layers
        
        # Frame attention (using FrameAttn with SD3JointTransformerBlock)
        self.frame_attn = FrameAttn(
            num_layers=frame_layers,
            num_heads=num_heads,
            inner_dim=inner_dim,
            cond_dim=cond_dim,
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
            for _ in range(global_layers)
        ])

        # Specialized Frame attention for compressed frames
        if self.if_framepack:
            self.compressed_frame_attn = FrameAttn(
                num_layers=frame_layers,
                num_heads=num_heads,
                inner_dim=inner_dim,
                cond_dim=cond_dim,
                eps=eps,
            )
        else:
            self.compressed_frame_attn = None
    
    def forward_frame_attn(self, layer_idx: int, x: torch.Tensor, cond: torch.Tensor, B: int, N_frame: int, N_points: int, HW: int, C: int):
        """
        Forward pass for frame attention.
        
        Args:
            layer_idx: Index of the transformer layer
            x: Point features [B, N_input, N_points, C]
            cond: Image features [B, N_input, HW, C]
        """
        if x.shape != (B, N_frame, N_points, C):
            x = x.reshape(B, N_frame, N_points, C)
        if cond.shape != (B, N_frame, HW, C):
            cond = cond.reshape(B, N_frame, HW, C)

        x_reshaped = x.reshape(B * N_frame, N_points, C)
        cond_reshaped = cond.reshape(B * N_frame, HW, C)
        
        if self.training and self.gradient_checkpointing:
            x_out, cond_out = checkpoint(
                self.frame_attn.layers[layer_idx],
                x_reshaped,
                cond_reshaped,
                use_reentrant=False,
            )
        else:
            x_out, cond_out = self.frame_attn.layers[layer_idx](
                hidden_states=x_reshaped,
                encoder_hidden_states=cond_reshaped,
                temb=None,
            )

        # Reshape outputs back to [B, N_frame, N_points, C] and [B, N_frame, HW, C]
        x_out = x_out.reshape(B, N_frame, N_points, C)
        cond_out = cond_out.reshape(B, N_frame, HW, C)

        return x_out, cond_out

    def forward_compressed_frame_attn(self, layer_idx: int, x: torch.Tensor, cond: torch.Tensor, B: int, N_frame: int, N_points: int, HW: int, C: int):
        """
        Forward pass for specialized compressed frame attention.
        """
        if x.shape != (B, N_frame, N_points, C):
            x = x.reshape(B, N_frame, N_points, C)
        if cond.shape != (B, N_frame, HW, C):
            cond = cond.reshape(B, N_frame, HW, C)

        x_reshaped = x.reshape(B * N_frame, N_points, C)
        cond_reshaped = cond.reshape(B * N_frame, HW, C)
        
        # Fallback to standard frame_attn if specialized one is not initialized
        attn_module = self.compressed_frame_attn if self.compressed_frame_attn is not None else self.frame_attn

        if self.training and self.gradient_checkpointing:
            x_out, cond_out = checkpoint(
                attn_module.layers[layer_idx],
                x_reshaped,
                cond_reshaped,
                use_reentrant=False,
            )
        else:
            x_out, cond_out = attn_module.layers[layer_idx](
                hidden_states=x_reshaped,
                encoder_hidden_states=cond_reshaped,
                temb=None,
            )

        # Reshape outputs back to [B, N_frame, N_points, C] and [B, N_frame, HW, C]
        x_out = x_out.reshape(B, N_frame, N_points, C)
        cond_out = cond_out.reshape(B, N_frame, HW, C)

        return x_out, cond_out
    

    def forward_global_attn(self, layer_idx: int, cond: torch.Tensor, combined_positions: torch.Tensor):
        """
        Forward pass for global attention.
        """
        # cond is already in shape [B, total_tokens, C] from forward()
        # combined_positions is [B, total_tokens, 2]
        
        if self.training and self.gradient_checkpointing:
            cond_out = checkpoint(
                self.global_attn_layers[layer_idx],
                cond,
                combined_positions,
                use_reentrant=False,
            )
        else:
            cond_out = self.global_attn_layers[layer_idx](
                cond,
                pos=combined_positions,
            )
        
        return cond_out
    

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None,
                compressed_x: torch.Tensor = None, compressed_cond: torch.Tensor = None,
                spatial_compression: int = None) -> torch.Tensor:
        """
        Alternating cross attention.
        Args:
            x: [B, N_input, N_points, C] - point features for base frames
            cond: [B, N_input, H*W, C] - image features for base frames
            compressed_x: [B, 1, N_points, C] - point features for compressed frame (optional)
            compressed_cond: [B, 1, compressed_tokens, C] - image features for compressed frame (optional)
            spatial_compression: int - Spatial compression ratio used for compressed frames (optional)
        Returns:
            torch.Tensor: Processed point features [B, N_output, N_points, C]
        """
        B, N_input, N_points, C = x.shape
        B2, N_input2, HW, C2 = cond.shape
        assert B == B2 and C == C2 and N_input == N_input2, "Batch, channel and frame dimensions must match"
        
        has_compressed = self.if_framepack and compressed_cond is not None and compressed_x is not None

        # ============================================================================
        # Step 1: Preprocess base frames
        # ============================================================================
        cond = self.frame_attn.linear_cond_proj(cond) if hasattr(self.frame_attn, 'linear_cond_proj') else cond

        frame_indices = torch.arange(N_input, device=cond.device)
        frame_emb = self.frame_idx_emb[frame_indices]  # [N_input, C]
        cond = cond + frame_emb[None, :, None, :]      # [B, N_input, HW, C]

        H = W = int(HW ** 0.5)
        total_tokens_per_frame = cond.shape[2]

        patch_positions = torch.stack(torch.meshgrid(
            torch.arange(H, device=cond.device),
            torch.arange(W, device=cond.device),
            indexing='ij'), dim=-1).reshape(-1, 2)  # [H*W, 2]

        patch_positions = patch_positions.repeat(N_input, 1)  # [N_input*HW, 2]
        base_combined_positions = patch_positions[None, :, :].expand(B, N_input * HW, 2)  # [B, N_input*HW, 2]

        # ============================================================================
        # Step 2: Preprocess compressed frame (if exists)
        # ============================================================================
        compressed_combined_positions = None
        if has_compressed:
            HW_compressed = compressed_cond.shape[2]  # compressed frame token count
            # Use specialized projection for compressed features if available, else fallback to base
            proj_module = self.compressed_frame_attn if self.compressed_frame_attn is not None else self.frame_attn
            compressed_cond = proj_module.linear_cond_proj(compressed_cond) if hasattr(proj_module, 'linear_cond_proj') else compressed_cond

            # Calculate compressed dimensions and frame count
            spatial_compression_ratio = spatial_compression if spatial_compression is not None else 8
            pad_h = (spatial_compression_ratio - (H % spatial_compression_ratio)) % spatial_compression_ratio
            pad_w = (spatial_compression_ratio - (W % spatial_compression_ratio)) % spatial_compression_ratio
            H_compressed = (H + pad_h) // spatial_compression_ratio
            W_compressed = (W + pad_w) // spatial_compression_ratio
            N_compressed_frames = HW_compressed // (H_compressed * W_compressed)

            # Reshape compressed_cond to per-frame format first
            compressed_cond = compressed_cond.reshape(B, N_compressed_frames, H_compressed * W_compressed, C)

            # Generate frame embeddings for compressed frames
            compressed_frame_indices = torch.arange(N_input, N_input + N_compressed_frames, device=compressed_cond.device)
            compressed_frame_emb = self.frame_idx_emb[compressed_frame_indices]  # [N_compressed_frames, C]
            compressed_cond = compressed_cond + compressed_frame_emb[None, :, None, :]  # [B, N_compressed_frames, H_compressed*W_compressed, C]

            # Calculate the number of tokens per compressed frame
            compressed_tokens_per_frame = H_compressed * W_compressed

            # Generate compressed patch coordinates as averages of their constituent original patches
            # This mirrors FramePack's frequency averaging through 3D pooling
            y_comp_indices = torch.arange(H_compressed, device=compressed_cond.device, dtype=torch.float)
            x_comp_indices = torch.arange(W_compressed, device=compressed_cond.device, dtype=torch.float)

            # Generate grid
            grid_y, grid_x = torch.meshgrid(y_comp_indices, x_comp_indices, indexing='ij')

            # For each compressed patch (i,j), compute average coordinate of the 8x8 original patches it represents
            # Average = (sum of original coords) / count = i*8 + (0+1+...+7)/8 = i*8 + 3.5
            offset = 1.0 if self.patch_start_idx > 0 else 0.0
            avg_y_coords = grid_y * spatial_compression_ratio + (spatial_compression_ratio - 1) / 2.0 + offset
            avg_x_coords = grid_x * spatial_compression_ratio + (spatial_compression_ratio - 1) / 2.0 + offset

            spatial_positions = torch.stack([avg_y_coords.flatten(), avg_x_coords.flatten()], dim=1)
            compressed_positions = spatial_positions.repeat(N_compressed_frames, 1)
            compressed_positions = compressed_positions[None, :, :].expand(B, N_compressed_frames * H_compressed * W_compressed, 2)
            
            # Calculate how many extra tokens were actually added to compressed frames
            num_extra_compressed = compressed_tokens_per_frame - H_compressed * W_compressed
            if num_extra_compressed > 0:
                # Handle extra positions for compressed frames (FLAME/Camera tokens)
                compressed_positions_per_frame = compressed_positions.reshape(B, N_compressed_frames, H_compressed * W_compressed, 2)
                extra_positions = torch.zeros(B, N_compressed_frames, num_extra_compressed, 2, device=compressed_positions.device, dtype=torch.long)
                combined_per_frame = torch.cat([extra_positions, compressed_positions_per_frame], dim=2)
                compressed_combined_positions = combined_per_frame.reshape(B, N_compressed_frames * compressed_tokens_per_frame, 2)
            else:
                compressed_combined_positions = compressed_positions

        # ============================================================================
        # Step 3: Apply transformer layers
        # ============================================================================
        frame_layer_idx = 0
        global_layer_idx = 0
        
        for _ in range(self.num_layers):
            for attn_type in self.aa_order:
                if attn_type == 'frame':
                    # Frame attention: process base frames and compressed frame separately
                    x, cond = self.forward_frame_attn(frame_layer_idx, x, cond, B, N_input, N_points, total_tokens_per_frame, C)
                    
                    if has_compressed:
                        # Reshape compressed_cond to match 1-frame format for frame attention
                        total_compressed_tokens = N_compressed_frames * (self.patch_start_idx + H_compressed * W_compressed)
                        compressed_cond_reshaped = compressed_cond.reshape(B, 1, total_compressed_tokens, C)

                        compressed_x, compressed_cond_out = self.forward_compressed_frame_attn(
                            frame_layer_idx, compressed_x, compressed_cond_reshaped,
                            B, 1, N_points, total_compressed_tokens, C
                        )

                        # Reshape back to multi-frame format for global attention
                        compressed_cond = compressed_cond_out.reshape(B, N_compressed_frames, self.patch_start_idx + H_compressed * W_compressed, C)
                    
                    frame_layer_idx += 1
                    
                elif attn_type == 'global':
                    if global_layer_idx < len(self.global_attn_layers):
                        # Global attention with compressed frames
                        if has_compressed:
                            # Concatenate all frames
                            base_cond_flat = cond.reshape(B, N_input * total_tokens_per_frame, C)
                            compressed_cond_flat = compressed_cond.reshape(B, N_compressed_frames * compressed_tokens_per_frame, C)
                            all_cond_flat = torch.cat([base_cond_flat, compressed_cond_flat], dim=1)
                            all_positions = torch.cat([base_combined_positions, compressed_combined_positions], dim=1)

                            # Apply global attention
                            all_cond_out = self.forward_global_attn(global_layer_idx, all_cond_flat, all_positions)

                            # Split back to base and compressed
                            base_tokens = N_input * total_tokens_per_frame
                            cond = all_cond_out[:, :base_tokens].reshape(B, N_input, total_tokens_per_frame, C)
                            compressed_cond = all_cond_out[:, base_tokens:].reshape(B, N_compressed_frames, compressed_tokens_per_frame, C)
                        else:
                            # Global attention on base frames only
                            cond_flat = cond.reshape(B, N_input * total_tokens_per_frame, C)
                            cond_flat = self.forward_global_attn(global_layer_idx, cond_flat, base_combined_positions)
                            cond = cond_flat.reshape(B, N_input, total_tokens_per_frame, C)
                        
                        global_layer_idx += 1
        
        # ============================================================================
        # Step 4: Prepare output
        # ============================================================================
        x = x.reshape(B, N_input, N_points, C)

        # Add compressed frame to output
        if has_compressed:
            x = torch.cat([x, compressed_x], dim=1)

        x = self.frame_attn.norm(x)

        return x
