"""
FramePack utilities for compressing long video sequences.
Adapted from HunyuanVideo implementation for FastAvatar.

Compression Levels (Spatial Compression Only):
- Level 2: 2x spatial compression
- Level 4: 4x spatial compression
- Level 8: 8x spatial compression

Usage in train.yaml:
    framepack_compression_level: 8  # Choose 2, 4, or 8
"""
import torch
import torch.nn as nn


def pad_for_2d_conv(x, kernel_size):
    """Pad tensor for 2D convolution to ensure divisibility."""
    b, c, h, w = x.shape
    ph, pw = kernel_size
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='replicate')


class FramePackCompressor(nn.Module):
    """
    FramePack compressor for video sequences.
    No temporal compression, only spatial compression using 2D convolution.
    """
    def __init__(self,
                 in_channels: int,
                 inner_dim: int,
                 compression_level: int = 8):
        """
        Args:
            in_channels: Input feature channels (e.g., encoder_feat_dim)
            inner_dim: Output feature dimension
            compression_level: Spatial compression factor (2, 4, or 8)
                - 2: 2x2 spatial compression
                - 4: 4x4 spatial compression
                - 8: 8x8 spatial compression
        """
        super().__init__()
        self.compression_level = compression_level

        if compression_level == 2:
            # 2x2 spatial compression
            self.proj = nn.Conv2d(
                in_channels,
                inner_dim,
                kernel_size=(2, 2),
                stride=(2, 2)
            )
        elif compression_level == 4:
            # 4x4 spatial compression
            self.proj = nn.Conv2d(
                in_channels,
                inner_dim,
                kernel_size=(4, 4),
                stride=(4, 4)
            )
        elif compression_level == 8:
            # 8x8 spatial compression
            self.proj = nn.Conv2d(
                in_channels,
                inner_dim,
                kernel_size=(8, 8),
                stride=(8, 8)
            )
        else:
            raise ValueError(f"Unsupported compression_level: {compression_level}. Supported: 2, 4, 8")

    @property
    def temporal_compression(self):
        """Return the temporal compression ratio (always 1)."""
        return 1

    @property
    def spatial_compression(self):
        """Return the spatial compression ratio."""
        return self.compression_level

    def forward(self, features):
        """
        Compress video features using FramePack (spatial only).

        Args:
            features: [B, N_frames, H, W, C] - Image features from DINO encoder

        Returns:
            compressed_features: [B, N_frames, H_compressed, W_compressed, C]
                - N_frames remains unchanged (no temporal compression)
                - Spatial dimensions compressed by compression_level
        """
        B, N_frames, H, W, C = features.shape

        # Reshape to process all frames at once: [B*N_frames, C, H, W]
        features_2d = features.permute(0, 1, 4, 2, 3).reshape(B * N_frames, C, H, W)

        # Pad if necessary for the compression level
        features_2d = pad_for_2d_conv(features_2d, (self.compression_level, self.compression_level))

        # Apply 2D convolution for spatial compression
        compressed_2d = self.proj(features_2d)  # [B*N_frames, inner_dim, H_compressed, W_compressed]

        # Reshape back to [B, N_frames, H_compressed, W_compressed, inner_dim]
        compressed_features = compressed_2d.reshape(B, N_frames, compressed_2d.shape[2], compressed_2d.shape[3], -1)

        return compressed_features, self.spatial_compression

