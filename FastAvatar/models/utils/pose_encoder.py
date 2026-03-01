import torch
import torch.nn as nn
from typing import Optional, Callable, Dict


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CameraPoseEncoder(nn.Module):
    """
    Camera pose encoder that encodes camera parameters (c2w and intrinsic) into tokens.
    Simply flattens and concatenates c2w and intrinsic matrices, then passes through MLP.
    """
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 output_dim: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        
        # Input dimension: c2w (16) + intrinsic (16) = 32
        input_dim = 32
        
        # MLP encoder
        self.encoder = MLP(
            in_features=input_dim,
            hidden_features=hidden_dim,
            out_features=output_dim,
            drop=dropout
        )
    
    def forward(self, c2w: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
        """
        Encode camera parameters into tokens.
        
        Args:
            c2w: [B, N_input, 4, 4] - Camera-to-world matrices
            intrinsic: [B, N_input, 4, 4] - Intrinsic matrices
        
        Returns:
            camera_tokens: [B, N_input, output_dim]
        """
        B, N = c2w.shape[:2]
        
        # Flatten matrices: [B, N, 4, 4] -> [B, N, 16]
        c2w_flat = c2w.float().reshape(B, N, 16)
        intrinsic_flat = intrinsic.float().reshape(B, N, 16)
        
        # Concatenate: [B, N, 32]
        combined = torch.cat([c2w_flat, intrinsic_flat], dim=-1)
        
        # Encode to tokens
        camera_tokens = self.encoder(combined)
        
        return camera_tokens