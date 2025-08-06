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


class FLAMEPoseEncoder(nn.Module):
    """
    FLAME pose encoder that encodes FLAME parameters into tokens.
    Set any parameter dimension to 0 to disable encoding that parameter.
    
    Args:
        expr_dim: Expression parameter dimension (0 to disable, default: 100)
        rotation_dim: Rotation parameter dimension (0 to disable, default: 3)
        neck_pose_dim: Neck pose parameter dimension (0 to disable, default: 3)
        jaw_pose_dim: Jaw pose parameter dimension (0 to disable, default: 3)
        eyes_pose_dim: Eyes pose parameter dimension (0 to disable, default: 6)
        translation_dim: Translation parameter dimension (0 to disable, default: 3)
        hidden_dim: Hidden dimension for MLP (default: 256)
        output_dim: Output token dimension (default: 1024)
        dropout: Dropout rate (default: 0.0)
    """
    def __init__(self, 
                 expr_dim: int = 100,
                 rotation_dim: int = 3,
                 neck_pose_dim: int = 3,
                 jaw_pose_dim: int = 3,
                 eyes_pose_dim: int = 6,
                 translation_dim: int = 3,
                 hidden_dim: int = 256,
                 output_dim: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        
        # Store parameter dimensions (only include non-zero dimensions)
        self.param_dims = {}
        if expr_dim > 0:
            self.param_dims['expr'] = expr_dim
        if rotation_dim > 0:
            self.param_dims['rotation'] = rotation_dim
        if neck_pose_dim > 0:
            self.param_dims['neck_pose'] = neck_pose_dim
        if jaw_pose_dim > 0:
            self.param_dims['jaw_pose'] = jaw_pose_dim
        if eyes_pose_dim > 0:
            self.param_dims['eyes_pose'] = eyes_pose_dim
        if translation_dim > 0:
            self.param_dims['translation'] = translation_dim
        
        # Calculate total input dimension (sum of enabled parameters)
        total_input_dim = sum(self.param_dims.values())
        
        if total_input_dim == 0:
            raise ValueError("At least one FLAME parameter dimension must be greater than 0")
        
        # MLP encoder
        self.encoder = MLP(
            in_features=total_input_dim,
            hidden_features=hidden_dim,
            out_features=output_dim,
            drop=dropout
        )
    
    def forward(self, flame_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode FLAME parameters into tokens.
        Only encodes parameters that are enabled (dim > 0 in config).
        
        Args:
            flame_params: Dictionary containing FLAME parameters
                - expr: [B, N_input, expr_dim] (if enabled)
                - rotation: [B, N_input, rotation_dim] (if enabled)
                - neck_pose: [B, N_input, neck_pose_dim] (if enabled)
                - jaw_pose: [B, N_input, jaw_pose_dim] (if enabled)
                - eyes_pose: [B, N_input, eyes_pose_dim] (if enabled)
                - translation: [B, N_input, translation_dim] (if enabled)
        
        Returns:
            flame_tokens: [B, N_input, output_dim]
        """
        # Extract only enabled parameters
        device = next(self.parameters()).device
        param_list = []
        
        for param_name, param_dim in self.param_dims.items():
            if param_name in flame_params:
                param = flame_params[param_name]
            else:
                # Create zero tensor for missing parameters
                param = torch.zeros(1, 1, param_dim, device=device)
            param_list.append(param)
        
        # Concatenate enabled parameters
        combined = torch.cat(param_list, dim=-1)
        
        # Encode to tokens
        flame_tokens = self.encoder(combined)
        
        return flame_tokens


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
        c2w_flat = c2w.reshape(B, N, 16)
        intrinsic_flat = intrinsic.reshape(B, N, 16)
        
        # Concatenate: [B, N, 32]
        combined = torch.cat([c2w_flat, intrinsic_flat], dim=-1)
        
        # Encode to tokens
        camera_tokens = self.encoder(combined)
        
        return camera_tokens