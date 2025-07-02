# Copyright (c) 2023-2024, Zexin He
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
from einops import rearrange

__all__ = ['PixelLoss', 'ConfPixelLoss']


class PixelLoss(nn.Module):
    """
    Pixel-wise loss between two images.
    """

    def __init__(self, option: str = 'mse'):
        super().__init__()
        self.loss_fn = self._build_from_option(option)

    @staticmethod
    def _build_from_option(option: str, reduction: str = 'none'):
        if option == 'mse':
            return nn.MSELoss(reduction=reduction)
        elif option == 'l1':
            return nn.L1Loss(reduction=reduction)
        else:
            raise ValueError(f'Invalid option: {option}')
        
    @torch.compile
    def forward(self, x, y, mask=None, EPS=1e-6):
        """
        Assume images are channel first.

        Args:
            x: [N, M, C, H, W] - predicted images
            y: [N, M, C, H, W] - target images
            mask: [N, M, 1, H, W] or [N, M, H, W] - optional mask where True indicates valid regions
                  If None, all pixels are considered valid
        
        Returns:
            Mean-reduced pixel loss across batch, only considering masked regions
        """
        N, M, C, H, W = x.shape
        x = rearrange(x, "n m c h w -> (n m) c h w")
        y = rearrange(y, "n m c h w -> (n m) c h w")
        
        # Calculate per-pixel loss
        image_loss = self.loss_fn(x, y)  # [N*M, C, H, W]
        
        if mask is not None:
            # Handle mask shape: [N, M, 1, H, W] or [N, M, H, W]
            if mask.shape[2] == 1:
                mask = mask.squeeze(2)  # Remove channel dimension if present
            mask = rearrange(mask, "n m h w -> (n m) h w")  # [N*M, H, W]
            
            # Apply mask to loss
            # Expand mask to match loss dimensions: [N*M, H, W] -> [N*M, C, H, W]
            mask_expanded = mask.unsqueeze(1).expand(-1, C, -1, -1)
            
            # Zero out loss for masked-out regions
            image_loss = image_loss * mask_expanded
            
            # Calculate mean only over valid regions
            # Sum over all dimensions except batch, then divide by number of valid pixels
            valid_pixels = mask_expanded.sum(dim=[1, 2, 3])  # [N*M]
            total_loss = image_loss.sum(dim=[1, 2, 3])  # [N*M]
            image_loss = total_loss / (valid_pixels + EPS)  # [N*M]
        else:
            # Original behavior: mean over all pixels
            image_loss = image_loss.mean(dim=[1, 2, 3])  # [N*M]
        
        # Reshape and average over frames and batch
        batch_loss = image_loss.reshape(N, M).mean(dim=1)  # [N]
        all_loss = batch_loss.mean()  # scalar
        return all_loss


class ConfPixelLoss(nn.Module):
    """
    Confidence-weighted pixel-wise loss, inspired by Dust3r implementation.
    
    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 
        
        alpha: hyperparameter to control confidence regularization
    """

    def __init__(self, option: str = 'mse', alpha: float = 0.05):
        super().__init__()
        assert alpha > 0, "alpha must be positive"
        self.alpha = alpha
        self.pixel_loss = PixelLoss(option)

    def get_conf_log(self, x):
        """Get confidence and its log, ensuring numerical stability"""
        conf = x  # Use confidence directly as output by DPT
        log_conf = torch.log(conf + 1e-8)  # Add small epsilon to prevent log(0)
        return conf, log_conf

    @torch.compile
    def forward(self, x, y, confidence, mask=None, EPS=1e-6):
        """
        Confidence-weighted pixel-wise loss with regularization.
        
        Args:
            x: [N, M, C, H, W] - predicted images
            y: [N, M, C, H, W] - target images
            confidence: [N, M, 1, H, W] or [N, M, H, W] - confidence map (raw logits)
            mask: [N, M, 1, H, W] or [N, M, H, W] - optional mask where True indicates valid regions
                  If None, all pixels are considered valid
            EPS: float - small value to prevent division by zero
        
        Returns:
            Confidence-weighted pixel loss with regularization
        """
        N, M, C, H, W = x.shape
        x = rearrange(x, "n m c h w -> (n m) c h w")
        y = rearrange(y, "n m c h w -> (n m) c h w")
        
        # Handle confidence shape: [N, M, 1, H, W] or [N, M, H, W]
        if confidence.shape[2] == 1:
            confidence = confidence.squeeze(2)  # Remove channel dimension if present
        confidence = rearrange(confidence, "n m h w -> (n m) h w")  # [N*M, H, W]
        
        # Calculate per-pixel loss using the base pixel loss
        image_loss = self.pixel_loss.loss_fn(x, y)  # [N*M, C, H, W]
        
        # Get confidence and its log
        conf, log_conf = self.get_conf_log(confidence)  # [N*M, H, W]
        
        # Apply confidence weighting
        # Expand confidence to match loss dimensions: [N*M, H, W] -> [N*M, C, H, W]
        conf_expanded = conf.unsqueeze(1).expand(-1, C, -1, -1)
        log_conf_expanded = log_conf.unsqueeze(1).expand(-1, C, -1, -1)
        
        if mask is not None:
            # Handle mask shape: [N, M, 1, H, W] or [N, M, H, W]
            if mask.shape[2] == 1:
                mask = mask.squeeze(2)  # Remove channel dimension if present
            mask = rearrange(mask, "n m h w -> (n m) h w")  # [N*M, H, W]
            mask_expanded = mask.unsqueeze(1).expand(-1, C, -1, -1)
            
            # Apply mask to all tensors
            image_loss = image_loss * mask_expanded
            conf_expanded = conf_expanded * mask_expanded
            log_conf_expanded = log_conf_expanded * mask_expanded
            
            # Compute confidence-weighted loss with regularization
            # conf_loss = loss * conf - alpha * log(conf)
            weighted_loss = image_loss * conf_expanded - self.alpha * log_conf_expanded
            
            # Calculate mean only over valid regions
            total_loss = weighted_loss.sum(dim=[1, 2, 3])  # [N*M]
            valid_pixels = mask_expanded.sum(dim=[1, 2, 3])  # [N*M]
            image_loss = total_loss / (valid_pixels + EPS)  # [N*M]
        else:
            # Compute confidence-weighted loss with regularization
            # conf_loss = loss * conf - alpha * log(conf)
            weighted_loss = image_loss * conf_expanded - self.alpha * log_conf_expanded
            
            # Calculate mean over all pixels
            image_loss = weighted_loss.mean(dim=[1, 2, 3])  # [N*M]
        
        # Reshape and average over frames and batch
        batch_loss = image_loss.reshape(N, M).mean(dim=1)  # [N]
        all_loss = batch_loss.mean()  # scalar
        return all_loss
