# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

__all__ = ['TrackingLoss']


def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """
    mask = mask.expand_as(input)
    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (1e-6 + denom)
    return mean


def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


class TrackingLoss(nn.Module):
    """
    Tracking loss module that combines flow loss, visibility loss and probability loss.
    """
    def __init__(
        self,
        gamma: float = 0.8,
        add_huber_loss: bool = True,
        loss_only_for_visible: bool = True,
        expected_dist_thresh: float = 12.0,
    ):
        """
        Initialize the tracking loss module.

        Args:
            gamma (float): Weight decay factor for multi-scale predictions. Default: 0.8
            add_huber_loss (bool): Whether to use Huber loss instead of L1 loss. Default: False
            loss_only_for_visible (bool): Whether to compute loss only for visible points. Default: False
            expected_dist_thresh (float): Threshold for probability loss. Default: 12.0
        """
        super().__init__()
        self.gamma = gamma
        self.add_huber_loss = add_huber_loss
        self.loss_only_for_visible = loss_only_for_visible
        self.expected_dist_thresh = expected_dist_thresh

    def sequence_loss(self, flow_preds, flow_gt, valids, vis=None):
        """Loss function defined over sequence of flow predictions
        Args:
            flow_preds: Predicted flow tensor with shape [B, S, N, 2]
            flow_gt: Ground truth flow tensor with shape [B, S, N, 2]
            valids: Validity mask with shape [B, S, N]
            vis: Optional visibility mask with shape [B, S, N]
        """
        B, S, N, D = flow_gt.shape  # S is sequence length (frames)
        B, S2, N = valids.shape
        assert S == S2
        
        if self.add_huber_loss:
            loss = huber_loss(flow_preds, flow_gt, delta=6.0)
        else:
            loss = (flow_preds - flow_gt).abs()  # B, S, N, 2
            
        loss = torch.mean(loss, dim=3)  # B, S, N
        valid_ = valids.clone()
        if self.loss_only_for_visible and vis is not None:
            valid_ = valid_ * vis
            
        return reduce_masked_mean(loss, valid_)

    def sequence_BCE_loss(self, vis_preds, vis_gts):
        """Compute binary cross entropy loss for visibility predictions
        Args:
            vis_preds: Predicted visibility tensor with shape [B, S, N]
            vis_gts: Ground truth visibility tensor with shape [B, S, N]
        """
        loss = F.binary_cross_entropy(vis_preds, vis_gts, reduction="none")  # [B, S, N]
        return torch.mean(loss, dim=[0, 1])  # [N]

    def sequence_prob_loss(self, tracks, confidence, target_points, visibility):
        """Loss for classifying if a point is within pixel threshold of its target.
        Args:
            tracks: Predicted track points with shape [B, S, N, 2]
            confidence: Confidence scores with shape [B, S, N]
            target_points: Target points with shape [B, S, N, 2]
            visibility: Visibility mask with shape [B, S, N]
        """
        err = torch.sum((tracks - target_points) ** 2, dim=-1)  # [B, S, N]
        valid = (err <= self.expected_dist_thresh**2).float()  # [B, S, N]
        logprob = F.binary_cross_entropy(confidence, valid, reduction="none")  # [B, S, N]
        logprob *= visibility
        return torch.mean(logprob, dim=[0, 1])  # [N]

    def forward(
        self,
        flow_preds: List[torch.Tensor],
        flow_gt: List[torch.Tensor],
        valids: List[torch.Tensor],
        vis_preds: Optional[List[torch.Tensor]] = None,
        vis_gts: Optional[List[torch.Tensor]] = None,
        tracks: Optional[List[torch.Tensor]] = None,
        confidence: Optional[List[torch.Tensor]] = None,
        target_points: Optional[List[torch.Tensor]] = None,
        visibility: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute all tracking losses.

        Args:
            flow_preds: List of predicted flow tensors
            flow_gt: List of ground truth flow tensors
            valids: List of validity masks
            vis_preds: List of predicted visibility tensors
            vis_gts: List of ground truth visibility tensors
            tracks: List of tracked point tensors
            confidence: List of confidence tensors
            target_points: List of target point tensors
            visibility: List of visibility masks

        Returns:
            Total loss combining all components
        """
        total_loss = 0.0

        # Add flow loss
        total_loss += self.sequence_loss(flow_preds, flow_gt, valids, visibility)

        # Add visibility loss if provided
        if vis_preds is not None and vis_gts is not None:
            total_loss += self.sequence_BCE_loss(vis_preds, vis_gts).mean()

        # Add probability loss if provided
        if tracks is not None and confidence is not None and target_points is not None and visibility is not None:
            total_loss += self.sequence_prob_loss(tracks, confidence, target_points, visibility).mean()

        return total_loss
