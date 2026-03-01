import torch
from FastAvatar.utils.camera_utils import c2w_intri_to_vec


def check_and_fix_inf_nan(tensor, name=""):
    """Check for nan/inf in tensor and replace with 0.0."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def compute_camera_loss(
    pred_dict,              # predictions dict, contains 'pose_enc_list'
    batch_data,             # batch data containing 'target_c2ws' and 'target_intrs'
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    feature_image_res=504,  # resolution of feature maps used for prediction (504x504)
    gt_image_res=512,
    **kwargs
):
    """
    Compute camera loss across all prediction stages.
    """
    pred_pose_encodings = pred_dict['pose_enc_list']

    # Ground truth: select frames corresponding to predictions
    if 'c2ws' not in batch_data or 'target_c2ws' not in batch_data:
        raise ValueError(f"Missing GT camera data. "
                         f"Need 'c2ws' and 'target_c2ws', found: {list(batch_data.keys())}")

    # Use only base frames for camera loss as requested
    base_frames = min(16, batch_data['c2ws'].shape[1])
    
    # Trim predictions to match base frames
    pred_pose_encodings = [p[:, :base_frames] for p in pred_pose_encodings]
    
    # Select only base frames for GT
    gt_c2ws = batch_data['c2ws'][:, :base_frames]
    gt_intrs = batch_data['intrs'][:, :base_frames]
    
    # Scale GT intrinsics to match feature resolution
    # GT intrinsics are at 512 resolution (verified: cx=256 → res=512)
    # No scaling needed since feature_image_res == gt_image_res == 512
    scale_factor = feature_image_res / gt_image_res
    gt_intrs_scaled = gt_intrs.clone()
    gt_intrs_scaled[..., 0, 0] *= scale_factor  # fx
    gt_intrs_scaled[..., 1, 1] *= scale_factor  # fy
    gt_intrs_scaled[..., 0, 2] *= scale_factor  # cx
    gt_intrs_scaled[..., 1, 2] *= scale_factor  # cy
    gt_pose_encoding = c2w_intri_to_vec(gt_c2ws, gt_intrs_scaled)

    n_stages = len(pred_pose_encodings)
    total_loss_T = total_loss_R = total_loss_FL = 0

    for stage_idx in range(n_stages):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
            pred_pose_stage.clone(),
            gt_pose_encoding.clone(),
            loss_type=loss_type,
            focal_scale=feature_image_res  # Use feature resolution for normalization
        )
        
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }


def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1", focal_scale=1024.0):
    """
    Compute losses for a single stage of pose encoding.
    
    IMPORTANT: Focal length normalization
    - Network predicts focal length in normalized space [0, ~2.5]
    - GT focal length is in pixel space (~1123px for 512 resolution)
    - We normalize GT by focal_scale to match prediction space
    - This ensures loss computation is in consistent scale
    """
    
    # Extract focal lengths
    pred_fl = pred_pose_enc[..., 7:9]  # Normalized: ~0.5 initially, should learn to ~2.2
    gt_fl = gt_pose_enc[..., 7:9]       # Pixels: ~1123px for 512 resolution
    
    # Normalize GT focal length to match network prediction space
    # Example: 1123px / 512 = 2.19 (normalized)
    gt_fl_normalized = gt_fl / focal_scale
    
    # Build normalized gt_pose_enc with corrected focal length
    gt_pose_enc_normalized = gt_pose_enc.clone()
    gt_pose_enc_normalized[..., 7:9] = gt_fl_normalized
    
    # Compute losses with matched scales
    if loss_type == "l1":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc_normalized[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc_normalized[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:9] - gt_pose_enc_normalized[..., 7:9]).abs()
    elif loss_type == "l2":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc_normalized[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc_normalized[..., 3:7]).norm(dim=-1, keepdim=True)
        loss_FL = (pred_pose_enc[..., 7:9] - gt_pose_enc_normalized[..., 7:9]).norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL
