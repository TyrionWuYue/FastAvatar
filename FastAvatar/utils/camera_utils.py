import torch
import torch.nn.functional as F


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) with zero subgradient at 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Standardize quaternion to have non-negative real part (W-first).
    Matches FastAvatar convention.
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to rotation matrix.
    Args:
        quat: [..., 4]
    Returns:
        rot_mat: [..., 3, 3]
    """
    # Keep the original project formula exactly for consistency
    w, x, y, z = quat.unbind(-1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    rot_mat = torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), 1 - 2 * (x2 + z2), 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (x2 + y2)
    ], dim=-1).reshape(quat.shape[:-1] + (3, 3))
    
    return rot_mat


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    Vectorized implementation to avoid Boolean Masking shape errors.
    Uses signs from the original project code (Conjugate logic).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22
            ], dim=-1
        )
    )
    
    quat_candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m12 - m21, m20 - m02, m01 - m10], dim=-1), # W largest
            torch.stack([m12 - m21, q_abs[..., 1] ** 2, m10 + m01, m20 + m02], dim=-1), # X largest
            torch.stack([m20 - m02, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], dim=-1), # Y largest
            torch.stack([m01 - m10, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1), # Z largest
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    denom = 2.0 * q_abs[..., None].max(flr)
    quat_candidates = quat_candidates / denom

    idx = q_abs.argmax(dim=-1)
    mask = F.one_hot(idx, num_classes=4) > 0.5
    out = quat_candidates[mask, :].reshape(batch_dim + (4,))

    return standardize_quaternion(out)


def c2w_intri_to_vec(c2ws, intrinsics):
    """Convert c2w and intrinsics to a compact 9-dim vector."""
    R = c2ws[:, :, :3, :3]  # BxSx3x3
    T = c2ws[:, :, :3, 3]   # BxSx3

    quat = mat_to_quat(R)
    
    # Extract fl_x, fl_y from intrinsics
    fl_x = intrinsics[:, :, 0, 0]
    fl_y = intrinsics[:, :, 1, 1]
    
    vec = torch.cat([T, quat, fl_x[..., None], fl_y[..., None]], dim=-1).float()
    return vec


def vec_to_c2w_intri(vec, hw):
    """Convert a 9-dim vector back to c2w and intrinsics.
    """
    T = vec[..., :3]
    quat = vec[..., 3:7]
    fl_x_normalized = vec[..., 7]
    fl_y_normalized = vec[..., 8]

    R = quat_to_mat(quat)
    c2w = torch.cat([R, T[..., None]], dim=-1)
    
    # Ensure c2w is 4x4
    B, S = vec.shape[:2]
    last_row = torch.tensor([[[0, 0, 0, 1]]], device=vec.device).expand(B, S, 1, 4).float()
    c2w = torch.cat([c2w, last_row], dim=2)

    H, W = hw
    fl_x = fl_x_normalized * W
    fl_y = fl_y_normalized * H
    
    intrinsics = torch.zeros(vec.shape[:2] + (4, 4), device=vec.device, dtype=vec.dtype)
    intrinsics[..., 0, 0] = fl_x
    intrinsics[..., 1, 1] = fl_y
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0
    intrinsics[..., 3, 3] = 1.0
    
    return c2w, intrinsics
