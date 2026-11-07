"""Linear robot skinning (Dr.Robot): pose the canonical Gaussians with the robot's forward kinematics."""
from typing import Tuple

import numpy as np
import rff
import torch
from pytorch_kinematics.chain import Chain
from torch.amp import autocast

from .pt3d import matrix_to_quaternion, quaternion_to_matrix

positional_encoding = rff.layers.PositionalEncoding(sigma=1.0, m=5)

# Rest pose of the canonical Gaussians. Models that store canonical_joint_positions.npy next to their checkpoints
# set this when they are loaded (scene/__init__.py); older models use the joint-limits rule below.
CANONICAL_POSE_OVERRIDE = None

_CANONICAL_POSE_CACHE = {}


def batch_forward_kinematics(chain, pose):
    """Pose (B, J) -> homogeneous transforms of all chain frames (B, F, 4, 4)."""
    tg = chain.forward_kinematics(pose)
    return torch.stack([tg[name]._matrix for name in tg.keys()], dim=1)


def toarray(tensor):
    return tensor.clone().detach().cpu().numpy()


def get_canonical_pose(lower_limits, upper_limits):
    # cached per (override, limits, device) so that several models can live in one process
    key = (None if CANONICAL_POSE_OVERRIDE is None else tuple(np.round(np.asarray(CANONICAL_POSE_OVERRIDE, np.float64), 6).tolist()),
           tuple(np.round(toarray(lower_limits), 6).tolist()), tuple(np.round(toarray(upper_limits), 6).tolist()),
           str(lower_limits.device))
    if key not in _CANONICAL_POSE_CACHE:
        _CANONICAL_POSE_CACHE[key] = _compute_canonical_pose(lower_limits, upper_limits)
    return _CANONICAL_POSE_CACHE[key]


def _compute_canonical_pose(lower_limits, upper_limits):
    if CANONICAL_POSE_OVERRIDE is not None:
        cp = torch.as_tensor(np.asarray(CANONICAL_POSE_OVERRIDE, dtype=np.float32), device=lower_limits.device)
        assert cp.shape[0] == lower_limits.shape[0], f"canonical pose has {cp.shape[0]} joints, chain has {lower_limits.shape[0]}"
        return cp
    # joints whose range excludes 0 sit at the limit closest to 0, all others at 0
    min_limits, max_limits = toarray(lower_limits), toarray(upper_limits)
    canonical_pose = np.zeros(lower_limits.shape[0])
    for i in np.where((min_limits > 0) | (max_limits < 0))[0]:
        canonical_pose[i] = min_limits[i] if min_limits[i] > 0 else max_limits[i]
    return torch.tensor(canonical_pose, dtype=torch.float32, device=lower_limits.device)


def compute_and_cache_zero_pose_data(chain, v_template, lower_limits, upper_limits):
    """Inverse rest-pose transforms (N, F, 4, 4), joint-to-point distances and offsets in the rest pose."""
    zero_pose = get_canonical_pose(lower_limits, upper_limits)
    A_zero = batch_forward_kinematics(chain, zero_pose)
    with autocast('cuda', enabled=False):
        A_zero_inverse = torch.linalg.inv(A_zero)
    zero_joint_positions = A_zero[:, :, :3, 3]
    offsets = zero_joint_positions[:, None] - v_template[:, :, None].detach()  # N, V, frames, 3
    distances = torch.norm(offsets, dim=-1)
    offsets = offsets[:, :, chain.joint_type_indices > 0, :]
    return A_zero_inverse, distances, offsets


def lrs(
    pose: torch.Tensor,
    v_template: torch.Tensor,
    lrs_weights: torch.Tensor,
    chain: Chain,
    pose_normalized: bool = False,
    lrs_model=None,
    rotations=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear robot skinning.

    pose (N, J) joint values (in [-1, 1] if pose_normalized), v_template (N, V, 3) canonical points,
    lrs_model the implicit skinning-weight MLP. Returns the posed points (N, V, 3) and, if rotations (V, 4)
    are given, the posed rotations as quaternions.
    """
    N, J = pose.shape
    V = v_template.shape[1]
    device, dtype = v_template.device, v_template.dtype
    num_frames = len(chain.joint_type_indices)

    lower_limits, upper_limits = chain.get_joint_limits()
    lower_limits = torch.tensor(lower_limits, dtype=dtype, device=device)
    upper_limits = torch.tensor(upper_limits, dtype=dtype, device=device)
    if pose_normalized:
        pose = (pose + 1) * (upper_limits - lower_limits) / 2 + lower_limits

    A = batch_forward_kinematics(chain, pose)
    with autocast('cuda', enabled=False):
        A_zero_inverse, dist, offsets = compute_and_cache_zero_pose_data(chain, v_template, lower_limits, upper_limits)
    A = A @ A_zero_inverse

    if lrs_model is None:
        raise NotImplementedError("explicit skinning weights are not supported; pass lrs_model")
    v_template_unencoded = v_template.reshape(N * V, 3)
    v_template_encoded = positional_encoding(v_template_unencoded)
    reshaped_offsets = offsets.reshape(N * V, J, 3)
    unencoded_offsets = reshaped_offsets.reshape(N * V, J * 3)
    encoded_offsets = positional_encoding(reshaped_offsets).reshape(N * V, -1)
    distances = torch.norm(offsets, dim=-1)
    unencoded_distances = distances.reshape(N * V, J)
    encoded_distances = positional_encoding(distances).reshape(N * V, -1)
    network_input = torch.cat([v_template_encoded, encoded_offsets, encoded_distances], dim=1)
    network_input = torch.cat([network_input, unencoded_offsets, v_template_unencoded, unencoded_distances], dim=1)
    W = lrs_model(network_input).reshape(N, V, num_frames)
    W = torch.softmax(W, dim=-1)

    T = torch.matmul(W, A.view(N, num_frames, 16)).view(N, -1, 4, 4)  # N, V, 4, 4
    homogen_coord = torch.ones([N, V, 1], dtype=dtype, device=device)
    v_template_homo = torch.cat([v_template, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_template_homo, dim=-1))

    if rotations is None:
        return v_homo[:, :, :3, 0], None
    R = quaternion_to_matrix(rotations)  # V, 3, 3
    R_homo = torch.cat([
        torch.cat([R, v_template[0, :, :, None]], dim=2),
        torch.cat([torch.zeros(V, 1, 3, device=rotations.device), torch.ones(V, 1, 1, device=rotations.device)], dim=2)
    ], dim=1)
    transformed = T @ R_homo
    return transformed[:, :, :3, 3], matrix_to_quaternion(transformed[:, ..., :3, :3])


def deform(pose, pc_template, pc_deformed, deform_model):
    """Pose-conditioned appearance deltas (scale, rotation, opacity, SH) from the appearance MLP."""
    N, V, _ = pc_template.shape
    J = pose.shape[-1]
    pc_templated_encoded = positional_encoding(pc_template.reshape(N * V, 3))
    pc_deformed_encoded = positional_encoding(pc_deformed.reshape(N * V, 3))
    pose = pose.reshape(N * V, J)
    output = deform_model(torch.cat([pose, pc_templated_encoded, pc_deformed_encoded], dim=1))
    delta_scale, delta_rot, delta_opacity, delta_shs = output[:, :3], output[:, 3:7], output[:, 7:8], output[:, 8:]
    return delta_scale, delta_rot, delta_opacity, delta_shs.view(N, V, 16, 3)


def pose_conditioned_deform(pc_template, pc_deformed, scales, rotations, opacity, shs, joints, deform_model):
    delta_scale, delta_rot, delta_opacity, delta_shs = deform(joints, pc_template, pc_deformed, deform_model)
    return scales + delta_scale, rotations + delta_rot, opacity + delta_opacity, shs + delta_shs
