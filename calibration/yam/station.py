"""ABC-130K station geometry and the two-arm YAM render.

Both arms are posed from ONE Dr.Robot YAM model: the Gaussians are deformed twice (left / right joint streams),
moved to their base transforms and rasterised together. World frame = the station frame of the YAM MJCF
(left base at (0.2525, 0.31, 0.75), right base at y = 0.31 - base_sep, z up). Cameras use OpenCV axes
(x right, y down, z forward); ``viewmat = inv(T_c2w)``.
Joint vectors are the 8-d chain order [joint1..joint6, left_finger, right_finger] (radians / metres).
"""
import math

import numpy as np
import torch

T_ABC_LEFT = np.array([[1, 0, 0, 0.2525], [0, 1, 0, 0.31], [0, 0, 1, 0.75], [0, 0, 0, 1]], np.float64)
T_ABC_RIGHT = np.array([[1, 0, 0, 0.2525], [0, 1, 0, -0.31], [0, 0, 1, 0.75], [0, 0, 0, 1]], np.float64)
BASE_SEP = 0.61         # right base y offset below the left base (m), from the recorded end-effector poses
# simulator top camera of the station MJCF (MuJoCo convention: x right, y up, -z forward), used as the start pose
SIM_TOP_POS = np.array([0.086, 0.0, 1.7043])
SIM_TOP_R_MJ = np.array([[0.0, 0.866, -0.5], [-1.0, 0.0, 0.0], [0.0, 0.5, 0.866]])
SIM_TOP_FOVY = 58.0
GRIPPER_MAX = 0.0475    # finger travel (m) at gripper = 1 (open)


def right_base(base_sep=BASE_SEP):
    T = T_ABC_RIGHT.copy()
    T[1, 3] = T_ABC_LEFT[1, 3] - base_sep
    return T


def load_model(model_path, iteration=None, device='cuda'):
    from drrobot.gaussian_model import load_gaussians
    return load_gaussians(model_path, iteration, device)


def joint_limits(pc):
    lo, hi = pc.chain.get_joint_limits()
    return np.asarray(lo, np.float64), np.asarray(hi, np.float64)


def normalize_q(pc, q):
    """Raw joints -> [-1, 1] (clip to the joint limits, then affine)."""
    lo, hi = joint_limits(pc)
    q = np.clip(np.asarray(q, np.float64), lo, hi)
    return ((q - lo) * 2.0 / (hi - lo) - 1.0).astype(np.float32)


def abc_state_to_q(q6, grip01):
    """ABC arm state (6 joints) + gripper in [0, 1] (1 = open) -> 8-d YAM joint vector."""
    g = float(np.clip(grip01, 0.0, 1.0)) * GRIPPER_MAX
    return np.r_[np.asarray(q6, np.float64), g, -g]


def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def rotmat_to_quat_wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z], np.float64)


def deform_arm(pc, q_raw8):
    """One arm at raw joints -> (means, quats wxyz, scales, opacity, SH) in the arm base frame."""
    from drrobot.lbs import lrs
    means3D, rotations = pc.get_xyz, pc.get_rotation
    jn = torch.as_tensor(normalize_q(pc, q_raw8), device=means3D.device).float()
    md, rd = lrs(jn[None], means3D[None], pc._lrs, pc.chain, pose_normalized=True, lrs_model=pc.lrs_model,
                 rotations=rotations)
    # the model has no pose-dependent appearance: scales, opacities and colours stay canonical
    return md[0], rd[0], pc.get_scaling, pc.get_opacity, pc.get_features


def place(means, quats, T):
    """Rigidly move a Gaussian set by the 4x4 transform T (base -> world)."""
    R = torch.as_tensor(T[:3, :3], device=means.device, dtype=means.dtype)
    t = torch.as_tensor(T[:3, 3], device=means.device, dtype=means.dtype)
    qR = torch.as_tensor(rotmat_to_quat_wxyz(T[:3, :3]), device=means.device, dtype=quats.dtype)
    return means @ R.T + t, quat_mul_wxyz(qR.expand_as(quats), quats)


def mujoco_cam_to_cv(pos, R_mj):
    """MuJoCo camera (c2w, y up, -z forward) -> OpenCV c2w 4x4."""
    T = np.eye(4)
    T[:3, :3] = R_mj @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = pos
    return T


def sim_top_camera(w, h, fovy_deg=SIM_TOP_FOVY):
    """K and OpenCV c2w of the simulator top camera for a w x h image (principal point centred)."""
    fy = 0.5 * h / math.tan(math.radians(fovy_deg) / 2)
    K = np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], np.float64)
    return K, mujoco_cam_to_cv(SIM_TOP_POS, SIM_TOP_R_MJ)


@torch.no_grad()
def render_two_arms(pc, K, w, h, T_c2w, q_left, q_right, T_left=T_ABC_LEFT, T_right=T_ABC_RIGHT, bg=(1.0, 1.0, 1.0),
                    arms=("left", "right"), deform=None):
    """rgb (h, w, 3) in [0, 1] and alpha (h, w) of the requested arms, one rasterisation."""
    from gsplat import rasterization
    deform = deform or deform_arm
    parts = []
    for name, q, T in (("left", q_left, T_left), ("right", q_right, T_right)):
        if name not in arms:
            continue
        m, r, s, o, c = deform(pc, q)
        m, r = place(m, r, T)
        parts.append((m, r, s, o, c))
    means, quats, scales, opac, shs = (torch.cat([p[i] for p in parts]) for i in range(5))
    dev = means.device
    viewmat = torch.as_tensor(np.linalg.inv(T_c2w), device=dev, dtype=torch.float32)
    Kt = torch.as_tensor(K, device=dev, dtype=torch.float32)
    colors, alphas, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac.squeeze(-1), colors=shs,
                                      viewmats=viewmat[None], Ks=Kt[None], backgrounds=torch.tensor(bg, device=dev)[None],
                                      width=int(w), height=int(h), packed=False, sh_degree=pc.active_sh_degree)
    return colors[0].clamp(0, 1).cpu().numpy(), alphas[0, ..., 0].cpu().numpy()


def overlay(rgb_real_u8, rgb_render01, alpha, strength=0.6):
    """Alpha-weighted blend of a render onto a real frame."""
    real = rgb_real_u8.astype(np.float32) / 255.0
    a = (alpha[..., None] * strength).astype(np.float32)
    return ((1 - a) * real + a * rgb_render01).clip(0, 1)


def se3_exp(xi):
    """xi (6,) = [rho, omega] -> 4x4 SE(3) (Rodrigues + V matrix), differentiable."""
    rho, om = xi[:3], xi[3:]
    th = torch.linalg.norm(om) + 1e-12
    k = om / th
    z = torch.zeros_like(th)
    Kx = torch.stack([torch.stack([z, -k[2], k[1]]), torch.stack([k[2], z, -k[0]]), torch.stack([-k[1], k[0], z])])
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)
    s, c = torch.sin(th), torch.cos(th)
    R = I + s * Kx + (1 - c) * Kx @ Kx
    V = I + (1 - c) / th * Kx + (th - s) / th * Kx @ Kx
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    return torch.cat([torch.cat([R, (V @ rho)[:, None]], 1), T[3:4]], 0)
