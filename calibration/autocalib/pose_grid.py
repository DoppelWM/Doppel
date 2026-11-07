"""Rotation/translation grid around a world-to-camera matrix."""
import torch


def rot_x(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    o, z = torch.ones_like(theta), torch.zeros_like(theta)
    return torch.stack([torch.stack([o, z, z], -1), torch.stack([z, c, -s], -1), torch.stack([z, s, c], -1)], -2)


def rot_y(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    o, z = torch.ones_like(theta), torch.zeros_like(theta)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)


def rot_z(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    o, z = torch.ones_like(theta), torch.zeros_like(theta)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def perturb_extrinsic(E, thetas_x, thetas_y, thetas_z, dxs, dys, dzs):
    """All combinations of camera-frame rotations (rad) and translations applied to w2c ``E`` -> (n, 4, 4)."""
    R = E[:3, :3]
    t = E[:3, 3:4]
    R_x = rot_x(thetas_x)
    R_y = rot_y(thetas_y)
    R_z = rot_z(thetas_z)
    R_delta = torch.matmul(R_z[:, None, None], torch.matmul(R_y[None, :, None], R_x[None, None, :]))
    n_rot = R_delta.shape[0] * R_delta.shape[1] * R_delta.shape[2]
    R_delta = R_delta.view(n_rot, 3, 3)
    all_delta_t_c = torch.cartesian_prod(dxs, dys, dzs)
    n_trans = all_delta_t_c.shape[0]
    all_delta_t_c = all_delta_t_c.view(n_trans, 3, 1)
    R_delta_T = R_delta.transpose(-2, -1)
    R_prime = torch.matmul(R_delta_T, R)
    temp = torch.matmul(R_delta_T, t)
    R_delta_T_delta_t_c = torch.matmul(R_delta_T[:, None], all_delta_t_c[None, :])
    t_prime = temp[:, None] - R_delta_T_delta_t_c
    R_prime_exp = R_prime[:, None].expand(-1, n_trans, 3, 3)
    E_top = torch.cat([R_prime_exp, t_prime], dim=3)
    last_row = torch.tensor([[0, 0, 0, 1]], dtype=E.dtype).expand(n_rot, n_trans, 1, 4)
    E_prime = torch.cat([E_top, last_row], dim=2)
    return E_prime.view(n_rot * n_trans, 4, 4)


def grid_from_ranges(E, rot_xy=0.0, rot_z=0.0, n_rot=1, trans=0.0, n_trans=1):
    """Convenience: symmetric grid of ``n_rot`` angles in [-rot_xy, rot_xy] (x, y), [-rot_z, rot_z] (z) and
    ``n_trans`` offsets in [-trans, trans] per axis. ``n_rot = n_trans = 1`` -> just ``E``."""
    E = torch.tensor(E, dtype=torch.float32)
    tx = torch.linspace(-rot_xy, rot_xy, n_rot)
    tz = torch.linspace(-rot_z, rot_z, n_rot)
    d = torch.linspace(-trans, trans, n_trans)
    return perturb_extrinsic(E, tx, tx, tz, d, d, d)
