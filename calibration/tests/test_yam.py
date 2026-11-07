"""CPU tests for the ABC/YAM per-video calibration (no model, no gsplat, no SAM 2)."""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yam import export, station  # noqa: E402
from yam import fit  # noqa: E402


def test_se3_exp_matches_rotation_vector():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    for _ in range(5):
        xi = rng.normal(size=6) * 0.3
        T = station.se3_exp(torch.tensor(xi, dtype=torch.float64)).numpy()
        assert np.allclose(T[:3, :3], Rotation.from_rotvec(xi[3:]).as_matrix(), atol=1e-10)
        assert np.allclose(T[3], [0, 0, 0, 1])
    assert np.allclose(station.se3_exp(torch.zeros(6, dtype=torch.float64)).numpy(), np.eye(4))
    # pure translation
    assert np.allclose(station.se3_exp(torch.tensor([0.1, -0.2, 0.3, 0, 0, 0], dtype=torch.float64)).numpy()[:3, 3], [0.1, -0.2, 0.3])


def test_abc_state_to_q():
    q = station.abc_state_to_q(np.arange(6), 1.0)
    assert q.shape == (8,) and np.allclose(q[6:], [0.0475, -0.0475])
    assert np.allclose(station.abc_state_to_q(np.zeros(6), 2.0)[6:], [0.0475, -0.0475])   # clipped
    assert np.allclose(station.abc_state_to_q(np.zeros(6), 0.0)[6:], 0.0)


def test_station_geometry():
    assert np.isclose(station.right_base(0.61)[1, 3], 0.31 - 0.61)
    K, c2w = station.sim_top_camera(640, 480)
    assert np.isclose(K[0, 2], 320) and np.isclose(K[1, 2], 240)
    R = c2w[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-3)
    assert c2w[:3, 2][2] < 0          # looks down


def test_quat_mul_identity():
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    assert torch.allclose(station.quat_mul_wxyz(torch.tensor([1.0, 0, 0, 0]).expand_as(q), q), q)


def test_lookat_w2c_points_at_target():
    pos, tgt = np.array([-0.5, 0.1, 1.6]), np.array([0.5, 0.0, 0.9])
    w2c = fit.lookat_w2c(pos, tgt)
    R = w2c[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
    p_cam = w2c[:3, :3] @ tgt + w2c[:3, 3]
    assert np.allclose(p_cam[:2], 0, atol=1e-9) and p_cam[2] > 0
    # image up follows world +z: a point above the target projects above the centre (y < 0)
    up = w2c[:3, :3] @ (tgt + [0, 0, 0.1]) + w2c[:3, 3]
    assert up[1] < 0


def test_camera_frame_grid_and_seeds():
    cg, meta = fit.camera_frame_grid(np.eye(4), 0.4, 0.3, 0.15, 0.2, 9, 7, 3, 3)
    assert cg.shape == (9 * 7 * 3 * 27, 4, 4) and len(meta) == len(cg)
    la, ml = fit.lookat_seeds([0.6, 0.0, 1.0], [0.8, 1.05, 1.3, 1.6], [40, 55, 70, 85], [-30, -15, 0, 15, 30], [-8, 0, 8])
    assert la.shape == (240, 4, 4)
    assert fit.pose_delta(np.eye(4), np.eye(4)) == (0.0, 0.0)


def test_soft_and_hard_iou():
    a = torch.zeros(10, 10)
    a[:5] = 1
    t = torch.zeros(10, 10)
    t[:5, :5] = 1
    w = torch.ones(10, 10)
    assert math.isclose(float(fit.soft_iou(a, t, w)), 0.5, rel_tol=1e-5)
    assert math.isclose(float(fit.hard_iou(a, t, w)), 0.5, rel_tol=1e-5)
    w[:, 5:] = 0                     # ignore band
    assert math.isclose(float(fit.hard_iou(a, t, w)), 1.0, rel_tol=1e-5)


def test_gauss_blur_preserves_constant():
    x = torch.ones(1, 1, 20, 20)
    assert torch.allclose(fit.gauss_blur(x, 2.0), x, atol=1e-6)


def _c2w(pos, pitch_deg=50.0):
    w2c = fit.lookat_w2c(pos, np.asarray(pos) + [math.cos(math.radians(pitch_deg)), 0, -math.sin(math.radians(pitch_deg))])
    return np.linalg.inv(w2c)


def test_pose_policy():
    m = _c2w([0.0, 0.0, 1.7])
    p = m.copy()
    p[0, 3] += 0.02
    assert export.choose_pose(m, p, 0.6, 0.58)[0] == "photo"
    assert export.choose_pose(m, p, 0.6, 0.54)[0] == "mask"       # held-out IoU drop > 0.05
    far = m.copy()
    far[0, 3] += 0.06
    assert export.choose_pose(m, far, 0.6, 0.6)[0] == "mask"      # > 5 cm
    assert export.choose_pose(m, None, 0.6, None)[0] == "mask"


def test_gate():
    rows = [dict(uuid=str(i), status="ok", camera="c", date="d", holdout_iou=0.6, c2w=_c2w([0.01 * i, 0, 1.7])) for i in range(4)]
    rows.append(dict(uuid="far", status="ok", camera="c", date="d", holdout_iou=0.6, c2w=_c2w([0.2, 0, 1.7])))
    rows.append(dict(uuid="low", status="ok", camera="e", date="d", holdout_iou=0.40, c2w=_c2w([0, 0, 1.7])))
    rows.append(dict(uuid="high", status="ok", camera="f", date="d", holdout_iou=0.6, c2w=_c2w([0, 0, 2.5])))
    rows.append(dict(uuid="err", status="error", camera="f", date="d"))
    g = {r["uuid"]: r["gate"] for r in export.apply_gate(rows)}
    assert [g[str(i)] for i in range(4)] == ["pass"] * 4
    assert g["far"] == "fail" and g["low"] == "fail" and g["high"] == "fail" and g["err"] == "error"


def test_pinhole_640x480():
    K = np.array([[420.0, 0, 424.0], [0, 420.0, 240.0], [0, 0, 1]])
    Kp, cls = export.pinhole_640x480(K, 848, 480)
    assert cls == "crop848" and np.isclose(Kp[0, 2], 320.0)
    K1 = np.array([[630.0, 0, 640.0], [0, 630.0, 360.0], [0, 0, 1]])
    Kp, cls = export.pinhole_640x480(K1, 1280, 720)
    assert cls == "crop1280" and np.allclose([Kp[0, 0], Kp[0, 2], Kp[1, 2]], [420.0, 320.0, 240.0])
    with pytest.raises(ValueError):
        export.pinhole_640x480(K, 1920, 1200)


def test_camera_row_conventions():
    c2w = _c2w([0.1, -0.2, 1.7])
    row = export.camera_row(c2w, np.eye(3), 640, 480, 0.61)
    T = np.asarray(row["T_base_cam"]).reshape(4, 4)
    assert np.allclose(T, c2w) and np.allclose(np.linalg.inv(T), np.linalg.inv(c2w))
    assert np.allclose(np.asarray(row["T_leftbase_cam"]).reshape(4, 4)[:3, 3], c2w[:3, 3] - [0.2525, 0.31, 0.75])
    q = np.array([row["qx"], row["qy"], row["qz"], row["qw"]])
    from scipy.spatial.transform import Rotation
    assert np.isclose(np.linalg.norm(q), 1) and q[3] >= 0 and np.allclose(Rotation.from_quat(q).as_matrix(), c2w[:3, :3])
    # a world point projects consistently through inv(T_base_cam)
    p = np.array([0.5, 0.0, 0.8, 1.0])
    pc = np.linalg.inv(T) @ p
    assert pc[2] > 0


def test_extract_distortion_identity_and_remap():
    from yam import extract
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.4, 0.4, 5))
    xd, yd = extract.forward_distort(x, y, [])
    assert np.allclose(xd, x) and np.allclose(yd, y)
    img = (np.arange(20 * 30 * 3) % 251).astype(np.uint8).reshape(20, 30, 3)
    K = [100.0, 0, 15.0, 0, 100.0, 10.0, 0, 0, 1]
    mx, my = extract.undistort_maps(K, [0, 0, 0, 0, 0], 30, 20, K)
    assert np.array_equal(extract.remap_bilinear(img, mx, my), img)
