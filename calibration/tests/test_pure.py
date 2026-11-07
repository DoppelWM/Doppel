"""CPU-only unit tests (no model, no GPU)."""
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from autocalib import config  # noqa: E402
from autocalib.calib_dict import bridge_keys, get_bridge_calibration, get_nested_dict, load_calib_dict, save_calib_dict, set_nested_dict  # noqa: E402
from autocalib.datasets.common import frame_index, list_frames  # noqa: E402
from autocalib.datasets.robomind import pair_ids  # noqa: E402
from autocalib.pose_grid import grid_from_ranges, perturb_extrinsic  # noqa: E402


def test_recipe_constants():
    assert config.DENSE.camera_lr == 0.001 and config.DENSE.epochs == 20 and config.DENSE.steps_per_batch == 10
    assert config.DENSE_ROBOMIND.camera_lr == 0.004 and config.DENSE_ROBOMIND.epochs == 16 and config.DENSE_ROBOMIND.cotracker_n_frame == 1
    assert config.DENSE.motion_threshold == 1.4 and config.DENSE.cotracker_n_frame == 5 and config.DENSE.batch_size == 3


def test_finger_mappings():
    f = config.bridge_fingers(np.array([[0.0], [1.0]]))
    assert f.shape == (2, 2) and np.allclose(f[1], [0.037, -0.037]) and np.allclose(f[0], [0.015, -0.015])
    g = config.robotiq_fingers(np.ones((3, 1)))
    assert g.shape == (3, 8) and np.allclose(g[0], config.ROBOTIQ_2F85_FINGER_GAINS[0])


def test_init_w2c_are_rigid():
    for w2c in [config.BRIDGE_INIT_W2C] + list(config.ROBOMIND_INIT_W2C.values()):
        R = w2c[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.allclose(w2c[3], [0, 0, 0, 1])


def test_pose_grid_identity_and_count():
    E = torch.tensor(config.BRIDGE_INIT_W2C, dtype=torch.float32)
    z = torch.zeros(1)
    assert torch.allclose(perturb_extrinsic(E, z, z, z, z, z, z)[0], E)
    t = torch.linspace(-0.2, 0.2, 3)
    grid = perturb_extrinsic(E, t, t, t, z, z, z)
    assert grid.shape == (27, 4, 4)
    assert grid_from_ranges(config.BRIDGE_INIT_W2C).shape == (1, 4, 4)
    for G in grid:
        R = G[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)


def test_frame_sorting(tmp_path):
    for n in ['im_10.jpg', 'im_2.jpg', 'im_0.jpg', 'super.png', '000003.jpg']:
        (tmp_path / n).write_bytes(b'')
    assert frame_index('super.png') is None
    assert list_frames(str(tmp_path)) == ['im_0.jpg', 'im_2.jpg', '000003.jpg', 'im_10.jpg']


def test_robomind_pair_schedule():
    assert pair_ids(108) == [(i, i + 1) for i in range(10, 88, 5)]
    assert pair_ids(35) == [(i, i + 1) for i in range(10, 15)]
    assert pair_ids(12) == []


def test_calib_dict_roundtrip(tmp_path):
    raw = str(tmp_path / 'raw')
    scene = os.path.join(raw, 'bridge_data_v2', 'kitchen', 'task', 'date', 'raw', 'traj_group0', 'traj0')
    d = {}
    set_nested_dict(d, bridge_keys(scene, raw), {'w2c': np.eye(4).tolist(), 'intrinsic': np.eye(3).tolist()})
    path = str(tmp_path / 'calib_dict.pkl')
    save_calib_dict(d, path)
    assert get_nested_dict(load_calib_dict(path), bridge_keys(scene, raw))['w2c'] == np.eye(4).tolist()
    w2c, K = get_bridge_calibration(os.path.join(os.path.dirname(scene), 'traj7'), path, raw)
    assert w2c == np.eye(4).tolist() and K == np.eye(3).tolist()
    with pytest.raises(KeyError):
        get_bridge_calibration(os.path.join(raw, 'other', 'raw', 'traj_group0', 'traj1'), path, raw)


def test_se3_exp_matches_rodrigues():
    from scipy.spatial.transform import Rotation
    from drrobot.graphics import se3_to_SE3
    w = torch.tensor([0.1, -0.2, 0.3])
    v = torch.tensor([0.05, 0.0, -0.1])
    T = se3_to_SE3(w, v).numpy()
    assert np.allclose(T[:3, :3], Rotation.from_rotvec(w.numpy()).as_matrix(), atol=1e-6)
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_binary_gripper_mapping():
    from autocalib.datasets.robomind import fingers_for, _CLOSED, _OPEN
    f = fingers_for(np.array([0.0, 0.1, 0.5, 0.6]), 'binary')
    assert f.shape == (4, 8)
    assert np.allclose(f[0], _OPEN + _OPEN) and np.allclose(f[3], _CLOSED + _CLOSED)


def test_canonical_pose_rule():
    from drrobot import lbs
    lbs.CANONICAL_POSE_OVERRIDE = None
    lo, hi = torch.tensor([-1.0, 0.2, -2.0]), torch.tensor([1.0, 1.0, -0.5])
    assert torch.allclose(lbs.get_canonical_pose(lo, hi), torch.tensor([0.0, 0.2, -0.5]))
    lbs.CANONICAL_POSE_OVERRIDE = np.array([0.1, 0.3, -1.0])
    try:
        assert torch.allclose(lbs.get_canonical_pose(lo, hi), torch.tensor([0.1, 0.3, -1.0]))
    finally:
        lbs.CANONICAL_POSE_OVERRIDE = None


def test_robomind_session_segmentation():
    sys.path.insert(0, os.path.join(REPO, 'tools'))
    from robomind_sessions import segment, sess_key
    assert sess_key('1023_133146') == '1023pm' and sess_key('1023_093146') == '1023am'
    lab = np.array([0] * 10 + [1] * 8 + [0] * 6)          # camera moved, then put back
    D = np.where(lab[:, None] == lab[None], 1.0, 40.0)
    np.fill_diagonal(D, 0.0)
    pose = segment(D, 8.0)      # change points can land one episode late (4x4 window)
    assert set(pose) == {1, 2} and pose[:10] == [1] * 10 and pose[12:17] == [2] * 5 and pose[-5:] == [1] * 5
