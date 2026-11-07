"""Nested result dictionary (``calib_dict.pkl``) helpers.

Bridge results are keyed by the trajectory's path relative to the raw root
(e.g. ``['bridge_data_v2', '<kitchen>', '<task>', '<n>', '<date>', 'raw', 'traj_group0', 'traj0']``).
Each leaf is ``{'w2c': 4x4 list, 'intrinsic': 3x3 list, 'status': {...}}``.
"""
import os
import pickle


def set_nested_dict(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def get_nested_dict(d, keys):
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d


def load_calib_dict(path):
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_calib_dict(d, path):
    with open(path, 'wb') as f:
        pickle.dump(d, f)


def bridge_keys(scene_path, raw_path):
    return os.path.relpath(scene_path, raw_path).split(os.sep)


def get_bridge_calibration(data_path, calib_dict_path, raw_path):
    """(w2c, intrinsic) for any Bridge trajectory: the calibration of ``traj0`` of its group."""
    traj0_path = os.path.join(os.path.dirname(data_path), 'traj0')
    calib = get_nested_dict(load_calib_dict(calib_dict_path), bridge_keys(traj0_path, raw_path))
    if calib is None:
        raise KeyError(f'no calibration for {traj0_path} in {calib_dict_path}')
    return calib['w2c'], calib['intrinsic']
