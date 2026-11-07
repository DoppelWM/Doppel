"""Monocular intrinsics with MoGe (Ruicheng/moge-vitl) for datasets without camera calibration (Bridge).

MoGe v1 ``infer`` per frame without a FoV prior; the calibration uses the mean focal lengths (pixels) and puts
the principal point at the image centre. For a fixed camera, ``tools/bridge_rigs.py`` writes one focal per
session (the median over its trajectories) to ``intrinsics.npy``, which takes precedence. Depths are stored but
not used.
"""
import os
import pickle

import numpy as np
import torch
from PIL import Image

MOGE_MODEL = 'Ruicheng/moge-vitl'


def denormalize_intrinsics(intrinsics, width, height):
    zeros = torch.zeros_like(width)
    ones = torch.ones_like(width)
    transform = torch.stack([1 / width, zeros, 0.5 / width, zeros, 1 / height, 0.5 / height, zeros, zeros, ones]).reshape(*zeros.shape, 3, 3).to(intrinsics)
    return torch.linalg.inv(transform) @ intrinsics


@torch.inference_mode()
def estimate_moge(image_dir, max_frames=200, model_name=MOGE_MODEL, device='cuda'):
    from moge.model.v1 import MoGeModel
    from .datasets.common import list_frames  # lazy: datasets import this module
    names = list_frames(image_dir)[:max_frames]
    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    intrinsics, depths = [], []
    for n in names:
        img = np.array(Image.open(os.path.join(image_dir, n)).convert('RGB'))
        h, w = img.shape[:2]
        x = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).to(device)
        out = model.infer(x)
        K = denormalize_intrinsics(out['intrinsics'], torch.tensor(w, device=device), torch.tensor(h, device=device))
        intrinsics.append(K.cpu().numpy())
        d = out['depth'].cpu().numpy()
        m = out['mask'].cpu().numpy() & np.isfinite(d)
        d[~m] = 0
        depths.append(d)
    return {'depths': np.stack(depths).astype(np.float32), 'intrinsics': np.stack(intrinsics).astype(np.float32)}


def ensure_moge_intrinsics(scene_path, image_dir, pkl_name='moge_static.pkl', max_frames=200):
    """Load ``<scene>/moge_static.pkl`` or create it with MoGe."""
    path = os.path.join(scene_path, pkl_name)
    if not os.path.exists(path):
        print(f'[intrinsics] running MoGe on {image_dir} -> {path}')
        with open(path, 'wb') as f:
            pickle.dump(estimate_moge(image_dir, max_frames=max_frames), f)
    with open(path, 'rb') as f:
        return pickle.load(f)


def moge_intrinsics_matrix(moge, W, H):
    fx = np.mean(moge['intrinsics'][:, 0, 0])
    fy = np.mean(moge['intrinsics'][:, 1, 1])
    return np.array([[fx, 0.0, W / 2], [0.0, fy, H / 2], [0.0, 0.0, 1.0]])
