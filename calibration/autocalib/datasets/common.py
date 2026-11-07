import os

import numpy as np
from PIL import Image


def frame_index(name):
    """Frame number from ``im_12.jpg`` / ``12.jpg`` / ``000012.jpg``; None for other files."""
    stem = os.path.splitext(name)[0].split('_')[-1]
    return int(stem) if stem.isdigit() else None


def list_frames(image_dir):
    names = [f for f in os.listdir(image_dir) if frame_index(f) is not None]
    return sorted(names, key=frame_index)


def load_images(image_dir, n=None):
    """(T, H, W, 3) uint8 array of the first ``n`` frames (all if None), sorted by frame index."""
    names = list_frames(image_dir)
    if n is not None:
        names = names[:n]
    return np.stack([np.array(Image.open(os.path.join(image_dir, f))) for f in names])


def lr_at(epoch, base_lr):
    """Camera learning-rate schedule shared by all recipes: halved every 4 epochs."""
    return base_lr * (0.5 ** (epoch // 4))
