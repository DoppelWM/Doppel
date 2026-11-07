"""
Video decoding and frame handling shared by the latent-store builder, inference and evaluation.
"""

from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image


def _read_all_frames(path: str) -> List[Image.Image]:
    """Decode every frame of a video file as RGB PIL images (imageio-ffmpeg)."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(path, "ffmpeg")
    try:
        all_frames = [Image.fromarray(fr).convert("RGB") for fr in reader]
    finally:
        reader.close()
    if len(all_frames) == 0:
        raise ValueError(f"No frames read from {path}")
    return all_frames


read_all_frames = _read_all_frames


def subsample_frames(all_frames: Sequence[Image.Image], num_frames: int, path: str = "") -> List[Image.Image]:
    """``num_frames`` frames at ``linspace(0, n - 1, num_frames)``; a 17-frame clip is returned as is."""
    if len(all_frames) < num_frames:
        raise ValueError(f"{path}: only {len(all_frames)} frames available, {num_frames} requested")
    idxs = np.linspace(0, len(all_frames) - 1, num_frames).astype(int)
    return [all_frames[i] for i in idxs]


def load_video_and_render_frames(vpath: str, rpath: str, num_frames: int
                                 ) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Load a camera video and its robot rendering on the same temporal grid.

    A rendering with one frame more than the video (one frame per trajectory step,
    DROID convention) has its trailing frame dropped first."""
    video_all = _read_all_frames(vpath)
    render_all = _read_all_frames(rpath)
    if len(render_all) == len(video_all) + 1:
        render_all = render_all[:-1]
    elif len(render_all) != len(video_all):
        warnings.warn(f"video/render frame counts differ ({vpath}: {len(video_all)}, {rpath}: "
                      f"{len(render_all)}); they are subsampled on different grids", stacklevel=2)
    return subsample_frames(video_all, num_frames, vpath), subsample_frames(render_all, num_frames, rpath)


def resize_frames(frames: Sequence[Image.Image], height: int, width: int) -> List[Image.Image]:
    """Lanczos resize to the model canvas (non-aspect-preserving), as in training."""
    return [f.resize((width, height), Image.LANCZOS) for f in frames]


def frames_to_uint8(frames) -> np.ndarray:
    """A list / array of HxWx3 frames in [0, 1] or [0, 255] -> uint8 (T, H, W, 3)."""
    arr = np.asarray(frames)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected video array shape {arr.shape}")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def write_video(path: str, frames: np.ndarray, fps: float = 8.0) -> None:
    """uint8 (T, H, W, 3) -> mp4 (libx264)."""
    import imageio.v2 as imageio

    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
