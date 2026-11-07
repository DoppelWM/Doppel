"""Render robot-only videos of ``scene_*`` folders with a trained Dr. Robot Gaussian robot model (CLI: ``main()``).

This script runs inside the Dr. Robot codebase (differentiable robot rendering, Liu et al. 2024): it imports that
codebase's ``arguments``, ``gaussian_renderer``, ``scene`` and ``utils`` modules and loads a robot model trained
with it. The model is fixed; only the joint trajectory and the camera change per episode.

A scene folder holds ``trajectory.h5`` (joint and gripper log, camera extrinsics), a metadata json and
``recordings/MP4/<serial>.mp4``: raw DROID episodes, and RoboMIND / Bridge episodes converted to the same layout.

* **Any chain articulates.** The gripper mechanism is located with ``gripper_map.build_gripper_map()``
  (``n_arm`` + ``qpos_idx``), so the 15-DoF FR3 (7 arm + 8 finger), the 14-DoF UR5e (6 + 8) and the 8-DoF
  WidowX (6 + 2) all work.
* **Normalisation.** ``render_scene_video`` normalises the *full* qpos to [-1, 1] with the chain limits (the
  WidowX finger joints are in metres, the arm joints in radians).
* **Camera.** The calibrated camera pose comes from the scene (``trajectory.h5:
  observation/camera_extrinsics/<serial>_left``, per frame with ``--trajectory_camera``, or the metadata's
  ``<camera>_cam_extrinsics``). Intrinsics come from the scene when it has them (``metadata.json
  ["camera_intrinsics"]`` or ``trajectory.h5:observation/camera_intrinsics/<serial>_left``); raw DROID has none,
  so the per-serial ZED table is used there (built-in serials, extended by ``--zed_intrinsics``).
* **Frame grid.** ``--frame_indices_from`` renders exactly the source-frame indices of each clip (the
  ``render_jobs_*.csv`` written by ``scripts/data/build_windows.py``), ``--max_frames 17`` the linspace grid of
  the full-span clip. Both grids are laid over the camera video's frame count, so the render, the RGB clip and
  the actions are sampled on the same source frames.

Each render is written to ``<output_dir>/<scene>/drrobot_render.mp4`` with a ``render_meta.json`` beside it
(robot type, camera, intrinsics, size, fps and the frame indices), which ``build_windows.py`` and
``build_index.py`` check.

Gripper: the logged closure (0 = open, 1 = closed) drives the finger joints through ``gripper_map``
(``closure_from_log``, then the model's closure -> finger-qpos table).

Output ``--fps`` defaults to ``match`` (the frame rate of ``recordings/MP4/<serial>.mp4``); ``--fps meta`` reads
``metadata.json["fps"]``; a number pins a fixed rate.

Example::

    MUJOCO_GL=egl python render_robot.py \\
        --model_path output/ur5e_2f85 --robot_type ur5e_2f85 \\
        --dataset_root data/robomind/scenes --output_dir data/robomind/renders \\
        --camera_key ext1 --fps meta --max_frames 17
"""

import argparse
import copy
import csv
import glob
import json
import os
import subprocess
from argparse import Namespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation
try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    ImageSequenceClip = None

from arguments import OptimizationParams
from gaussian_renderer import GaussianModel, render
from scene import RobotScene
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix, getWorld2View2

# ---------------------------------------------------------------------------
# ZED 720p left-eye intrinsics per camera serial (raw DROID episodes carry no intrinsics in trajectory.h5).
# ---------------------------------------------------------------------------

ZED_INTRINSICS_720P: Dict[str, Dict[str, float]] = {
    # wrist
    "16787047": {"fx": 771.21, "fy": 771.04, "cx": 646.12, "cy": 365.87},
    # ext1
    "20103212": {"fx": 533.41, "fy": 533.46, "cx": 646.49, "cy": 360.19},
    # ext2
    "20655732": {"fx": 533.73, "fy": 533.62, "cx": 638.15, "cy": 355.50},
}

ZED_DEFAULT_INTRINSICS_720P: Dict[str, float] = {
    "fx": 533.0, "fy": 533.0, "cx": 640.0, "cy": 360.0,
}


def load_zed_intrinsics_table(path: str) -> int:
    """Merge a per-serial intrinsics table into ``ZED_INTRINSICS_720P`` (``--zed_intrinsics``).

    The table is the JSON the data scripts take as ``--intrinsics`` (``{"intrinsics": {serial: {fx, fy, cx,
    cy}}}``). The built-in serials above take precedence: the table only adds serials they do not list.
    Returns the number of serials added.
    """
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    n_before = len(ZED_INTRINSICS_720P)
    for serial, d in blob.get("intrinsics", blob).items():
        ZED_INTRINSICS_720P.setdefault(str(serial), {k: float(v) for k, v in d.items()})
    return len(ZED_INTRINSICS_720P) - n_before


def _K_to_intrinsics(K) -> Optional[Dict[str, float]]:
    """3x3 (or flat 9) camera matrix -> ``{fx, fy, cx, cy}``; None if it is not a usable K."""
    try:
        arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    except Exception:
        return None
    if not np.isfinite(arr).all() or arr[0, 0] <= 0 or arr[1, 1] <= 0:
        return None
    return {"fx": float(arr[0, 0]), "fy": float(arr[1, 1]), "cx": float(arr[0, 2]), "cy": float(arr[1, 2])}


def scene_own_intrinsics(scene_dir: str, serial: str) -> Optional[Dict[str, float]]:
    """The scene's own measured intrinsics, or None if it has none (raw DROID).

    Converted RoboMIND / Bridge scenes carry both forms; they are preferred over the ZED serial table, which
    only describes DROID cameras:

      1. ``metadata.json["camera_intrinsics"][serial]`` -- a ``{fx, fy, cx, cy, width, height}`` dict
         (or a 3x3 K, accepted too)
      2. ``trajectory.h5:observation/camera_intrinsics/<serial>_left`` -- a 3x3 K
    """
    md = load_scene_metadata(scene_dir) or {}
    entry = (md.get("camera_intrinsics") or {})
    if isinstance(entry, dict):
        d = entry.get(str(serial), entry.get(f"{serial}_left"))
        if isinstance(d, dict) and all(k in d for k in ("fx", "fy", "cx", "cy")):
            return {k: float(d[k]) for k in ("fx", "fy", "cx", "cy")}
        if d is not None:
            k = _K_to_intrinsics(d)
            if k is not None:
                return k

    h5_path = os.path.join(scene_dir, "trajectory.h5")
    if not os.path.isfile(h5_path):
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get("observation/camera_intrinsics")
            if not isinstance(grp, h5py.Group):
                return None
            for name in (f"{serial}_left", str(serial), f"{serial}_right"):
                if name in grp and isinstance(grp[name], h5py.Dataset):
                    return _K_to_intrinsics(grp[name][()])
    except Exception as e:  # a broken h5 must not kill the render; fall through to the ZED path
        print(f"[intrinsics] could not read camera_intrinsics from {h5_path}: {e}")
    return None


def get_scene_camera_intrinsics(scene_dir: str, camera_key: str) -> Dict[str, float]:
    """Return intrinsics ``{fx, fy, cx, cy}`` for a scene's camera.

    Lookup order:
      1. The scene's own intrinsics (:func:`scene_own_intrinsics`) -- converted RoboMIND/Bridge scenes
      2. ``ZED_INTRINSICS_720P`` (built-in DROID serials plus any ``--zed_intrinsics`` table)
      3. ``ZED_DEFAULT_INTRINSICS_720P``
    """
    md = load_scene_metadata(scene_dir)
    if md is None:
        print(f"[intrinsics] no metadata for {os.path.basename(scene_dir)}, using defaults")
        return ZED_DEFAULT_INTRINSICS_720P

    try:
        serial = camera_serial_for_key(md, camera_key)
    except KeyError:
        print(f"[intrinsics] no serial for {camera_key} in metadata, using defaults")
        return ZED_DEFAULT_INTRINSICS_720P

    own = scene_own_intrinsics(scene_dir, serial)
    if own is not None:
        print(f"[intrinsics] scene-provided for serial {serial}: "
              f"fx={own['fx']:.2f} fy={own['fy']:.2f} cx={own['cx']:.2f} cy={own['cy']:.2f}")
        return own

    if serial in ZED_INTRINSICS_720P:
        return ZED_INTRINSICS_720P[serial]

    print(f"[intrinsics] using defaults for unknown serial {serial}")
    return ZED_DEFAULT_INTRINSICS_720P


def _parse_r_frame_rate(rate: str) -> Optional[float]:
    """Parse ffprobe ``r_frame_rate`` like ``60/1`` or ``30000/1001``."""
    s = (rate or "").strip()
    if not s or s == "N/A":
        return None
    if "/" in s:
        num, den = s.split("/", 1)
        try:
            d = float(den)
            return float(num) / d if d else None
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def recording_mp4_for_camera(scene_dir: str, camera_key: Optional[str]) -> Optional[str]:
    """Path of ``recordings/MP4/<serial>.mp4`` for this camera key, or None."""
    if not camera_key:
        return None
    try:
        md = load_scene_metadata(scene_dir)
        if md is None:
            return None
        serial = camera_serial_for_key(md, camera_key)
    except Exception:
        return None
    mp4_dir = os.path.join(scene_dir, "recordings", "MP4")
    for name in (f"{serial}.mp4", f"{serial}-stereo.mp4"):
        p = os.path.join(mp4_dir, name)
        if os.path.isfile(p):
            return p
    return None


def count_frames(mp4_path: str) -> int:
    """Decoded frame count of the camera mp4. Decodes rather than trusting the container frame count: DROID
    containers carry a 60 fps tag on 15 fps content, and this number decides which trajectory steps a
    render samples."""
    import cv2
    cap = cv2.VideoCapture(str(mp4_path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def scene_metadata_fps(scene_dir: str) -> Optional[float]:
    """``metadata.json["fps"]`` of a scene, else None.

    RoboMIND/Bridge episodes converted to the DROID scene layout record their frame rate there: 10 fps for
    RoboMIND (its h5 carries no timestamps, so the rate is assumed), 5 fps for Bridge (from the observation
    time stamps). Neither is recoverable from the mp4 container.
    """
    md = load_scene_metadata(scene_dir) or {}
    try:
        fps = float(md.get("fps"))
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def metadata_camera_size(scene_dir: str, camera_key: Optional[str]) -> Optional[Tuple[int, int]]:
    """(width, height) the scene's ``metadata.json`` claims for this camera, or None.

    This is the size the metadata's own ``fx``/``fy`` belong to, which is why it is preferred: the
    render FoV is derived from ``fx`` and the render width, so intrinsics and size must describe the
    same image.
    """
    md = load_scene_metadata(scene_dir) or {}
    serial = None
    try:
        serial = camera_serial_for_key(md, camera_key) if camera_key else None
    except KeyError:
        serial = None
    entry = md.get("camera_intrinsics") or {}
    d = entry.get(str(serial), entry.get(f"{serial}_left")) if isinstance(entry, dict) and serial else None
    if isinstance(d, dict) and d.get("width") and d.get("height"):
        return int(d["width"]), int(d["height"])
    if md.get("width") and md.get("height"):
        return int(md["width"]), int(md["height"])
    return None


def mp4_camera_size(scene_dir: str, camera_key: Optional[str]) -> Optional[Tuple[int, int]]:
    """(width, height) read out of ``recordings/MP4/<serial>.mp4``'s header, or None.

    Uses ffprobe when available, else the ffmpeg binary that imageio-ffmpeg ships. Returns None when nothing
    can be read; the caller then keeps its own default rather than guessing.
    """
    mp4_path = recording_mp4_for_camera(scene_dir, camera_key)
    if mp4_path is None:
        return None
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
           "-of", "csv=s=x:p=0", mp4_path]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        w, h = (int(v) for v in out.split("x")[:2])
        return w, h
    except Exception:
        pass
    try:
        import imageio.v2 as imageio
        rd = imageio.get_reader(mp4_path, "ffmpeg")
        size = rd.get_meta_data().get("size")
        rd.close()
        if size and len(size) == 2:
            return int(size[0]), int(size[1])
    except Exception:
        return None
    return None


def scene_native_size(scene_dir: str, camera_key: Optional[str],
                      prefer: str = "meta") -> Optional[Tuple[int, int]]:
    """Native (width, height) of the real camera. ``prefer`` picks which source is trusted first.

    ``meta`` metadata intrinsics, then the mp4 header (what ``--render_size meta`` means).
    ``mp4``  the mp4 header, then the metadata (what ``--render_size native`` means).
    """
    order = (mp4_camera_size, metadata_camera_size) if prefer == "mp4" else (metadata_camera_size,
                                                                             mp4_camera_size)
    for fn in order:
        wh = fn(scene_dir, camera_key)
        if wh is not None:
            return wh
    return None


def recording_fps_for_camera(scene_dir: str, camera_key: Optional[str]) -> Optional[float]:
    """Read FPS from ``recordings/MP4/<serial>.mp4`` for this camera with ffprobe (None if unavailable)."""
    if not camera_key:
        return None
    mp4_path = recording_mp4_for_camera(scene_dir, camera_key)
    if mp4_path is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nw=1:nk=1",
        mp4_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return _parse_r_frame_rate(out)


def parse_render_size(value: str) -> Union[str, Tuple[int, int]]:
    """argparse type for ``--render_size``: ``meta`` | ``native`` | ``WxH`` (e.g. ``640x480``)."""
    s = str(value).strip().lower()
    if s in ("meta", "native"):
        return s
    for sep in ("x", "X", ","):
        if sep in s:
            try:
                w, h = (int(v) for v in s.split(sep, 1))
            except ValueError:
                break
            if w > 0 and h > 0:
                return w, h
            break
    raise argparse.ArgumentTypeError(f"--render_size must be meta, native or WxH (got {value!r})")


def resolve_render_size(
    scene_dir: str, camera_key: Optional[str], spec: Union[str, Tuple[int, int]],
    fallback: Tuple[int, int] = (1280, 720),
) -> Tuple[int, int, str]:
    """``--render_size`` spec -> (width, height, source). The source is printed.

    ``meta`` trusts ``metadata.json`` first (converted RoboMIND/Bridge episodes record the real camera's W x H there),
    ``native`` trusts the recorded mp4 first. They only differ if a scene's metadata disagrees with its
    own video; the render must have the same W x H as the video. ``fallback`` (DROID 1280x720) is used
    only when neither can be read.

    An explicit ``WxH`` is passed through, with a warning when it is not the camera's own size: the
    render FoV comes from ``fx`` and the render width (see :func:`override_camera_intrinsics`), so a
    size the intrinsics do not belong to crops the view instead of rescaling it.
    """
    if isinstance(spec, (tuple, list)):
        w, h = int(spec[0]), int(spec[1])
        native = scene_native_size(scene_dir, camera_key)
        if native is not None and native != (w, h):
            print(f"[render_size] WARN: explicit {w}x{h} but this camera is {native[0]}x{native[1]}; "
                  f"FoV is derived from fx and the render width, so this crops the view rather than "
                  f"rescaling it. Pass --fov to set the field of view explicitly.")
        return w, h, "explicit"
    prefer = "mp4" if spec == "native" else "meta"
    first, second = (("mp4", mp4_camera_size), ("scene metadata", metadata_camera_size)) if \
        prefer == "mp4" else (("scene metadata", metadata_camera_size), ("mp4", mp4_camera_size))
    for src, fn in (first, second):
        wh = fn(scene_dir, camera_key)
        if wh is not None:
            return wh[0], wh[1], src
    return int(fallback[0]), int(fallback[1]), "fallback (nothing readable)"


def resolve_fps(
    scene_dir: str, camera_key: Optional[str], spec: Union[str, int, float], fallback: float = 15.0,
) -> Tuple[float, str]:
    """``--fps`` spec -> (fps, source). ``meta`` = ``metadata.json["fps"]``, ``match`` = ffprobe the mp4.

    Both fall back to *fallback* (15.0). ``meta`` is the right choice for converted scenes (RoboMIND 10,
    Bridge 5), whose rate is not in the mp4 container. The source string is returned so the caller can
    record it per episode.
    """
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return float(spec), "explicit"
    s = str(spec).strip().lower()
    if s not in ("meta", "match", "auto"):
        return float(spec), "explicit"
    if s == "meta":
        fps = scene_metadata_fps(scene_dir)
        if fps is not None:
            return float(fps), "metadata.json fps"
    inferred = recording_fps_for_camera(scene_dir, camera_key)
    if inferred is not None:
        return float(inferred), "ffprobe mp4"
    return float(fallback), f"fallback {fallback:g} (no {'metadata fps nor ' if s == 'meta' else ''}ffprobe)"


def write_video(frames, output_path: str, fps: Union[int, float]):
    if ImageSequenceClip is not None:
        clip = ImageSequenceClip(frames, fps=float(fps))
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        return

    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError(
            "Neither moviepy nor imageio is available. Install one with:\n"
            "pip install moviepy\n"
            "or\n"
            "pip install imageio imageio-ffmpeg"
        ) from e

    with imageio.get_writer(output_path, fps=float(fps), codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def load_cfg_args(model_path: str) -> Namespace:
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"--model_path is not a directory: {model_path}\n"
            "Pass the directory the Dr. Robot trainer wrote (it must contain cfg_args)."
        )
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"Missing cfg_args in model directory:\n  {cfg_path}\n\n"
            "Pass the full path to a trained model (directory with cfg_args, sample_cameras.pkl, "
            "point_cloud/, robot_xml/)."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_string = f.read()
    return eval(cfg_string)


def load_model(model_path: str, background: str = "white"):
    model_path = os.path.abspath(os.path.expanduser(model_path))
    cfg_args = load_cfg_args(model_path)
    cfg_args.model_path = model_path

    parser = argparse.ArgumentParser()
    opt_params = OptimizationParams(parser)

    sh_degree = getattr(cfg_args, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree, opt_params)
    scene = RobotScene(cfg_args, gaussians, opt_params=opt_params, from_ckpt=True, load_iteration=-1)
    gaussians.model_path = scene.model_path

    if background == "black":
        bg_rgb = (0.0, 0.0, 0.0)
    else:
        bg_rgb = (1.0, 1.0, 1.0)
    bg_color = torch.tensor(bg_rgb, dtype=torch.float32, device="cuda")
    sample_cameras = scene.getSampleCameras(stage="pose_conditioned")
    if len(sample_cameras) == 0:
        raise RuntimeError("No sample cameras found in loaded checkpoint.")
    return gaussians, bg_color, sample_cameras


def droid_6dof_to_w2c(values) -> np.ndarray:
    """Convert DROID [x, y, z, roll, pitch, yaw] -> 4x4 W2C for the Gaussian renderer.

    Steps (DROID convention):
      1. Build T_base_cam (C2W) from position + ``from_euler('xyz', euler)``
      2. Invert to get W2C for ``apply_extrinsic_to_camera``
    """
    v = np.asarray(values, dtype=np.float64).reshape(6)
    pos = v[:3]
    euler = v[3:]

    rot = ScipyRotation.from_euler("xyz", euler).as_matrix()

    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3, :3] = rot
    T_base_cam[:3, 3] = pos

    T_w2c = np.linalg.inv(T_base_cam).astype(np.float32)
    return T_w2c


def load_scene_metadata(scene_dir: str) -> Optional[dict]:
    # Match both DROID's `metadata_<uuid>.json` and the `metadata.json` of converted scenes.
    paths = (
        glob.glob(os.path.join(scene_dir, "metadata_*.json"))
        + glob.glob(os.path.join(scene_dir, "metadata.json"))
    )
    if not paths:
        return None
    with open(paths[0], "r", encoding="utf-8") as f:
        return json.load(f)


def camera_serial_for_key(metadata: dict, camera_key: str) -> str:
    field = f"{camera_key}_cam_serial"
    if field not in metadata:
        raise KeyError(f"Metadata missing {field!r} (keys like wrist_cam_serial, ext1_cam_serial, ...).")
    return str(metadata[field])


def parse_camera_extrinsic(scene_dir: str, camera_key: Optional[str]) -> Optional[np.ndarray]:
    candidates = [
        os.path.join(scene_dir, "camera_extrinsics.npy"),
        os.path.join(scene_dir, "extrinsics.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape == (4, 4):
                return arr.astype(np.float32)
            if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                return arr[0].astype(np.float32)
            raise ValueError(f"Unsupported extrinsics shape in {path}: {arr.shape}")

    metadata_paths = (
        glob.glob(os.path.join(scene_dir, "metadata_*.json"))
        + glob.glob(os.path.join(scene_dir, "metadata.json"))
    )
    if len(metadata_paths) == 0 or not camera_key:
        return None

    with open(metadata_paths[0], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    field = f"{camera_key}_cam_extrinsics"
    values = metadata.get(field)
    if isinstance(values, list) and len(values) == 6:
        return droid_6dof_to_w2c(values)

    # Converted RoboMIND / Bridge scenes carry no "<key>_cam_extrinsics" in metadata.json: their pose lives in
    # the h5 per-frame stream ([T, 6], constant for a fixed camera).
    try:
        serial = camera_serial_for_key(metadata, camera_key)
        arr, ds_path = load_trajectory_camera_extrinsics(scene_dir, serial, "left")
    except Exception as e:
        # Never silent: falling through here renders the scene from the model's sample camera.
        print(f"[parse_camera_extrinsic] WARN: no {field} in metadata and no h5 extrinsics for "
              f"--camera_key={camera_key!r} in {scene_dir}: {e}. The render will use the model's "
              f"sample camera.")
        return None
    print(f"[parse_camera_extrinsic] no {field} in metadata; using row 0 of {ds_path}")
    return droid_6dof_to_w2c(arr[0])


def override_camera_intrinsics(
    camera, width: int, height: int,
    fov_x_deg: Optional[float] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
):
    """Override resolution and optionally FOV.

    Priority: explicit ``fov_x_deg`` > ``fx``/``fy`` from the camera intrinsics > keep the model's FOV.
    """
    camera.image_width = width
    camera.image_height = height

    if fov_x_deg is not None:
        camera.FoVx = float(np.radians(fov_x_deg))
    elif fx is not None:
        camera.FoVx = float(2.0 * np.arctan(width / (2.0 * fx)))

    if fy is not None and fov_x_deg is None:
        camera.FoVy = float(2.0 * np.arctan(height / (2.0 * fy)))
    else:
        camera.FoVy = float(
            2.0 * np.arctan(np.tan(camera.FoVx / 2.0) * height / width)
        )


def apply_extrinsic_to_camera(camera, extrinsic_matrix: np.ndarray):
    rotation_world_to_camera = extrinsic_matrix[:3, :3].astype(np.float32)
    translation_world_to_camera = extrinsic_matrix[:3, 3].astype(np.float32)

    camera.R = rotation_world_to_camera.T
    camera.T = translation_world_to_camera

    w2v = getWorld2View2(camera.R, camera.T, camera.trans, camera.scale)
    camera.world_view_transform = torch.tensor(w2v, device=camera.data_device).transpose(0, 1)
    camera.projection_matrix = getProjectionMatrix(
        znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx, fovY=camera.FoVy
    ).transpose(0, 1).to(camera.data_device)
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]


def _read_float_dataset(h5: h5py.Group, path: str) -> Optional[np.ndarray]:
    """Load a 1D/2D numeric dataset as float32, or None if missing or not a Dataset.

    HDF5 root satisfies ``"/" in h5`` and ``h5["/"]`` is a Group, so empty or root-only paths and
    non-Dataset objects are rejected.
    """
    p = (path or "").strip()
    if not p or p == "/":
        return None
    p = p.lstrip("/")
    if not p or p not in h5:
        return None
    obj = h5[p]
    if not isinstance(obj, h5py.Dataset):
        return None
    return np.asarray(obj[()], dtype=np.float32)


def load_trajectory_camera_extrinsics(
    scene_dir: str, serial: str, stereo: str
) -> Tuple[np.ndarray, str]:
    """Load per-frame [tx, ty, tz, rx, ry, rz] for one camera serial + stereo eye from trajectory.h5."""
    h5_path = os.path.join(scene_dir, "trajectory.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    ds_name = f"{serial}_{stereo}"
    ds_path = f"observation/camera_extrinsics/{ds_name}"
    with h5py.File(h5_path, "r") as f:
        arr = _read_float_dataset(f, ds_path)
        if arr is None and stereo != "left":
            ds_path_fallback = f"observation/camera_extrinsics/{serial}_left"
            arr = _read_float_dataset(f, ds_path_fallback)
            if arr is not None:
                ds_path = ds_path_fallback
        if arr is None:
            if "observation" in f and "camera_extrinsics" in f["observation"]:
                avail = sorted(f["observation"]["camera_extrinsics"].keys())
            else:
                avail = []
            raise KeyError(
                f"No dataset {ds_path!r} in {h5_path}. "
                f"Available observation/camera_extrinsics keys (sample): {avail[:20]}{'...' if len(avail) > 20 else ''}"
            )
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"Expected camera extrinsics [T, 6], got {arr.shape} at {ds_path}")
        return arr.astype(np.float32), ds_path


def load_joints_and_gripper(scene_dir: str):
    h5_path = os.path.join(scene_dir, "trajectory.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing {h5_path}")

    joint_key = "observation/robot_state/joint_positions"
    gripper_key = "observation/robot_state/gripper_position"

    with h5py.File(h5_path, "r") as f:
        joints = _read_float_dataset(f, joint_key)
        gripper = _read_float_dataset(f, gripper_key)
        if joints is None:
            raise KeyError(f"Dataset missing or not numeric: {joint_key} in {h5_path}")
        return joints, gripper


def _gripper_map_module():
    """Import the sibling ``gripper_map.py`` lazily (it pulls in mujoco, which the pure-render path does not need)."""
    import importlib
    import sys as _sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in _sys.path:
        _sys.path.insert(0, here)
    return importlib.import_module("gripper_map")


def build_gripper_map_for_model(
    model_xml_dir: str,
    n_samples: int = 33,
    settle_steps: int = 400,
):
    """``<model_path>/robot_xml`` -> a ``gripper_map.GripperMap``, or None if the model has no gripper.

    The map carries ``n_arm`` / ``qpos_idx`` / joint limits, so the caller never has to guess where the finger
    columns live. It handles all three chains -- FR3 2F-85 (7 arm + 8 finger, tendon LUT), UR5e 2F-85 (6 + 8,
    tendon LUT) and WidowX-250s (6 + 2, linear metre-valued prismatic fingers).

    Returns None (with a printed reason) instead of raising, so a model whose mechanism is not recognised
    still renders its arm.
    """
    try:
        gm_mod = _gripper_map_module()
        gmap = gm_mod.build_gripper_map(model_xml_dir, n_samples=n_samples, settle_steps=settle_steps)
    except Exception as e:
        print(f"[gripper_map] no usable gripper for {model_xml_dir}: {e}")
        return None
    print(f"[gripper_map] {gmap.xml_path}: kind={gmap.kind} actuator={gmap.actuator} "
          f"n_arm={gmap.n_arm} n_grip={gmap.n_grip} qpos_idx={np.asarray(gmap.qpos_idx).tolist()}  "
          f"driver open={gmap.lut[0, 0]:.4f} closed={gmap.lut[-1, 0]:.4f}", flush=True)
    return gmap


def render_scene_video(
    scene_dir: str,
    output_path: str,
    gaussians,
    bg_color,
    base_camera,
    fps: Union[int, float],
    camera_key: Optional[str],
    use_trajectory_camera: bool = False,
    camera_stereo: str = "left",
    gripper_map=None,
    gripper_convention: str = "closure",
    max_frames: int = 0,
    n_camera_frames: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
):
    """Render one scene to ``output_path``.

    ``frame_indices`` renders exactly those camera-grid steps (0-based, strictly increasing, all
    ``< min(T_h5, n_camera_frames)``) and takes precedence over ``max_frames`` (``max_frames`` frames at
    linspace over the camera grid). Without either, every step is rendered.

    ``gripper_map`` is a ``gripper_map.GripperMap`` (from :func:`build_gripper_map_for_model`): it knows
    ``n_arm`` and ``qpos_idx``, so any chain articulates.

    The scene's gripper log is closure (0 = open, 1 = closed) for every dataset, hence
    ``gripper_convention="closure"``.
    """
    joints, gripper = load_joints_and_gripper(scene_dir)
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2:
        raise ValueError(f"Expected joints with shape [T, J], got {joints.shape}")

    # The chain limits are the single source of truth for both the DoF count and the normalisation.
    lower, upper = gaussians.chain.get_joint_limits()
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    expected_dof = int(lower.shape[0])
    t_len = joints.shape[0]

    # Where the finger joints live in the chain: from the GripperMap when we have one.
    if gripper_map is not None:
        gs = np.asarray(gripper_map.gs, dtype=np.float64)
        lut = np.asarray(gripper_map.lut, dtype=np.float64)
        qpos_idx = np.asarray(gripper_map.qpos_idx, dtype=int)
        n_arm = int(gripper_map.n_arm)
        if n_arm + len(qpos_idx) != expected_dof:
            raise ValueError(f"chain has {expected_dof} dof but the gripper map describes "
                             f"{n_arm} arm + {len(qpos_idx)} gripper joints ({gripper_map.xml_path})")
    else:
        gs = lut = None
        qpos_idx = np.empty(0, dtype=int)
        n_arm = min(expected_dof, joints.shape[1])

    # --- assemble the full qpos in chain order, in native units (rad for hinges, m for the WidowX
    # prismatic fingers), then normalise once with the chain limits.
    qpos = np.zeros((t_len, expected_dof), dtype=np.float64)
    used_dof = min(joints.shape[1], n_arm)
    qpos[:, :used_dof] = joints[:, :used_dof]
    if joints.shape[1] > n_arm:
        print(f"[render_scene_video] NOTE: log has {joints.shape[1]} joint columns, chain arm is "
              f"{n_arm}; using the first {used_dof}")
    print(f"[render_scene_video] chain_dof={expected_dof} = {n_arm} arm + {len(qpos_idx)} gripper; "
          f"source_dof={joints.shape[1]} used={used_dof} T={t_len}")

    if lut is not None and gripper is not None and np.size(gripper) == 0:
        gripper = None      # an empty log is no log; hold the open pose below instead of indexing [-1]
        print("[render_scene_video] WARN: gripper dataset is empty; treating it as absent")
    if lut is not None and gripper is not None:
        closure = _gripper_map_module().closure_from_log(
            np.asarray(gripper, dtype=np.float64).reshape(-1)[:t_len],
            convention=gripper_convention,
        )
        grip_qpos = np.stack([np.interp(closure, gs, lut[:, k]) for k in range(lut.shape[1])], axis=1)
        n = min(len(closure), t_len)
        qpos[:n, qpos_idx] = grip_qpos[:n]
        if n < t_len:  # gripper log shorter than the joint log: hold the last commanded state
            qpos[n:, qpos_idx] = grip_qpos[n - 1]
            print(f"[render_scene_video] NOTE: gripper log has {n} of {t_len} rows; holding the last state")
        print(f"[render_scene_video] gripper closure "
              f"[{closure.min():.3f}, {closure.max():.3f}] -> driver "
              f"[{grip_qpos[:, 0].min():.4f}, {grip_qpos[:, 0].max():.4f}]")
    elif lut is not None:
        qpos[:, qpos_idx] = lut[0]  # no gripper log: hold the fully-open pose rather than a raw 0
        print("[render_scene_video] WARN: no gripper log in this scene; gripper held open")
    elif expected_dof > n_arm:
        print(f"[render_scene_video] WARN: gripper not articulated (no gripper map for this "
              f"model); {expected_dof - n_arm} trailing chain dof stay at 0")

    # gaussian_renderer lrs(..., pose_normalized=True) expects [-1, 1].
    concatenated_joints = np.clip(
        2.0 * (qpos - lower) / np.maximum(upper - lower, 1e-9) - 1.0, -1.0, 1.0,
    ).astype(np.float32)
    print(f"[render_scene_video] normalised with chain limits: t=0 qpos={np.round(qpos[0], 4)} -> "
          f"{np.round(concatenated_joints[0], 3)}")

    cam_extr_traj: Optional[np.ndarray] = None
    cam_traj_path: Optional[str] = None
    if use_trajectory_camera:
        if camera_key is None:
            raise ValueError("--trajectory_camera needs a real camera; use --camera_key wrist, ext1, or ext2.")
        md = load_scene_metadata(scene_dir)
        if md is None:
            raise FileNotFoundError(f"No metadata_*.json in {scene_dir}; cannot map camera_key to device serial.")
        serial = camera_serial_for_key(md, camera_key)
        cam_extr_traj, cam_traj_path = load_trajectory_camera_extrinsics(scene_dir, serial, camera_stereo)
        if cam_extr_traj.shape[0] != concatenated_joints.shape[0]:
            raise ValueError(
                f"Camera trajectory length {cam_extr_traj.shape[0]} != joint trajectory {concatenated_joints.shape[0]} "
                f"({cam_traj_path})"
            )
        print(
            f"[render_scene_video] viewpoint=trajectory {cam_traj_path} serial={serial} stereo={camera_stereo} "
            f"shape={cam_extr_traj.shape}"
        )

    camera = copy.deepcopy(base_camera)

    if cam_extr_traj is None:
        extrinsic = parse_camera_extrinsic(scene_dir, camera_key)
        if extrinsic is not None:
            apply_extrinsic_to_camera(camera, extrinsic)
            print(f"[render_scene_video] viewpoint=static (--camera_key={camera_key!r}).")
        else:
            print(
                "[render_scene_video] viewpoint=model sample camera only (no scene extrinsics for this camera_key); "
                "try wrist/ext1/ext2 or add camera_extrinsics.npy."
            )

    frames = []
    _T = concatenated_joints.shape[0]
    _sel = list(range(_T))
    # Raw DROID logs one post-final control step after the last camera capture (h5 T == mp4 frames + 1),
    # so a grid over T_h5 would land on different source steps than the video's grid over T_mp4.
    # Spanning min(T_h5, n_camera_frames) makes render, video and actions agree.
    # Converted RoboMIND/Bridge scenes have T_h5 == T_mp4, so this is a no-op for them.
    _T_grid = _T
    if n_camera_frames is not None and 0 < int(n_camera_frames) < _T:
        _T_grid = int(n_camera_frames)
        print(f"[render_scene_video] NOTE: camera mp4 has {_T_grid} frames but the trajectory has {_T} "
              f"steps; the frame grid spans the first {_T_grid} steps so it matches the video.")
    if frame_indices is not None:
        _sel = [int(i) for i in frame_indices]
        if not _sel or any(b <= a for a, b in zip(_sel, _sel[1:])) or _sel[0] < 0 or _sel[-1] >= _T_grid:
            raise ValueError(f"frame_indices must be strictly increasing within [0, {_T_grid}): "
                             f"got {_sel[:4]}..{_sel[-1:]} ({len(_sel)} frames)")
        print(f"[render_scene_video] explicit frame grid: {len(_sel)} frames {_sel[0]}..{_sel[-1]} of {_T_grid}")
        if gripper is not None:
            gripper = np.asarray(gripper)[_sel]
    elif max_frames and max_frames > 0 and _T_grid > max_frames:
        _sel = [int(i) for i in np.linspace(0, _T_grid - 1, max_frames)]
        if gripper is not None:
            gripper = np.asarray(gripper)[_sel]
    elif _T_grid < _T:
        _sel = list(range(_T_grid))
        if gripper is not None:
            gripper = np.asarray(gripper)[_sel]
    for t in _sel:
        if cam_extr_traj is not None:
            apply_extrinsic_to_camera(camera, droid_6dof_to_w2c(cam_extr_traj[t]))
        camera.joint_pose = torch.from_numpy(concatenated_joints[t]).to(camera.data_device)
        frame = render(camera, gaussians, bg_color)["render"]
        frame = torch.clamp(frame, 0.0, 1.0)
        frame_np = (frame.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        frames.append(frame_np)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    write_video(frames, output_path, fps)
    if gripper is not None:
        np.save(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_gripper.npy"), gripper)


def load_frame_indices_csv(path: str) -> Dict[str, List[int]]:
    """CSV with ``scene`` and ``frame_indices`` (space-separated ints) columns -> {scene: indices}."""
    with open(path, newline="", encoding="utf-8") as f:
        return {str(r["scene"]).strip(): [int(t) for t in str(r["frame_indices"]).split()]
                for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser("Render scene_* folders with a Dr. Robot Gaussian robot model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Trained Dr. Robot model directory (contains cfg_args, point_cloud/, robot_xml/)",
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="Path containing scene_* folders")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Renders go to <output_dir>/<scene>/drrobot_render.mp4 (+ render_meta.json)")
    parser.add_argument("--robot_type", type=str, required=True,
                        help="Embodiment key recorded in render_meta.json (panda | ur5e_2f85 | widowx)")
    parser.add_argument(
        "--background",
        type=str,
        default="black",
        choices=["white", "black"],
        help="Rasterizer clear color behind the Gaussians.",
    )
    def _fps_arg(value: str) -> Union[str, float]:
        """``meta`` = metadata.json fps; ``match``/``auto`` = ffprobe the MP4; else a positive float."""
        s = str(value).strip().lower()
        if s == "meta":
            return "meta"
        if s in ("match", "auto"):
            return "match"
        v = float(value)
        if v <= 0:
            raise argparse.ArgumentTypeError("--fps must be positive or meta/match/auto")
        return v

    parser.add_argument(
        "--fps",
        type=_fps_arg,
        default="match",
        help="Output video frame rate. match (default): ffprobe the MP4 for --camera_key. "
        "meta: metadata.json['fps'] (converted RoboMIND/Bridge scenes). A number pins a fixed rate.",
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="ext1",
        choices=["wrist", "ext1", "ext2"],
        help="Camera whose serial / pose / intrinsics are read from the scene.",
    )
    parser.add_argument(
        "--trajectory_camera",
        action="store_true",
        help="Use per-timestep extrinsics from trajectory.h5 (observation/camera_extrinsics/<serial>_<stereo>) "
        "instead of a single static pose.",
    )
    parser.add_argument(
        "--camera_stereo",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Stereo eye when reading trajectory.h5 camera extrinsics (falls back to _left if _right is missing).",
    )
    parser.add_argument("--width", type=int, default=None, help="Override render width.")
    parser.add_argument("--height", type=int, default=None, help="Override render height.")
    parser.add_argument(
        "--fov", type=float, default=None,
        help="Horizontal FOV in degrees (overrides the camera intrinsics). Default: from the intrinsics.",
    )
    parser.add_argument(
        "--render_size", type=parse_render_size, default="meta",
        help="meta (default): W x H from the scene metadata / the real mp4. native: probe the mp4 first. "
             "Or WxH, e.g. 640x480. --width/--height still override this.",
    )
    parser.add_argument(
        "--zed_intrinsics", type=str, default=None,
        help="Per-serial ZED 720p intrinsics JSON (the table the data scripts take as --intrinsics), used "
             "for DROID cameras without scene intrinsics. The built-in serials take precedence.",
    )
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Render N frames at linspace over the camera grid (17 for the full-span clip); "
                             "0 = every frame")
    parser.add_argument("--frame_indices_from", type=str, default="",
                        help="CSV with scene + frame_indices columns (render_jobs_*.csv of build_windows.py): "
                             "render exactly those source frames of each scene")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    if args.zed_intrinsics:
        n_added = load_zed_intrinsics_table(args.zed_intrinsics)
        print(f"[intrinsics] {args.zed_intrinsics}: {n_added} serials added to the built-in table")
    gaussians, bg_color, sample_cameras = load_model(args.model_path, background=args.background)
    base_camera = sample_cameras[0]

    gripper_map = build_gripper_map_for_model(os.path.join(args.model_path, "robot_xml"))
    grids = load_frame_indices_csv(args.frame_indices_from) if args.frame_indices_from else None

    scene_dirs = sorted(glob.glob(os.path.join(args.dataset_root, "scene_*")))
    if len(scene_dirs) == 0:
        raise RuntimeError(f"No scene directories found at {args.dataset_root}")

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        out_dir = os.path.join(args.output_dir, scene_name)
        output_path = os.path.join(out_dir, "drrobot_render.mp4")
        try:
            frame_indices = None
            if grids is not None:
                if scene_name not in grids:
                    raise KeyError(f"no frame_indices row in {args.frame_indices_from}")
                frame_indices = grids[scene_name]

            cam = copy.deepcopy(base_camera)

            # --- per-scene intrinsics (always returns a dict, falls back to defaults) ---
            intr = get_scene_camera_intrinsics(scene_dir, args.camera_key)
            print(f"[{scene_name}] intrinsics: fx={intr['fx']:.2f} fy={intr['fy']:.2f}")

            w, h, size_src = resolve_render_size(scene_dir, args.camera_key, args.render_size)
            w_ovr, h_ovr = (args.width or w), (args.height or h)
            if (w_ovr, h_ovr) != (w, h):
                print(f"[{scene_name}] WARN: --width/--height force {w_ovr}x{h_ovr} over the "
                      f"camera's {w}x{h}; FoV comes from fx and the render width, so this crops "
                      f"the view rather than rescaling it (pass --fov to set it explicitly).")
                size_src = "--width/--height"
            w, h = w_ovr, h_ovr
            override_camera_intrinsics(
                cam, w, h,
                fov_x_deg=args.fov, fx=intr["fx"], fy=intr["fy"],
            )
            print(
                f"[{scene_name}] render: {w}x{h} ({size_src})  "
                f"FoVx={np.degrees(cam.FoVx):.1f}°  FoVy={np.degrees(cam.FoVy):.1f}°"
            )

            fps_use, fps_src = resolve_fps(scene_dir, args.camera_key, args.fps)
            print(f"[{scene_name}] output fps={fps_use:.4g} ({fps_src})")

            # Lay the frame grid over the camera video's decoded length (see render_scene_video).
            n_cam = None
            if frame_indices is not None or args.max_frames > 0:
                mp4 = recording_mp4_for_camera(scene_dir, args.camera_key)
                n_cam = count_frames(mp4) if mp4 else None
                if frame_indices is not None and n_cam is not None and frame_indices[-1] >= n_cam:
                    raise ValueError(f"frame_indices end at {frame_indices[-1]} but the camera mp4 has {n_cam} frames")

            render_scene_video(
                scene_dir=scene_dir,
                output_path=output_path,
                gaussians=gaussians,
                bg_color=bg_color,
                base_camera=cam,
                fps=fps_use,
                camera_key=args.camera_key,
                use_trajectory_camera=args.trajectory_camera,
                camera_stereo=args.camera_stereo,
                gripper_map=gripper_map,
                max_frames=args.max_frames,
                n_camera_frames=n_cam,
                frame_indices=frame_indices,
            )
            meta = {
                "robot_type": args.robot_type,
                "render_model": os.path.basename(os.path.abspath(args.model_path)),
                "gripper_mode": "raw",
                "fps": float(fps_use), "fps_source": fps_src,
                "width": int(w), "height": int(h), "size_source": size_src,
                "camera_key": args.camera_key, "camera_stereo": args.camera_stereo,
                "trajectory_camera": bool(args.trajectory_camera),
                "intrinsics": {k: float(v) for k, v in intr.items()},
                "frame_selection": ("explicit" if frame_indices is not None
                                    else "linspace" if args.max_frames > 0 else "all"),
            }
            if frame_indices is not None:
                meta["frame_indices"] = [int(i) for i in frame_indices]
            with open(os.path.join(out_dir, "render_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1)
            print(f"[OK] {scene_name} -> {output_path}")
        except Exception as e:
            print(f"[SKIP] {scene_name}: {e}")


if __name__ == "__main__":
    main()
