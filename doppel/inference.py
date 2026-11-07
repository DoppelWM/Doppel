"""
Inference: predict a 17-frame clip from a first frame, the embodiment's rendering and its actions.

Python::

    from doppel.inference import DoppelWorldModel
    wm = DoppelWorldModel.from_pretrained(checkpoint="checkpoints/doppel-emb8",
                                          config="checkpoints/doppel-emb8/config.json")
    video = wm.predict(first_frame, render_frames, actions, robot_type="panda", seed=0)

Command line (one clip; inputs as files)::

    python -m doppel.inference --config configs/emb8_render_action_4node.json \\
        --checkpoint checkpoints/doppel-emb8 \\
        --video clip.mp4 --render render.mp4 --actions actions.npz --robot_type panda --out pred.mp4

``--checkpoint`` is a released model directory (``transformer_overlay/`` safetensors) or a
training save's ``conditioner.pt``; ``--config`` is the released ``config.json``, a training
config (``configs/*.json``) or a save's ``meta.json``.

``--video`` supplies the first frame (and the ground truth, for a PSNR printout); use
``--image first_frame.png`` instead when there is no ground-truth video.

Sampling settings match the evaluation in the paper: 50 Euler steps of the flow-matching
scheduler, no classifier-free guidance, empty prompt, bf16 autocast, 240x432, 17 frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from doppel import data_multi as dm
from doppel.checkpoint import load_conditioner, model_config
from doppel.data import (
    frames_to_uint8, load_video_and_render_frames, read_all_frames, resize_frames,
    subsample_frames, write_video,
)

WAN_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
#: Hugging Face revision of the backbone the released models were trained from.
WAN_REVISION = "b184e23a8a16b20f108f727c902e769e873ffc73"
DEFAULT_STATS_DIR = os.path.join(dm.ASSETS_DIR, "action_stats")


def clip_seed(key: str, base_seed: int = 42) -> int:
    """Per-clip seed used by the evaluation: ``base_seed + int(sha1(key)[:8], 16)`` mod 2^31."""
    return (int(base_seed) + int(hashlib.sha1(str(key).encode()).hexdigest()[:8], 16)) % (2 ** 31)


def build_pipeline(cfg: Mapping[str, Any], checkpoint: Optional[str], model_path: str = WAN_MODEL_ID,
                   device: Union[str, torch.device] = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Wan 2.1 I2V pipeline with the conditioned DiT and (optionally) trained weights loaded.

    Returns ``(pipe, load_info)``."""
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPVisionModel

    from doppel.model import RenderConditionedWanI2VPipeline
    from doppel.train import _load_dit_only, _materialize_meta_submodules
    from doppel.train_utils import resolve_wan_model_root

    root = resolve_wan_model_root(model_path, revision=WAN_REVISION if model_path == WAN_MODEL_ID else None)
    dit = _load_dit_only(root, dtype, cfg["conditioning"],
                         redar_adapter_kwargs=cfg.get("redar_adapter_kwargs"),
                         action_only_adapter_kwargs=cfg.get("action_only_adapter_kwargs"))
    _materialize_meta_submodules(dit)
    vae = AutoencoderKLWan.from_pretrained(root, subfolder="vae", torch_dtype=dtype)
    enc = CLIPVisionModel.from_pretrained(root, subfolder="image_encoder", torch_dtype=torch.float32)
    pipe = RenderConditionedWanI2VPipeline.from_pretrained(
        root, transformer=dit, vae=vae, image_encoder=enc, torch_dtype=dtype)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.to(device)
    info = load_conditioner(pipe.transformer, checkpoint) if checkpoint else {"loaded": False}
    return pipe, info


def load_stats(robot_types: Sequence[str], stats_dir: str = DEFAULT_STATS_DIR) -> Dict[str, Dict[str, Any]]:
    """Action statistics ``doppel/assets/action_stats/<robot_type>.json`` for the given embodiments."""
    return dm.load_stats_map({rt: os.path.join(stats_dir, f"{rt}.json") for rt in robot_types})


class DoppelWorldModel:
    """A trained Doppel model: first frame + rendering + actions -> video."""

    def __init__(self, pipe, cfg: Mapping[str, Any], stats: Mapping[str, Mapping[str, Any]],
                 dtype: torch.dtype = torch.bfloat16):
        self.pipe = pipe
        self.cfg = dict(cfg)
        self.stats = dict(stats)
        self.dtype = dtype
        self.H, self.W, self.T = int(cfg["height"]), int(cfg["width"]), int(cfg["num_frames"])

    @classmethod
    def from_pretrained(cls, checkpoint: Optional[str], config: Union[str, Mapping[str, Any]],
                        model_path: str = WAN_MODEL_ID, stats_dir: str = DEFAULT_STATS_DIR,
                        device: Union[str, torch.device] = "cuda", dtype: torch.dtype = torch.bfloat16):
        """``checkpoint``: a released model directory (safetensors overlay) or a ``conditioner.pt``.
        ``config``: the released ``config.json``, a training config (``configs/*.json``) or a
        checkpoint's ``meta.json``."""
        if isinstance(config, str):
            with open(config, encoding="utf-8") as f:
                config = json.load(f)
        cfg = model_config(config)
        stats = load_stats(cfg["robot_types"], stats_dir)
        pipe, info = build_pipeline(cfg, checkpoint, model_path, device, dtype)
        print(f"[doppel] conditioning={cfg['conditioning']} action_pad_dim={cfg['action_pad_dim']} "
              f"embodiments={cfg['robot_types']} checkpoint={json.dumps(info)}", flush=True)
        return cls(pipe, cfg, stats, dtype)

    # ------------------------------------------------------------------ inputs
    def prepare_actions(self, actions: np.ndarray, robot_type: str, normalized: bool = False) -> torch.Tensor:
        """RAW ``(T, D)`` actions of ``robot_type`` -> the model's ``(1, T, action_dim)`` input:
        root unwrap + z-score (unless ``normalized``), pad slots 0, zero-extension to
        ``action_pad_dim`` for the 8-embodiment model."""
        if robot_type not in self.stats:
            raise KeyError(f"this model was not trained on {robot_type!r}; it knows {sorted(self.stats)}")
        if normalized:
            a = np.asarray(actions, np.float32)
            m = np.asarray(self.stats[robot_type]["dof_mask"], np.float32)
        else:
            a, m = dm.normalize_actions(actions, robot_type, self.stats)
        if self.cfg["action_pad_dim"]:
            a, m = dm.pad_actions(a, m, self.cfg["action_pad_dim"])
        return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0)

    def _frames(self, x) -> List[Image.Image]:
        if isinstance(x, str):
            x = subsample_frames(read_all_frames(x), self.T, x)
        if isinstance(x, np.ndarray):
            x = [Image.fromarray(f) for f in frames_to_uint8(x)]
        return resize_frames([f.convert("RGB") for f in x], self.H, self.W)

    # ------------------------------------------------------------------ predict
    @torch.inference_mode()
    def predict(self, image, render, actions: np.ndarray, robot_type: str, seed: int = 0,
                num_inference_steps: int = 50, normalized_actions: bool = False,
                drop_render: bool = False) -> np.ndarray:
        """Predict ``num_frames`` frames; returns uint8 ``(T, H, W, 3)`` (frame 0 = the input frame).

        image: PIL image / HxWx3 uint8 array / path of the first frame.
        render: the rendering, as T PIL frames, a (T, H, W, 3) array or a video path.
        actions: RAW ``(T, D)`` states in the layout of ``doppel/assets/embodiments.json``."""
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img = resize_frames([image.convert("RGB")], self.H, self.W)[0]
        device = self.pipe._execution_device
        acts = self.prepare_actions(actions, robot_type, normalized_actions).to(device=device, dtype=self.dtype)
        rd = None if drop_render else self._frames(render)
        gen = torch.Generator(device=device).manual_seed(int(seed))
        with torch.autocast(device_type=torch.device(device).type, dtype=self.dtype):
            out = self.pipe(image=img, prompt="", negative_prompt="", render_video=rd,
                            drop_render_conditioning=drop_render, actions=acts,
                            num_frames=self.T, height=self.H, width=self.W,
                            num_inference_steps=num_inference_steps, guidance_scale=1.0,
                            max_sequence_length=512, generator=gen, output_type="np", return_dict=False)
        return frames_to_uint8(out[0])[: self.T]


def load_actions_file(path: str, field: str = "state") -> np.ndarray:
    """Actions from ``.npz`` (key ``field``), ``.npy`` or ``.json`` (a list of lists)."""
    if path.endswith(".npz"):
        return np.asarray(np.load(path)[field], np.float32)
    if path.endswith(".npy"):
        return np.asarray(np.load(path), np.float32)
    with open(path, encoding="utf-8") as f:
        return np.asarray(json.load(f), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict one clip with a Doppel model.")
    ap.add_argument("--config", required=True,
                    help="released config.json, configs/*.json of the model, or its meta.json")
    ap.add_argument("--checkpoint", required=True,
                    help="released model dir (transformer_overlay/ safetensors) or conditioner.pt")
    ap.add_argument("--model_path", default=WAN_MODEL_ID, help="Wan 2.1 I2V 14B 480P Diffusers (HF id or dir)")
    ap.add_argument("--stats_dir", default=DEFAULT_STATS_DIR)
    ap.add_argument("--video", default=None, help="ground-truth clip: first frame + PSNR printout")
    ap.add_argument("--image", default=None, help="first frame (when there is no --video)")
    ap.add_argument("--render", required=True, help="rendering of the embodiment (video, >= 17 frames)")
    ap.add_argument("--actions", required=True, help="(17, D) raw actions: .npz (key 'state'), .npy or .json")
    ap.add_argument("--robot_type", required=True, help="see doppel/assets/embodiments.json")
    ap.add_argument("--seed", type=int, default=None, help="default: the evaluation's per-clip seed (--seed_key)")
    ap.add_argument("--seed_key", default=None, help="string hashed into the seed (the eval uses the clip's row_key)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True, help="output .mp4 (the raw uint8 frames are also saved as .npy)")
    a = ap.parse_args()
    if bool(a.video) == bool(a.image):
        ap.error("pass exactly one of --video / --image")

    wm = DoppelWorldModel.from_pretrained(a.checkpoint, a.config, a.model_path, a.stats_dir)
    gt = None
    if a.video:
        gt_p, rd_p = load_video_and_render_frames(a.video, a.render, wm.T)
        gt_p, rd_p = resize_frames(gt_p, wm.H, wm.W), resize_frames(rd_p, wm.H, wm.W)
        image, gt = gt_p[0], frames_to_uint8(np.stack([np.asarray(f) for f in gt_p]))
    else:
        image, rd_p = Image.open(a.image), a.render
    seed = a.seed if a.seed is not None else clip_seed(a.seed_key or os.path.basename(a.video or a.image))
    t0 = time.time()
    pred = wm.predict(image, rd_p, load_actions_file(a.actions), a.robot_type, seed=seed, num_inference_steps=a.steps)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_video(a.out, pred)
    np.save(os.path.splitext(a.out)[0] + ".npy", pred)
    msg = f"[doppel] wrote {a.out} ({len(pred)} frames, seed {seed}, {time.time() - t0:.0f} s)"
    if gt is not None:
        from doppel.metrics import frame_metrics
        m = frame_metrics(pred, gt, lpips=False)
        msg += f"; PSNR frames 1-{len(pred) - 1}: {m['psnr']:.2f} dB, SSIM {m['ssim']:.4f}"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
