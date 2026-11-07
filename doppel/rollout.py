"""
Chained long-horizon rollout.

An episode is cut into back-to-back 17-frame windows where frame 16 of window k is frame 0
of window k+1. Window 1 is conditioned on the real first frame; every later window on the
model's own last predicted frame, with that window's rendering and actions. After frame 0
the model never sees a real frame again. The result has ``1 + 16 * n_windows`` frames.

Command line::

    python -m doppel.rollout --config configs/emb8_render_action_4node.json \\
        --checkpoint checkpoints/doppel-emb8 --episode episode.json --out rollout.mp4

``episode.json``::

    {"name": "my_episode", "robot_type": "panda",
     "windows": [{"render": "w1_render.mp4", "actions": "w1_actions.npz", "video": "w1.mp4"},
                 {"render": "w2_render.mp4", "actions": "w2_actions.npz", "video": "w2.mp4"}]}

``video`` is needed for window 1 (its first frame starts the rollout; or give ``"image"`` at
the top level) and is optional afterwards (only used for the per-frame PSNR printout).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from doppel.data import (
    frames_to_uint8, load_video_and_render_frames, read_all_frames, resize_frames,
    subsample_frames, write_video,
)
from doppel.inference import DEFAULT_STATS_DIR, WAN_MODEL_ID, DoppelWorldModel, load_actions_file


def window_seed(name: str, k: int, base_seed: int = 42) -> int:
    return (base_seed + int(hashlib.sha1(f"{name}|{k}".encode()).hexdigest()[:8], 16)) % (2 ** 31)


def chained_rollout(wm: DoppelWorldModel, first_frame, windows: Sequence[Dict[str, Any]], robot_type: str,
                    seeds: Sequence[int], num_inference_steps: int = 50) -> np.ndarray:
    """``windows``: dicts with ``render`` (17 frames: PIL list / array / path) and ``actions`` (RAW (17, D)).
    Returns uint8 ``(1 + 16 * len(windows), H, W, 3)``."""
    preds: List[np.ndarray] = []
    cond = first_frame
    for k, w in enumerate(windows):
        pred = wm.predict(cond, w["render"], w["actions"], robot_type, seed=int(seeds[k]),
                          num_inference_steps=num_inference_steps)
        preds.append(pred if k == 0 else pred[1:])      # frame 0 of window k+1 = last frame of window k
        cond = Image.fromarray(pred[-1])
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chained long-horizon rollout with a Doppel model.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_path", default=WAN_MODEL_ID)
    ap.add_argument("--stats_dir", default=DEFAULT_STATS_DIR)
    ap.add_argument("--episode", required=True, help="episode JSON (see module docstring)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True, help="output .mp4; frames and per-frame PSNR also go to <out>.npz")
    a = ap.parse_args()

    ep = json.load(open(a.episode))
    base = os.path.dirname(os.path.abspath(a.episode))
    p = lambda x: x if os.path.isabs(x) else os.path.join(base, x)   # noqa: E731
    wm = DoppelWorldModel.from_pretrained(a.checkpoint, a.config, a.model_path, a.stats_dir)

    wins, gts = [], []
    for w in ep["windows"]:
        if w.get("video"):
            gt_p, rd_p = load_video_and_render_frames(p(w["video"]), p(w["render"]), wm.T)
            gts.append(frames_to_uint8(np.stack([np.asarray(f) for f in resize_frames(gt_p, wm.H, wm.W)])))
        else:
            rd_p = subsample_frames(read_all_frames(p(w["render"])), wm.T, w["render"])
            gts.append(None)
        wins.append({"render": rd_p, "actions": load_actions_file(p(w["actions"]))})
    if ep.get("image"):
        first = Image.open(p(ep["image"]))
    elif gts[0] is not None:
        first = Image.fromarray(gts[0][0])
    else:
        raise SystemExit("episode needs a top-level 'image' or a 'video' for window 1")
    name = ep.get("name", os.path.splitext(os.path.basename(a.episode))[0])
    seeds = [window_seed(name, k) for k in range(len(wins))]
    pred = chained_rollout(wm, first, wins, ep["robot_type"], seeds, a.steps)

    out: Dict[str, Any] = {"pred": pred, "seeds": np.asarray(seeds)}
    if all(g is not None for g in gts):
        gt = np.concatenate([g if k == 0 else g[1:] for k, g in enumerate(gts)])
        mse = ((pred.astype(np.float32) - gt.astype(np.float32)) ** 2).mean((1, 2, 3))
        psnr = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-6))
        out.update(gt=gt, psnr=psnr)
        per_win = [float(psnr[1 + 16 * k: 17 + 16 * k].mean()) for k in range(len(wins))]
        print(f"[rollout] PSNR per window: {', '.join(f'{x:.2f}' for x in per_win)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_video(a.out, pred)
    np.savez_compressed(os.path.splitext(a.out)[0] + ".npz", **out)
    print(f"[rollout] wrote {a.out} ({len(pred)} frames)", flush=True)


if __name__ == "__main__":
    main()
