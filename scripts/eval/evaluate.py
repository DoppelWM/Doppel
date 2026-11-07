"""Evaluate a Doppel model on an evaluation set (one row per clip) -> one JSON line per clip.

    python scripts/eval/evaluate.py --config configs/emb8_render_action_4node.json \
        --checkpoint checkpoints/doppel-emb8 --evalset evalsets/test_droid.csv \
        --data_root data --out results/test_droid.jsonl [--shard 0/8] [--save_videos results/videos]

Eval-set CSV columns: ``video`` (17-frame clip), ``render``, ``actions`` (npz with a (17, D)
``state``), ``robot_type``, ``dataset``, and optionally ``row_key`` (else derived from
dataset + video) and ``evalset``. Relative paths resolve against ``--data_root``.

Protocol (as in the paper): 50 sampling steps, no guidance, empty prompt, per-clip seed
``42 + int(sha1(row_key)[:8], 16)``; PSNR / SSIM / LPIPS over predicted frames 1-16; SCS with
SAM 2 (``--no_scs`` skips it). Rows already in ``--out`` are skipped, so a run can be resumed.
Aggregate with ``scripts/eval/summarize.py`` (its ``scs_objects`` column is the paper's SCS).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from doppel import data_multi as dm
from doppel.data import frames_to_uint8, load_video_and_render_frames, resize_frames, write_video
from doppel.inference import DEFAULT_STATS_DIR, WAN_MODEL_ID, DoppelWorldModel, clip_seed
from doppel.metrics import SCS, frame_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_path", default=WAN_MODEL_ID)
    ap.add_argument("--stats_dir", default=DEFAULT_STATS_DIR)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--rows", default="", help="comma-separated row indices (default: all)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process scores rows with index % n == i")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_scs", action="store_true")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--save_videos", default="", help="directory for pred_<row_key>.mp4")
    a = ap.parse_args()

    df = pd.read_csv(a.evalset, keep_default_na=False)
    if "row_key" not in df.columns or (df["row_key"] == "").any():
        df["row_key"] = [dm.row_key(r) for r in df.to_dict("records")]
    i, n = (int(x) for x in a.shard.split("/"))
    idx = [int(x) for x in a.rows.split(",") if x] if a.rows else list(range(len(df)))
    idx = [r for r in idx if r % n == i]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["row_key"] for l in open(a.out) if l.strip()}
    idx = [r for r in idx if df.iloc[r].row_key not in done]
    if not idx:
        print("[eval] nothing to do")
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    absp = lambda p: p if os.path.isabs(p) else os.path.join(a.data_root, p)  # noqa: E731
    wm = DoppelWorldModel.from_pretrained(a.checkpoint, a.config, a.model_path, a.stats_dir)
    scs = None if a.no_scs else SCS()
    for r_i in idx:
        r = df.iloc[r_i]
        t0 = time.time()
        gt_p, rd_p = load_video_and_render_frames(absp(r.video), absp(r.render), wm.T)
        gt_p, rd_p = resize_frames(gt_p, wm.H, wm.W), resize_frames(rd_p, wm.H, wm.W)
        acts, _ = dm.load_actions(absp(r.actions), r.robot_type, wm.stats, field=wm.cfg["action_field"])
        seed = clip_seed(r.row_key, a.seed)
        pred = wm.predict(gt_p[0], rd_p, acts, r.robot_type, seed=seed,
                          num_inference_steps=a.steps, normalized_actions=True)
        gt = frames_to_uint8(np.stack([np.asarray(f) for f in gt_p]))
        rd = frames_to_uint8(np.stack([np.asarray(f) for f in rd_p]))
        rec = dict(row=r_i, row_key=str(r.row_key), dataset=str(r.dataset), robot_type=str(r.robot_type),
                   evalset=str(r.get("evalset", "")), seed=seed)
        rec.update(frame_metrics(pred, gt, first=1, lpips=not a.no_lpips))
        if scs is not None:
            rec.update(scs(pred, gt, rd))
        rec["seconds"] = round(time.time() - t0, 1)
        if a.save_videos:
            os.makedirs(a.save_videos, exist_ok=True)
            write_video(os.path.join(a.save_videos, f"pred_{r.row_key}.mp4"), pred)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[eval] row {r_i} {r.dataset} psnr(1-16)={rec['psnr']:.2f} ssim={rec['ssim']:.4f} "
              f"lpips={rec.get('lpips', float('nan')):.4f} scs={rec.get('scs', float('nan')):.4f} "
              f"({rec['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
