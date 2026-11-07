"""Per-video calibration of ABC-130K top cameras: one pose per video, fitted from scratch.

    python -m yam.run_videos --queue videos.csv --out_root calib_videos --model <yam model dir> \\
        [--sam2_ckpt sam2.1_hiera_small.pt] [--claims calib_videos/_claims --worker_index i --n_workers n]

``videos.csv`` has columns uuid, task, mcap[, duration_s, camera, date]. The robot model and SAM 2 are loaded once.
Per video: ~12-20 frames are decoded (one every max(2 s, duration / 18)), the arms are segmented (masks.py), the
camera is fitted to the masks (fit.py, 6 fit / 6 held-out frames) and refined photometrically from that pose.
Outputs: <out_root>/<uuid>/{eps/<task>__<uuid8>/, fit/, photo/, result.json}. Videos with a result.json are
skipped; with --claims several processes (one or two per GPU) share one queue.
"""
import argparse
import csv
import json
import os
import time

import torch

from . import extract, fit, masks, station

PHOTO_ARGS = ["--loss", "photo", "--no_grid", "--n_seeds", "1", "--burnin_steps", "20", "--stages", "0.5:2:300:1e-3,1.0:1:400:4e-4"]


def claim(out_root, claims, uuid, worker):
    if os.path.exists(f"{out_root}/{uuid}/result.json"):
        return False
    if not claims:
        return True
    try:
        os.mkdir(f"{claims}/{uuid}")
    except FileExistsError:
        return False
    with open(f"{claims}/{uuid}/owner", "w") as f:
        f.write(f"{worker} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    return True


def calibrate_video(r, a, pc, sam):
    root = f"{a.out_root}/{r['uuid']}"
    ep = f"{root}/eps/{r['task']}__{r['uuid'][:8]}"
    T = {}
    t0 = time.time()
    if not os.path.exists(f"{ep}/states.json"):
        every = max(2.0, float(r.get("duration_s") or 60) / 18.0)
        extract.extract(r["mcap"], ep, every_s=round(every, 2), max_frames=20, task=os.path.basename(ep))
    T["extract"] = round(time.time() - t0, 1)
    t1 = time.time()
    masks.robot_masks(pc, sam, ep)
    T["masks"] = round(time.time() - t1, 1)
    common = ["--episodes", ep, "--model", a.model, "--iteration", str(a.iteration), "--holdout", "even",
              "--max_frames", str(a.max_frames), "--k_mode", "auto", "--base_sep", str(a.base_sep)]
    t1 = time.time()
    fs = fit.main(common + ["--out", f"{root}/fit"], pc=pc)["final"]
    T["mask_fit"] = round(time.time() - t1, 1)
    ps = None
    if a.photo:
        t1 = time.time()
        ps = fit.main(common + ["--out", f"{root}/photo", "--init_pose", f"{root}/fit/c2w.npy"] + PHOTO_ARGS, pc=pc)["final"]
        T["photo"] = round(time.time() - t1, 1)
    T["total"] = round(time.time() - t0, 1)
    return dict(uuid=r["uuid"], task=r["task"], camera=r.get("camera"), date=r.get("date"), status="ok", seconds=T,
                mask=dict(fit_iou=fs["fit_mean_iou"], holdout_iou=fs["holdout_mean_iou"]),
                photo=None if ps is None else dict(fit_iou=ps["fit_mean_iou"], holdout_iou=ps["holdout_mean_iou"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--model", required=True, help="YAM Dr.Robot model directory")
    ap.add_argument("--iteration", type=int, default=80000)
    ap.add_argument("--sam2_ckpt", default=None)
    ap.add_argument("--photo", type=int, default=1, help="run the photometric refinement")
    ap.add_argument("--max_frames", type=int, default=12)
    ap.add_argument("--base_sep", type=float, default=station.BASE_SEP)
    ap.add_argument("--claims", default="", help="claims dir: one atomic mkdir per video, so several workers share a queue")
    ap.add_argument("--worker_index", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if a.claims:
        os.makedirs(a.claims, exist_ok=True)
    rows = list(csv.DictReader(open(a.queue)))
    if a.n_workers > 1:   # each worker starts at its own offset; the claims keep the work disjoint
        off = (a.worker_index * len(rows)) // a.n_workers
        rows = rows[off:] + rows[:off]
    worker = f"{os.uname().nodename} pid {os.getpid()}"
    pc = station.load_model(a.model, a.iteration)
    sam = masks.load_sam2(a.sam2_ckpt)
    n_done = 0
    for r in rows:
        if a.limit and n_done >= a.limit:
            break
        if not claim(a.out_root, a.claims, r["uuid"], worker):
            continue
        os.makedirs(f"{a.out_root}/{r['uuid']}", exist_ok=True)
        try:
            res = calibrate_video(r, a, pc, sam)
        except Exception as e:  # noqa: BLE001  (a failed video is recorded and skipped)
            import traceback
            res = dict(uuid=r["uuid"], task=r["task"], camera=r.get("camera"), date=r.get("date"), status="error",
                       error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        json.dump(res, open(f"{a.out_root}/{r['uuid']}/result.json", "w"), indent=1)
        torch.cuda.empty_cache()
        n_done += 1
        print(f"{time.strftime('%H:%M:%S')} [{n_done}] {r['uuid'][:8]} {res['status']} {res.get('seconds', '')} {res.get('error', '')[:200]}", flush=True)


if __name__ == "__main__":
    main()
