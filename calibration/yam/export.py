"""Per-video fits -> camera table (one row per video) with the pose policy and the quality gate.

    python -m yam.export --root calib_videos --out abc_cameras.parquet [--meta videos.csv]

Pose policy: the photometric refinement, unless it moved more than 3 deg / 5 cm from the mask pose or lowered the
held-out mask IoU by more than 0.05; then the mask pose.
Gate (``gate = pass``): held-out mask IoU >= 0.45; camera 1.3-2.2 m above the station floor and looking 20-75 deg
below the horizontal; and, when the same camera has >= 3 videos on the same day, within 5 cm / 3 deg of that day's
median pose (the medoid for the rotation).

Columns: T_base_cam (4x4 camera-to-station, OpenCV axes, metres, row-major 16 floats; station frame of the YAM
MJCF, left base at (0.2525, 0.31, 0.75), right base at y = 0.31 - base_sep, z up), T_leftbase_cam (the same pose in
the left-arm base frame), tx ty tz qx qy qz qw, fx fy cx cy width height. Intrinsics are given for the 640x480
pinhole frame: 640 streams as fitted; 848x480 streams centre-cropped to 640 (cx - 104); 1280x720 streams centre
960x720 resampled by 2/3. The pose does not depend on the crop.
``--meta`` (columns uuid, camera, date) supplies the camera id and recording day when result.json lacks them.
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import math
import os

import numpy as np

from .station import T_ABC_LEFT

MIN_IOU, MAX_DEV_CM, MAX_DEV_DEG = 0.45, 5.0, 3.0
HEIGHT_M, TILT_DEG = (1.3, 2.2), (20.0, 75.0)


def rot_deg(A, B):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2))))


def choose_pose(c2w_mask, c2w_photo, hold_mask, hold_photo, max_deg=3.0, max_cm=5.0, max_drop=0.05):
    """('photo' | 'mask', c2w): the photometric pose unless it collapsed."""
    if c2w_photo is None:
        return "mask", c2w_mask
    ang = rot_deg(c2w_mask, c2w_photo)
    dcm = float(np.linalg.norm(c2w_photo[:3, 3] - c2w_mask[:3, 3]) * 100)
    if ang <= max_deg and dcm <= max_cm and (hold_mask is None or hold_photo is None or hold_photo >= hold_mask - max_drop):
        return "photo", c2w_photo
    return "mask", c2w_mask


def height_tilt(c2w):
    """Camera height (m) and angle below the horizontal of the optical axis (deg)."""
    fwd = c2w[:3, 2]
    return float(c2w[2, 3]), math.degrees(math.asin(max(-1.0, min(1.0, -fwd[2]))))


def apply_gate(rows, min_iou=MIN_IOU, max_dev_cm=MAX_DEV_CM, max_dev_deg=MAX_DEV_DEG):
    """rows: dicts with c2w, holdout_iou, camera, date, status. Adds dev_cm, dev_deg, gate, reason in place."""
    groups = collections.defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            groups[(row["camera"], row["date"])].append(row)
    for rs in groups.values():
        if len(rs) < 3:
            for row in rs:
                row["dev_cm"] = row["dev_deg"] = float("nan")
            continue
        med = np.median(np.stack([row["c2w"][:3, 3] for row in rs]), 0)
        ref = min(rs, key=lambda row: np.linalg.norm(row["c2w"][:3, 3] - med))["c2w"]   # medoid of the day
        for row in rs:
            row["dev_cm"] = float(np.linalg.norm(row["c2w"][:3, 3] - med) * 100)
            row["dev_deg"] = rot_deg(row["c2w"], ref)
    for row in rows:
        if row["status"] != "ok":
            row["gate"], row["reason"] = "error", row.get("reason", "")
            continue
        why = []
        if row["holdout_iou"] is None or row["holdout_iou"] < min_iou:
            why.append(f"holdout_iou {row['holdout_iou']}")
        hgt, tilt = height_tilt(row["c2w"])
        if not (HEIGHT_M[0] <= hgt <= HEIGHT_M[1]) or not (TILT_DEG[0] <= tilt <= TILT_DEG[1]):
            why.append(f"implausible height {hgt:.2f} tilt {tilt:.0f}")
        if row["dev_cm"] == row["dev_cm"] and (row["dev_cm"] > max_dev_cm or row["dev_deg"] > max_dev_deg):
            why.append(f"off the day median {row['dev_cm']:.1f} cm / {row['dev_deg']:.1f} deg")
        row["gate"], row["reason"] = ("pass" if not why else "fail"), "; ".join(why)
    return rows


def pinhole_640x480(K, w, h):
    """(K (3x3) of the 640x480 pinhole frame, k_class) for the K and size a video was fitted at."""
    K = np.asarray(K, np.float64).reshape(3, 3).copy()
    if (w, h) == (1280, 720):
        K[0, 2] -= 160.0
        K[:2] *= 2.0 / 3.0
        return K, "crop1280"
    if (w, h) == (848, 480):
        K[0, 2] -= 104.0
        return K, "crop848"
    if (w, h) == (640, 480):
        return K, "calib"
    raise ValueError(f"unsupported stream size {w}x{h}")


def rot_to_quat_xyzw(R):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(R, np.float64)).as_quat()
    return -q if q[3] < 0 else q


def pose_id(T, nd=4):
    return hashlib.sha1(np.round(np.asarray(T, np.float64)[:3, :4], nd).tobytes()).hexdigest()[:12]


def camera_row(c2w, K, w, h, base_sep):
    """Table columns of one pose."""
    T = np.asarray(c2w, np.float64)
    q = rot_to_quat_xyzw(T[:3, :3])
    return dict(base_sep_m=base_sep, T_base_cam=T.reshape(-1).tolist(), T_leftbase_cam=(np.linalg.inv(T_ABC_LEFT) @ T).reshape(-1).tolist(),
                tx=T[0, 3], ty=T[1, 3], tz=T[2, 3], qx=q[0], qy=q[1], qz=q[2], qw=q[3],
                fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), width=int(w), height=int(h),
                pose_id=pose_id(T))


def collect(root, meta=None):
    """Read <root>/<uuid>/result.json + fit/ + photo/; returns rows ready for apply_gate."""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*", "result.json"))):
        r = json.load(open(p))
        m = (meta or {}).get(r["uuid"], {})
        row = dict(uuid=r["uuid"], task=r.get("task"), camera=m.get("camera") or r.get("camera") or r.get("station"),
                   date=m.get("date") or r.get("date"), status=r["status"])
        if r["status"] != "ok":
            row["reason"] = (r.get("error") or "")[:120]
            rows.append(row)
            continue
        d = os.path.dirname(p)
        c2w_m = np.load(f"{d}/fit/c2w.npy")
        c2w_p = np.load(f"{d}/photo/c2w.npy") if r.get("photo") else None
        kind, c2w = choose_pose(c2w_m, c2w_p, r["mask"]["holdout_iou"], (r.get("photo") or {}).get("holdout_iou"))
        st = json.load(open(f"{d}/{'photo' if kind == 'photo' else 'fit'}/status.json"))
        ep = st["episodes"][0]
        k_mode = ep.get("k_mode")
        if k_mode is None and os.path.exists(os.path.join(ep.get("dir", ""), "states.json")):
            hyp = json.load(open(os.path.join(ep["dir"], "states.json"))).get("cx_shift_hypothesis")
            k_mode = "cxfix" if hyp and hyp.get("K_hypothesis") else "calib"
        row.update(pose=kind, c2w=c2w, holdout_iou=r[kind]["holdout_iou"], K_fit=np.load(f"{d}/fit/K.npy"),
                   w_fit=int(ep["w"]), h_fit=int(ep["h"]), k_mode=k_mode, base_sep=float(st.get("base_sep", 0.61)))
        rows.append(row)
    return rows


def table(rows):
    out = []
    for row in rows:
        rec = dict(source_episode_id=row["uuid"], task=row.get("task"), camera_id=row.get("camera"), date=row.get("date"),
                   gate=row.get("gate"), reason=row.get("reason", ""))
        if row["status"] == "ok":
            K, k_class = pinhole_640x480(row["K_fit"], row["w_fit"], row["h_fit"])
            if k_class == "calib" and row.get("k_mode") == "cxfix":
                k_class = "cxfix"
            rec.update(pose_kind=row["pose"], quality_metric="heldout_mask_iou", quality=row["holdout_iou"],
                       dev_from_day_median_cm=row["dev_cm"], dev_from_day_median_deg=row["dev_deg"], k_class=k_class,
                       **camera_row(row["c2w"], K, 640, 480, row["base_sep"]))
        out.append(rec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="--out_root of run_videos.py")
    ap.add_argument("--out", required=True, help=".parquet or .csv")
    ap.add_argument("--meta", default=None, help="csv with uuid, camera, date")
    ap.add_argument("--all", action="store_true", help="also write failed videos (default: gate = pass only)")
    a = ap.parse_args(argv)
    meta = {r["uuid"]: r for r in csv.DictReader(open(a.meta))} if a.meta else None
    rows = apply_gate(collect(a.root, meta))
    import pandas as pd
    df = pd.DataFrame(table(rows))
    if not a.all:
        df = df[df.gate == "pass"]
    if a.out.endswith(".csv"):
        for c in ("T_base_cam", "T_leftbase_cam"):
            if c in df:
                df[c] = df[c].map(lambda x: " ".join(f"{v:.9g}" for v in x) if isinstance(x, list) else "")
        df.to_csv(a.out, index=False)
    else:
        df.to_parquet(a.out, index=False)
    c = collections.Counter(row.get("gate") for row in rows)
    print(f"{len(rows)} videos: {dict(c)} -> {a.out}")


if __name__ == "__main__":
    main()
