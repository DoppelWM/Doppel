"""Collect the sweep results into one table per dataset (one row per session).

    python tools/export_extrinsics.py --dataset robomind --out_root <sweep> --out robomind_sessions.csv \\
        [--tag seedA --compare_tag seedB]

Columns: session, scene (the trajectory the fit was run on, relative to out_root), T_base_cam (16 numbers,
row-major), tx ty tz qx qy qz qw, fx fy cx cy width height, motion_recall, selected_epoch, n_trajectories,
seed_deg / seed_cm (distance to the --compare_tag pose) and state.

``T_base_cam`` maps camera coordinates to the robot base frame (x_base = T_base_cam @ x_cam), OpenCV camera axes
(x right, y down, z forward), metres; it is the inverse of the fitted world-to-camera matrix. The intrinsics are the
ones the fit rendered with: fx, fy from the scene's intrinsics.npy and the principal point at the image centre.

state: ``ok`` when the two runs agree within --max_deg / --max_cm, ``unverified`` when there is no second run or
it disagrees. Look at the overlay of every unverified session before using it.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from compare_poses import pose_diff


def scene_dirs(out_root, sid, v, dataset):
    if v['episodes'] and 'dir' in v['episodes'][0]:
        scenes = [os.path.join(out_root, e['dir']) for e in v['episodes']]
    else:
        scenes = [os.path.join(out_root, sid, e['task'], e['ep']) for e in v['episodes']]
    return [s if dataset == 'bridge' else os.path.join(s, 'camera_top') for s in scenes]


def run_results(out_root, tag):
    p = os.path.join(out_root, '_runs', tag, 'results.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def image_size(cam):
    sub = 'images0' if os.path.isdir(os.path.join(cam, 'images0')) else 'images'
    first = sorted(glob.glob(os.path.join(cam, sub, '*.jpg')))[0]
    return Image.open(first).size


def row_for(sid, cam, w2c, status, other, a, out_root):
    T = np.linalg.inv(np.asarray(w2c, np.float64))
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    q = -q if q[3] < 0 else q
    K = np.load(os.path.join(cam, 'intrinsics.npy'))
    w, h = image_size(cam)
    deg = cm = None
    state = 'unverified'
    if other is not None:
        deg, _, c = pose_diff(np.asarray(w2c, np.float64), np.asarray(other, np.float64))
        cm = 100 * c
        state = 'ok' if deg <= a.max_deg and cm <= a.max_cm else 'unverified'
    return dict(session=sid, scene=os.path.relpath(cam, out_root), T_base_cam=' '.join(f'{x:.9g}' for x in T.reshape(-1)),
                tx=T[0, 3], ty=T[1, 3], tz=T[2, 3], qx=q[0], qy=q[1], qz=q[2], qw=q[3],
                fx=K[0, 0], fy=K[1, 1], cx=w / 2, cy=h / 2, width=w, height=h,
                motion_recall=status.get('motion_recall_final'), selected_epoch=status.get('selected_epoch'),
                n_trajectories=status.get('n_trajectories'), seed_deg=deg, seed_cm=cm, state=state)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['robomind', 'bridge'])
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--out', required=True, help='output .csv')
    ap.add_argument('--tag', default='seedA')
    ap.add_argument('--compare_tag', default='seedB')
    ap.add_argument('--max_deg', type=float, default=5.0)
    ap.add_argument('--max_cm', type=float, default=5.0)
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(a.out_root, 'manifest.json')))
    res, other = run_results(a.out_root, a.tag), run_results(a.out_root, a.compare_tag)
    rows, missing = [], []
    for sid, v in manifest.items():
        cam = scene_dirs(a.out_root, sid, v, a.dataset)[0]
        r = res.get(sid)
        if r is None and os.path.exists(os.path.join(cam, 'extrinsics.npy')):
            st = os.path.join(cam, 'calibration_status.json')
            r = {'w2c': np.load(os.path.join(cam, 'extrinsics.npy')).tolist(), 'status': json.load(open(st)) if os.path.exists(st) else {}}
        if r is None:
            missing.append(sid)
            continue
        rows.append(row_for(sid, cam, r['w2c'], r.get('status', {}), (other.get(sid) or {}).get('w2c'), a, a.out_root))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['session'])
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r['state'] == 'ok' for r in rows)
    print(f'{len(rows)} sessions ({n_ok} ok, {len(rows) - n_ok} unverified), {len(missing)} without a pose -> {a.out}')


if __name__ == '__main__':
    main()
