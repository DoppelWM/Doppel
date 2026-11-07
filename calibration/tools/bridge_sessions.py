"""Prepare Bridge sessions for ``tools/sweep_calib.py --dataset bridge``.

A session is one fixed camera: a kitchen + camera variation (``rig``), optionally restricted to one collection date
(the camera moves between dates, so the released poses are one per rig x date). Input CSV columns::

    session      id used for the output directory and manifest key
    kitchen      kitchen directory name in the raw layout (candidate scenes must contain /<kitchen>/)
    ref_scene    a trajectory of this session (path relative to --raw_root), defines the camera
    date         optional, e.g. 2023-03-15: only scenes whose path contains it
    init_w2c     optional start pose (.npy); otherwise put init_w2c.npy into the bundled scenes yourself

Stages (``--stage all`` runs them in order):

    group       scenes of the same kitchen whose median background is within --thresh px of the reference
                (SIFT + RANSAC homography, tools/camera_motion.py); keeps the --per_session closest
    intrinsics  one focal per session = median of the scenes' MoGe estimates (tools/moge_intrinsics.py must have
                written moge_static.pkl), principal point at the image centre, written to every scene's intrinsics.npy
    bundle      <out>/scenes/<session>/trajN/ with images0 (symlink), obs_dict.pkl, intrinsics.npy, init_w2c.npy
                and <out>/manifest.json in the sweep format

Then ``tools/sweep_flow.py --dataset bridge --stage flow --backend raft`` and ``tools/sweep_calib.py --dataset bridge``.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import re
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_motion import background_shift  # noqa: E402


def median_bg(image_dir, n=12, size=(640, 480)):
    fs = sorted(glob.glob(os.path.join(image_dir, '*.jpg')), key=lambda p: int(re.findall(r'(\d+)', os.path.basename(p))[-1]))
    idx = np.linspace(0, len(fs) - 1, min(n, len(fs))).astype(int)
    return np.median(np.stack([cv2.resize(cv2.imread(fs[i]), size) for i in idx]), axis=0).astype(np.uint8)


def stage_group(a, sessions):
    scenes = [os.path.dirname(p) for p in glob.glob(os.path.join(a.raw_root, '**', 'images0'), recursive=True)
              if os.path.exists(os.path.join(os.path.dirname(p), 'moge_static.pkl'))]
    print(f'{len(scenes)} scenes with MoGe intrinsics')
    sift, matcher = cv2.SIFT_create(), cv2.BFMatcher()
    groups = {}
    for s in sessions:
        ref = os.path.join(a.raw_root, s['ref_scene'])
        if not os.path.isdir(os.path.join(ref, 'images0')):
            print(f"{s['session']}: reference scene not found")
            continue
        b0 = median_bg(os.path.join(ref, 'images0'))
        same = [(0.0, ref)]
        for c in scenes:
            if c == ref or f"/{s['kitchen']}/" not in c or (s.get('date') and s['date'] not in c):
                continue
            d, _ = background_shift(b0, median_bg(os.path.join(c, 'images0')), sift, matcher)
            if d is not None and d <= a.thresh:
                same.append((d, c))
            if len(same) >= a.per_session * 3:
                break
        same.sort()
        groups[s['session']] = dict(scenes=[p for _, p in same[:a.per_session]], init=s.get('init_w2c') or None)
        print(f"{s['session']:40s} {len(groups[s['session']]['scenes']):3d} scenes")
    json.dump(groups, open(os.path.join(a.out, 'groups.json'), 'w'), indent=1)


def stage_intrinsics(a):
    groups = json.load(open(os.path.join(a.out, 'groups.json')))
    for sid, g in sorted(groups.items()):
        fx = [float(np.mean(np.asarray(pickle.load(open(os.path.join(s, 'moge_static.pkl'), 'rb'))['intrinsics'])[:, 0, 0]))
              for s in g['scenes'] if os.path.exists(os.path.join(s, 'moge_static.pkl'))]
        if not fx:
            print(f'{sid}: no MoGe estimates, skipped')
            continue
        f = float(np.median(fx))
        g['fx'] = f
        for s in g['scenes']:
            im = sorted(glob.glob(os.path.join(s, 'images0', '*.jpg')))
            if im:
                h, w = cv2.imread(im[0]).shape[:2]
                np.save(os.path.join(s, 'intrinsics.npy'), np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]))
        spread = 100 * (max(fx) - min(fx)) / np.mean(fx)
        print(f'{sid:40s} fx {f:7.1f} (spread {spread:.1f}% over {len(fx)} scenes)')
    json.dump(groups, open(os.path.join(a.out, 'groups.json'), 'w'), indent=1)


def stage_bundle(a):
    groups = json.load(open(os.path.join(a.out, 'groups.json')))
    manifest = {}
    for sid, g in sorted(groups.items()):
        eps = []
        for i, s in enumerate(g['scenes']):
            d = os.path.join(a.out, 'scenes', sid, f'traj{i}')   # the Bridge loader reads the trajectory index from the name
            os.makedirs(d, exist_ok=True)
            if not os.path.lexists(os.path.join(d, 'images0')):
                os.symlink(os.path.join(s, 'images0'), os.path.join(d, 'images0'))
            for f in ['obs_dict.pkl', 'intrinsics.npy', 'moge_static.pkl']:
                if os.path.exists(os.path.join(s, f)) and not os.path.exists(os.path.join(d, f)):
                    shutil.copy(os.path.join(s, f), d)
            if g.get('init') and os.path.exists(g['init']):
                shutil.copy(g['init'], os.path.join(d, 'init_w2c.npy'))
            eps.append({'dir': f'scenes/{sid}/traj{i}', 'src': os.path.relpath(s, a.raw_root),
                        'frames': len(glob.glob(os.path.join(s, 'images0', '*.jpg')))})
        manifest[sid] = {'episodes': eps, 'fx': g.get('fx')}
    json.dump(manifest, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)
    print(f"{len(manifest)} sessions, {sum(len(v['episodes']) for v in manifest.values())} scenes -> {a.out}/manifest.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', required=True, help='CSV: session, kitchen, ref_scene[, date, init_w2c]')
    ap.add_argument('--raw_root', required=True, help='raw Bridge layout (tools/extract_bridge_lerobot.py)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', default='all', choices=['all', 'group', 'intrinsics', 'bundle'])
    ap.add_argument('--per_session', type=int, default=10)
    ap.add_argument('--thresh', type=float, default=6.0, help='background shift (px) that still counts as the same camera')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage in ('all', 'group'):
        stage_group(a, list(csv.DictReader(open(a.sessions))))
    if a.stage in ('all', 'intrinsics'):
        stage_intrinsics(a)
    if a.stage in ('all', 'bundle'):
        stage_bundle(a)


if __name__ == '__main__':
    main()
