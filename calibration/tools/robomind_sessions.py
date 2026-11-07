"""Group RoboMIND UR5e episodes into recording sessions and camera poses -> ``<survey_dir>/camera_check.json``.

A session is the date + am/pm of the episode id (``MMDD_HHMMSS``). Inside a session the camera can still move, so
episodes are split into camera poses from their backgrounds:

    per episode   temporal median of 15 evenly spaced camera_top frames
    per session   static-structure mask = pixels whose background varies little across the session
                  (median absolute deviation < 10 grey levels), so relocated props cannot look like a camera move
    per pair      SIFT inside the mask, Lowe 0.8, RANSAC homography (3 px), median inlier displacement
                  (>= 20 inliers, else unmeasured)
    segmentation  change points where the median displacement over the 4x4 block of pairs straddling a boundary
                  exceeds --thr px, then segments that agree (median cross-distance < --thr) are merged

Input: ``<survey_dir>/h5/<task>__<episode>.hdf5`` (tools/fetch_robomind.py). The output is read by
``tools/sweep_flow.py --stage select``, which keeps one pose per session (sessions whose camera moved are dropped).

    python tools/robomind_sessions.py --survey_dir <dir> [--thr 8]
"""
import argparse
import glob
import json
import os
import time
from collections import OrderedDict, defaultdict

import cv2
import h5py
import numpy as np


def sess_key(ep):
    d, t = (ep.split('_') + ['12'])[:2]
    return d[:10] + ('am' if t[:2].isdigit() and int(t[:2]) < 12 else 'pm')


def ep_time(ep):
    d, t = (ep.split('_') + ['000000'])[:2]
    return d + t[:6]


def background(h5_path, cache, n_frames, camera='camera_top'):
    if os.path.exists(cache):
        return cv2.imread(cache, cv2.IMREAD_GRAYSCALE)
    with h5py.File(h5_path, 'r') as f:
        d = f[f'observations/rgb_images/{camera}']
        idx = np.linspace(0, max(len(d) - 1, 0), min(n_frames, max(len(d), 1))).astype(int)
        frames = [cv2.imdecode(np.frombuffer(bytes(d[int(i)]), np.uint8), cv2.IMREAD_COLOR) for i in idx]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.imwrite(cache, bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)


def displacement(fa, fb, bf):
    """Median inlier displacement (px) between two backgrounds' SIFT features, or None."""
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None, 0
    good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.8 * n.distance] if len(kb) > 1 else []
    if len(good) < 12:
        return None, len(good)
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None or inl is None or int(inl.sum()) < 20:
        return None, int(0 if inl is None else inl.sum())
    inl = inl.ravel().astype(bool)
    return float(np.median(np.linalg.norm(pb[inl] - pa[inl], axis=1))), int(inl.sum())


def static_mask(bgs, mad_thr=10, min_frac=0.05):
    stack = np.stack([cv2.GaussianBlur(b, (0, 0), 2) for b in bgs if b is not None]).astype(np.float32)
    med = np.median(stack, axis=0)
    m = (np.median(np.abs(stack - med), axis=0) < mad_thr).astype(np.uint8)
    m = cv2.dilate(cv2.erode(m, np.ones((7, 7), np.uint8)), np.ones((5, 5), np.uint8))
    if m.mean() < min_frac:     # degenerate: use the full frame
        m[:] = 1
    return m * 255


def segment(D, thr, W=4):
    """Pose label (1, 2, ... in order of first appearance) per episode from the pairwise displacement matrix."""
    n = len(D)
    cuts = []
    for b in range(n - 1):
        v = D[max(0, b - W + 1):b + 1, b + 1:min(n, b + 1 + W)]
        v = v[np.isfinite(v)]
        if len(v) >= 3 and np.median(v) > thr:
            cuts.append(b + 1)
    segs = np.split(np.arange(n), cuts)
    seg_lab = list(range(len(segs)))

    def seg_dist(a, b):
        v = D[np.ix_(segs[a], segs[b])]
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) >= 3 else np.inf

    merged = True
    while merged and len(set(seg_lab)) > 1:     # merge segments that agree (camera back at an earlier pose)
        merged = False
        groups = {}
        for si, l in enumerate(seg_lab):
            groups.setdefault(l, []).append(si)
        keys, best = sorted(groups), None
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                dv = [d for d in (seg_dist(si, sj) for si in groups[keys[x]] for sj in groups[keys[y]]) if np.isfinite(d)]
                if dv and np.median(dv) < thr and (best is None or np.median(dv) < best[0]):
                    best = (np.median(dv), keys[x], keys[y])
        if best:
            for si in groups[best[2]]:
                seg_lab[si] = best[1]
            merged = True
    for si in range(len(segs)):     # segments of <= 2 episodes without evidence of a move join a neighbour
        if len(segs[si]) <= 2 and len(set(seg_lab)) > 1:
            prev_ = si - 1 if si > 0 else None
            next_ = si + 1 if si + 1 < len(segs) else None
            dp = seg_dist(si, prev_) if prev_ is not None else np.inf
            dn = seg_dist(si, next_) if next_ is not None else np.inf
            if not np.isfinite(dp) and not np.isfinite(dn):
                seg_lab[si] = seg_lab[prev_] if prev_ is not None else seg_lab[next_]
            elif np.isfinite(dp) and dp < thr:
                seg_lab[si] = seg_lab[prev_]
            elif np.isfinite(dn) and dn < thr:
                seg_lab[si] = seg_lab[next_]
    lab = np.zeros(n, int)
    for si, l in enumerate(seg_lab):
        lab[segs[si]] = l + 1
    order, pose = {}, []
    for l in lab:
        order.setdefault(int(l), len(order) + 1)
        pose.append(order[int(l)])
    return pose


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--survey_dir', required=True)
    ap.add_argument('--thr', type=float, default=8.0, help='px displacement above which the camera counts as moved')
    ap.add_argument('--n_frames', type=int, default=15)
    ap.add_argument('--sessions', nargs='*', default=None)
    a = ap.parse_args()

    bg_dir = os.path.join(a.survey_dir, 'backgrounds')
    os.makedirs(bg_dir, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(a.survey_dir, 'h5', '*.hdf5'))):
        name = os.path.basename(p)[:-5]
        task, ep = name.rsplit('__', 1)
        rows.append(dict(name=name, task=task, episode=ep, h5=p, session=sess_key(ep)))
    sessions = OrderedDict()
    for r in sorted(rows, key=lambda r: ep_time(r['episode'])):
        sessions.setdefault(r['session'], []).append(r)
    if a.sessions:
        sessions = OrderedDict((k, v) for k, v in sessions.items() if k in a.sessions)

    t0 = time.time()
    bgs = {}
    for i, r in enumerate(x for v in sessions.values() for x in v):
        bgs[r['name']] = background(r['h5'], os.path.join(bg_dir, r['name'] + '.jpg'), a.n_frames)
        if i % 200 == 0:
            print(f'backgrounds {i}/{len(rows)} ({time.time() - t0:.0f}s)', flush=True)

    sift, bf = cv2.SIFT_create(nfeatures=1500), cv2.BFMatcher(cv2.NORM_L2)
    result = dict(threshold_px=a.thr, sessions={}, episodes={})
    for sk, eps in sessions.items():
        names = [r['name'] for r in eps]
        n = len(names)
        mask = static_mask([bgs[nm] for nm in names])
        feats = {nm: (sift.detectAndCompute(bgs[nm], mask) if bgs[nm] is not None else ([], None)) for nm in names}
        D = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                d, _ = displacement(feats[names[i]], feats[names[j]], bf)
                D[i, j] = D[j, i] = d if d is not None else np.nan
        np.fill_diagonal(D, 0.0)
        pose = segment(D, a.thr)
        per_pose = defaultdict(list)
        for r, p in zip(eps, pose):
            per_pose[p].append(r['episode'])
        result['sessions'][sk] = dict(n_episodes=n, n_poses=len(per_pose),
                                      poses={str(p): dict(n=len(v), first=v[0], last=v[-1]) for p, v in per_pose.items()})
        for r, p in zip(eps, pose):
            result['episodes'][r['name']] = dict(session=sk, pose=p)
        print(f'{sk}: {n} episodes -> {len(per_pose)} camera pose(s)', flush=True)
    json.dump(result, open(os.path.join(a.survey_dir, 'camera_check.json'), 'w'), indent=1)
    print(f"{len(result['sessions'])} sessions, {len(result['episodes'])} episodes -> {a.survey_dir}/camera_check.json")


if __name__ == '__main__':
    main()
