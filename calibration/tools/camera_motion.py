"""Group trajectories by camera pose and flag the ones where the camera was moved.

Static structure projects to the same pixels while the camera does not move, so the camera is compared through the
backgrounds, not image correlation (which mostly measures repositioned props):

    temporal-median background per trajectory   (the moving arm averages out)
      -> SIFT keypoints, Lowe-ratio matching
      -> RANSAC homography                      (repositioned props are rejected as outliers)
      -> median displacement of the inliers, in pixels

The displacements are bimodal (a few pixels for a fixed camera, tens to hundreds after a move). By default all pairs
of a group are compared and single-link clustered into poses; ``--vs_reference`` compares against the first
trajectory only. ``background_shift`` is also used by tools/bridge_sessions.py.

    python tools/camera_motion.py --scenes <traj>/<camera> ... [--thresh 6] [--out sheet.png]
    python tools/camera_motion.py --glob '<root>/*/*/camera_top' --group_by parent2
"""
import argparse
import collections
import glob as globmod
import itertools
import os

import cv2
import numpy as np


def median_background(image_dir, n=15, size=(640, 480)):
    """Static background of a trajectory: the moving arm is suppressed by a temporal median."""
    names = sorted(os.listdir(image_dir), key=lambda x: int(x.split('.')[0]))
    idx = np.linspace(0, len(names) - 1, min(n, len(names))).astype(int)
    frames = [cv2.resize(cv2.imread(os.path.join(image_dir, names[i])), size) for i in idx]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def background_shift(a, b, sift, matcher, min_inliers=12, ransac_px=3.0):
    """Median pixel displacement of static structure between two background frames.

    Returns (displacement_px, n_inliers); displacement is None when matching fails.
    """
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < min_inliers or len(kb) < min_inliers:
        return None, 0
    good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None, len(good)
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, ransac_px)
    if H is None or mask is None or mask.sum() < min_inliers:
        return None, len(good)
    inl = mask.ravel().astype(bool)
    return float(np.median(np.linalg.norm(pb[inl] - pa[inl], axis=1))), int(inl.sum())


def cluster_poses(D, thresh):
    """Single-link clustering of a pairwise displacement matrix -> a pose label per trajectory."""
    n = len(D)
    lab = [-1] * n
    g = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = g
        stack = [i]
        while stack:
            k = stack.pop()
            for j in range(n):
                if lab[j] < 0 and not np.isnan(D[k, j]) and D[k, j] <= thresh:
                    lab[j] = g
                    stack.append(j)
        g += 1
    return lab, g


def group_key(path, mode):
    """How to bucket scenes before comparing them within a bucket."""
    parts = os.path.normpath(path).split(os.sep)
    if mode == 'parent2':      # <task>/<episode>/<camera> -> episode-id prefix (RoboMIND: the date)
        return parts[-2][:4] if len(parts) >= 2 else 'all'
    if mode == 'parent':
        return parts[-2] if len(parts) >= 2 else 'all'
    return 'all'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='*', default=None, help='camera dirs, each containing images/')
    ap.add_argument('--glob', default=None, help="glob for camera dirs, e.g. '<root>/*/*/camera_top'")
    ap.add_argument('--group_by', default='parent2', choices=['all', 'parent', 'parent2'],
                    help='bucket scenes before comparing (default parent2 = RoboMIND recording date)')
    ap.add_argument('--thresh', type=float, default=6.0, help='px displacement that counts as a different pose')
    ap.add_argument('--vs_reference', action='store_true',
                    help='compare against the first trajectory only instead of clustering all pairs')
    ap.add_argument('--max_per_group', type=int, default=14, help='cap the pairwise matrix per group')
    ap.add_argument('--out', default=None, help='write a contact sheet of the detected moves here')
    args = ap.parse_args()

    scenes = list(args.scenes or [])
    if args.glob:
        scenes += sorted(globmod.glob(args.glob))
    scenes = [s for s in scenes if os.path.isdir(os.path.join(s, 'images'))]
    if not scenes:
        ap.error('no scenes with an images/ subdirectory')

    groups = collections.defaultdict(list)
    for s in scenes:
        groups[group_key(s, args.group_by)].append(s)
    for k in groups:
        groups[k].sort()

    sift = cv2.SIFT_create(nfeatures=1500)
    matcher = cv2.BFMatcher()
    print(f'{len(scenes)} trajectories in {len(groups)} groups; '
          f'>{args.thresh:.0f}px = a different camera pose\n')
    print(f"{'group':10s} {'n':>3s} {'poses':>6s} {'max px':>8s}  detail")
    print('-' * 88)

    moved = []
    for gkey in sorted(groups):
        ps = groups[gkey][:args.max_per_group]
        if len(ps) < 2:
            print(f'{gkey:10s} {len(ps):3d} {1:6d} {"-":>8s}  single trajectory')
            continue
        beds = [median_background(os.path.join(p, 'images')) for p in ps]
        n = len(ps)
        if args.vs_reference:
            d = [background_shift(beds[0], beds[j], sift, matcher)[0] for j in range(1, n)]
            vals = [x for x in d if x is not None]
            npose = 1 + sum(1 for v in vals if v > args.thresh)
            mx = max(vals) if vals else float('nan')
            det = ', '.join(f'{os.path.basename(os.path.dirname(ps[j+1]))} {d[j]:.0f}px'
                            for j in range(n - 1) if d[j] and d[j] > args.thresh)
        else:
            D = np.zeros((n, n))
            for i, j in itertools.combinations(range(n), 2):
                v, _ = background_shift(beds[i], beds[j], sift, matcher)
                D[i, j] = D[j, i] = v if v is not None else np.nan
            lab, npose = cluster_poses(D, args.thresh)
            iu = np.triu_indices(n, 1)
            finite = D[iu][~np.isnan(D[iu])]
            mx = finite.max() if len(finite) else float('nan')
            det = ''
            if npose > 1:
                for c in range(npose):
                    eps = [os.path.basename(os.path.dirname(ps[i])) for i in range(n) if lab[i] == c]
                    det += f'[pose {c+1}: {len(eps)} traj] '
                moved.append((gkey, ps, lab))
        flag = '' if npose == 1 else '  <-- camera moved'
        print(f'{gkey:10s} {n:3d} {npose:6d} {mx:8.1f}  {det}{flag}')

    n_multi = len(moved) if not args.vs_reference else None
    if n_multi is not None:
        print(f'\n{n_multi} of {len(groups)} groups contain more than one camera pose')

    if args.out and moved:
        tiles = []
        for gkey, ps, lab in moved[:4]:
            first = {}
            for i, l in enumerate(lab):
                first.setdefault(l, i)
            for l, i in sorted(first.items())[:3]:
                t = cv2.resize(median_background(os.path.join(ps[i], 'images')), (300, 225))
                t = cv2.copyMakeBorder(t, 24, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(t, f'{gkey} pose {l+1}: {os.path.basename(os.path.dirname(ps[i]))}',
                            (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
                tiles.append(t)
        if tiles:
            rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
            w = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
            cv2.imwrite(args.out, np.vstack(rows))
            print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
