"""Distance between two world-to-camera matrices: rotation (deg), translation and camera-centre distance (cm).

    python tools/compare_poses.py a.npy b.npy [--quiet]
"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_diff(a, b):
    ang = np.degrees(np.linalg.norm(R.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()))
    ca, cb = -a[:3, :3].T @ a[:3, 3], -b[:3, :3].T @ b[:3, 3]
    return float(ang), float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.linalg.norm(ca - cb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    a, b = np.load(args.a), np.load(args.b)
    ang, t, c = pose_diff(a, b)
    print(f'rotation diff {ang:.3f} deg | t diff {100*t:.2f} cm | camera-center diff {100*c:.2f} cm')
    if not args.quiet:
        np.set_printoptions(precision=5, suppress=True)
        print('A:\n', a, '\nB:\n', b)


if __name__ == '__main__':
    main()
