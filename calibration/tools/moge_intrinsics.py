"""CLI wrapper: estimate per-frame intrinsics with MoGe and write ``<scene>/moge_static.pkl``.

    python tools/moge_intrinsics.py --scene_path <dir with images0/> [--image_dir images0] [--out file.pkl]
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autocalib.intrinsics import estimate_moge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_path', required=True)
    ap.add_argument('--image_dir', default='images0')
    ap.add_argument('--out', default=None)
    ap.add_argument('--max_frames', type=int, default=200)
    args = ap.parse_args()
    res = estimate_moge(os.path.join(args.scene_path, args.image_dir), max_frames=args.max_frames)
    out = args.out or os.path.join(args.scene_path, 'moge_static.pkl')
    with open(out, 'wb') as f:
        pickle.dump(res, f)
    K = res['intrinsics']
    print(f'[moge_intrinsics] {len(K)} frames -> {out}; mean fx={K[:,0,0].mean():.3f} fy={K[:,1,1].mean():.3f}')


if __name__ == '__main__':
    main()
