#!/usr/bin/env python
"""Session-level camera calibration for Bridge (WidowX) and RoboMIND (UR5e).

    python calibrate.py --dataset bridge --model_path models/widowx --raw_root <raw> --scene <raw>/.../traj0 \
        --extra_scenes <more trajectories of the same session> --init_w2c init.npy
    python calibrate.py --dataset robomind --model_path models/ur5e_2f85 --cameras camera_top --scene <traj> \
        --extra_scenes <more trajectories of the same session> --recipe dense --grip_mode binary --init_w2c init.npy

``tools/sweep_calib.py`` runs this over a manifest of sessions with the settings used for the released poses.
"""
import argparse
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=['bridge', 'robomind'])
    p.add_argument('--scene', required=True, help='trajectory / scene directory (see the dataset module docstrings for the layout)')
    p.add_argument('--model_path', required=True, help='trained Dr.Robot model directory')
    p.add_argument('--raw_root', default=None, help='[bridge] raw dataset root; the result-dict key is the scene path relative to it')
    p.add_argument('--dict', default=None, help='[bridge] calib_dict.pkl to read warm starts from and append results to')
    p.add_argument('--cameras', nargs='+', default=['camera_left', 'camera_right'], help='[robomind] camera sub-directories to calibrate')
    p.add_argument('--init_w2c', default=None, help='.npy 4x4 world-to-camera initial guess (overrides the dataset default)')
    p.add_argument('--init_candidates', nargs='*', default=None, help='extra start poses (.npy w2c); the best-scoring one seeds the run')
    p.add_argument('--init_pool_from_dict', action='store_true', help='[bridge] also try every converged entry of --dict as an initial pose')
    p.add_argument('--init_score', default='feature', choices=['feature', 'flow', 'alpha_iou'], help='criterion for ranking start poses: DINOv2 ViT-B/14 feature loss (default), flow loss, or overlap of the rendered robot with the motion mask (alpha_iou)')
    p.add_argument('--recipe', default=None, choices=['dense', 'dense_robomind'], help='override the dataset default recipe (see autocalib/config.py)')
    p.add_argument('--epochs', type=int, default=None, help='override the recipe epoch count')
    p.add_argument('--init_grid_rot', type=float, default=0.0, help='half-range (rad) of the start-pose rotation grid (0 = only the start pose)')
    p.add_argument('--init_grid_n', type=int, default=1, help='rotation grid points per axis')
    p.add_argument('--skip_init_scoring', action='store_true', help='skip the initial-pose scoring / grid search')
    p.add_argument('--extra_scenes', nargs='*', default=None, help='more trajectories of the same session (fixed camera) to fit jointly with --scene')
    p.add_argument('--init_burnin_epochs', type=int, default=0, help='optimise this many epochs from the best candidate of every start pose and continue from the one with the best motion recall (0 = off)')
    p.add_argument('--init_topk', type=int, default=None, help='limit the burn-in to the k best-scored start poses')
    p.add_argument('--select_best_epoch', action='store_true', help='keep the epoch with the best motion recall instead of the last one')
    p.add_argument('--steps_per_batch', type=int, default=None, help='override the recipe Adam steps per batch')
    p.add_argument('--lr_halving_epochs', type=int, default=None, help='halve the camera lr every N epochs (recipe: 4)')
    p.add_argument('--lr_min', type=float, default=0.0, help='floor for the camera lr after halving (0 = no floor)')
    p.add_argument('--frame_stride', type=int, default=None, help='spacing between consecutive optimisation windows; default = recipe batch_size, i.e. every window')
    p.add_argument('--batch_size', type=int, default=None,
                   help='override the recipe frames per batch (dense: 3)')
    p.add_argument('--camera_lr', type=float, default=None,
                   help='override the recipe camera learning rate (dense: 0.001, dense_robomind: 0.004); the effective '
                        'step is about batch_size x this (see autocalib/core.py)')
    p.add_argument('--grip_mode', default='legacy', choices=['legacy', 'binary'],
                   help="[robomind] 'binary': Robotiq fingers at their settled open/closed joint values; "
                        "'legacy': linear map of the logged closedness")
    p.add_argument('--grip_thresh', type=float, default=None,
                   help='[robomind] closedness threshold for --grip_mode binary (default: midpoint of the episode range)')
    p.add_argument('--seed', type=int, default=None, help='seed torch/numpy (the camera twist starts at 1e-6 noise)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dataset == 'bridge' and not args.raw_root:
        raise SystemExit('--raw_root is required for --dataset bridge')
    import numpy as np
    import torch
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    from autocalib.datasets import DRIVERS
    t0 = time.time()
    params = DRIVERS[args.dataset](args)
    print(f'[calibrate] done in {time.time() - t0:.0f}s; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    return params


if __name__ == '__main__':
    main()
