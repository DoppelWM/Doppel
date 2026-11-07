"""RoboMIND (extracted per-camera layout, ``tools/convert_robomind_h5.py``): UR5e or Franka FR3, Robotiq 2F-85.

Trajectory layout (``--scene`` is the trajectory dir)::

    <traj>/<camera>/images/*.jpg        (frame index in the file name)
    <traj>/<camera>/joints.npy          (T, 6) UR5e or (T, 7) Franka arm joints
    <traj>/<camera>/grippers.npy        (T, 1) gripper closedness in [0, 1]
    <traj>/<camera>/intrinsics.npy      (3, 3)
    <traj>/<camera>/tracker_data_n5.pth flow cache (tools/sweep_flow.py); CoTracker3 if missing

Outputs: ``<traj>/<camera>/extrinsics.npy``, ``<traj>/<camera>/calibration_status.json`` and ``<traj>/../results.pkl``.

The default recipe ``dense_robomind`` is the RobotArena∞ one (20 isolated frame pairs, lr 4e-3). The released UR5e poses
use ``--recipe dense`` with the overrides in ``tools/sweep_calib.py``, ``--grip_mode binary`` and ``--extra_scenes``
(several trajectories of one recording session fitted jointly). Start-pose candidates, burn-in and best-epoch
selection work as for Bridge (see ``bridge.py``).
"""
import json
import os
import pickle

import numpy as np
import torch

from .. import config
from ..core import CameraOptimizer
from ..flow import cached_cotracker_flow
from ..model import load_robot_model
from ..pose_grid import grid_from_ranges
from .bridge import convergence_status
from .common import load_images

OPTIMIZE_FRAME_NUM = 20

# Settled Robotiq 2F-85 joint values (MuJoCo stepped to rest at ctrl 0 / 255), in chain order
# [driver, coupler, spring_link, follower] x (right, left). The linear map in config.robotiq_fingers
# disagrees at closed: coupler -0.4 vs 0.001, follower -0.26 vs -0.757.
_OPEN = [0.003, 0.000, 0.003, -0.003]
_CLOSED = [0.782, 0.001, 0.781, -0.757]


def pair_ids(L, n_pairs=OPTIMIZE_FRAME_NUM):
    skip_step = max(1, L // n_pairs)
    ids = []
    for i in range(10, L, skip_step):
        if i >= L - 20:
            break
        ids.append(i)
    return [(i, i + 1) for i in ids]


def fingers_for(grippers, grip_mode, grip_thresh=None):
    g = np.asarray(grippers).reshape(-1, 1)
    if grip_mode != 'binary':
        return config.robotiq_fingers(g)
    lo, hi = float(g.min()), float(g.max())
    thr = grip_thresh if grip_thresh is not None else (lo + hi) / 2.0
    closed = g > thr
    print(f'[robomind] gripper binary: range [{lo:.3f}, {hi:.3f}] thresh {thr:.3f}; '
          f'{int(closed.sum())}/{len(closed)} frames closed')
    return np.where(closed, np.array([_CLOSED + _CLOSED]), np.array([_OPEN + _OPEN]))


def load_camera(cam_dir, grip_mode='legacy', grip_thresh=None, max_frames=config.MAX_FRAMES):
    grippers = np.load(os.path.join(cam_dir, 'grippers.npy')).reshape(-1, 1)
    joints = np.load(os.path.join(cam_dir, 'joints.npy'))
    intrinsics = np.load(os.path.join(cam_dir, 'intrinsics.npy'))
    joints_whole = np.concatenate([joints, fingers_for(grippers, grip_mode, grip_thresh)], axis=1)
    images = load_images(os.path.join(cam_dir, 'images'))
    n = min(max_frames, len(images), len(joints_whole))
    return images[:n], joints_whole[:n], intrinsics


def run(args):
    traj = os.path.abspath(args.scene)
    traj_name = os.path.basename(traj)
    results_file = os.path.join(os.path.dirname(traj), 'results.pkl')
    results = pickle.load(open(results_file, 'rb')) if os.path.exists(results_file) else {}
    gaussians, background_color = load_robot_model(args.model_path, background='white')
    recipe = config.RECIPES[args.recipe] if args.recipe else config.DENSE_ROBOMIND
    if getattr(args, 'batch_size', None) or getattr(args, 'camera_lr', None):
        import dataclasses
        recipe = dataclasses.replace(
            recipe,
            batch_size=getattr(args, 'batch_size', None) or recipe.batch_size,
            camera_lr=getattr(args, 'camera_lr', None) or recipe.camera_lr)
    epochs = args.epochs or recipe.epochs
    grip_mode = getattr(args, 'grip_mode', 'legacy')
    grip_thresh = getattr(args, 'grip_thresh', None)
    # 'dense_robomind' batches isolated frame pairs; 'dense' walks the whole clip in batches of 3
    pair_batches = recipe.name == 'dense_robomind'
    params = None

    for cam in args.cameras:
        cam_dir = os.path.join(traj, cam)
        init_default = np.load(args.init_w2c) if args.init_w2c else config.ROBOMIND_INIT_W2C[cam]
        images, joints_whole, intrinsics = load_camera(cam_dir, grip_mode, grip_thresh)
        T, H, W, _ = images.shape
        flow = cached_cotracker_flow(images, os.path.join(cam_dir, f'tracker_data_n{recipe.cotracker_n_frame}.pth'),
                                     n_frame=recipe.cotracker_n_frame)
        fwd_flows_whole, fwd_valids_whole = flow['fwd_flows'], flow['fwd_valids']
        print(f'[robomind] {cam}: {T} frames, recipe {recipe.name}, lr {recipe.camera_lr}, '
              f'cotracker window {recipe.cotracker_n_frame}, '
              f'{"per-pair batches" if pair_batches else "whole clip in batches of %d" % recipe.batch_size}')

        # more trajectories recorded by the SAME camera (same session) -> one pose, much more motion evidence
        segments = [(images, joints_whole, fwd_flows_whole, fwd_valids_whole)]
        for extra in (getattr(args, 'extra_scenes', None) or []):
            e_dir = os.path.join(os.path.abspath(extra), cam)
            im_e, jo_e, _ = load_camera(e_dir, grip_mode, grip_thresh)
            if im_e.shape[1:3] != (H, W):
                raise ValueError(f'{e_dir}: image size differs from the primary trajectory')
            fl_e = cached_cotracker_flow(im_e, os.path.join(e_dir, f'tracker_data_n{recipe.cotracker_n_frame}.pth'),
                                         n_frame=recipe.cotracker_n_frame)
            segments.append((im_e, jo_e, fl_e['fwd_flows'], fl_e['fwd_valids']))
        if len(segments) > 1:
            print(f'[robomind] {cam}: calibrating on {len(segments)} trajectories of the session '
                  f'({sum(len(s[0]) for s in segments)} frames)')

        opt = CameraOptimizer(gaussians, background_color, recipe, W, H, intrinsics)

        # ---- initial-pose scoring: candidate pool x rotation grid
        pool = [(os.path.basename(f), np.load(f)) for f in (getattr(args, 'init_candidates', None) or [])]
        if pool:
            print(f'[robomind] {len(pool)} candidate initial poses from the pool (+ the default)')
        best_w2c, best_name = init_default, 'default'
        seed_best = [('default', init_default, float('nan'))] + [(n, p, float('nan')) for n, p in pool]
        if not args.skip_init_scoring:
            seeds = [('default', init_default)] + pool
            best_score = float('inf')
            ids = [0, 1, 2]
            ranking, seed_best = [], []
            for name, seed in seeds:
                cands = grid_from_ranges(seed, rot_xy=args.init_grid_rot,
                                         rot_z=args.init_grid_rot / 2, n_rot=args.init_grid_n)
                s_score, s_w2c = float('inf'), None
                for cand in cands:
                    _, score = opt.score_pose(images[ids], joints_whole[ids], cand.numpy(),
                                              fwd_flows_whole[ids[:-1]], fwd_valids_whole[ids[:-1]],
                                              mode=args.init_score)
                    ranking.append((score, name))
                    if score < s_score:
                        s_score, s_w2c = score, cand.numpy()
                    if score < best_score:
                        best_score, best_w2c, best_name = score, cand.numpy(), name
                seed_best.append((name, s_w2c, s_score))
            for score, name in sorted(ranking)[:10]:
                print(f'[robomind] init candidate {name}: score {score:.5f}')
            print(f'[robomind] best initial pose: {best_name} (score {best_score:.5f})')

        halving = getattr(args, 'lr_halving_epochs', None) or 4
        steps = getattr(args, 'steps_per_batch', None) or recipe.steps_per_batch
        batch_size = recipe.batch_size
        step_size = getattr(args, 'frame_stride', None) or recipe.batch_size
        track_epochs = bool(getattr(args, 'select_best_epoch', False))
        epoch_track = []

        def overlap_all(w):
            """Motion overlap (recall, precision) averaged over every trajectory of the session."""
            rs, ps = [], []
            for im_s, jo_s, fl_s, va_s in segments:
                ids_s = list(range(0, len(im_s) - 1, max(1, (len(im_s) - 1) // 24)))
                r, p_ = opt.motion_overlap(jo_s, w, fl_s, va_s, ids_s)
                rs.append(r)
                ps.append(p_)
            return float(np.mean(rs)), float(np.mean(ps))

        def batches_for(Ts):
            if pair_batches:
                return [[i0, i1] for i0, i1 in pair_ids(Ts)]
            return [[b * step_size + k for k in range(batch_size) if b * step_size + k < Ts]
                    for b in range(Ts // step_size)]

        def run_schedule(init, n_epochs, base_lr, tag, segs=None):
            opt.reset()
            p = None
            for epoch in range(n_epochs):
                for si, (im_s, jo_s, fl_s, va_s) in enumerate(segs or segments):
                    for idb in batches_for(len(im_s)):
                        if len(idb) < 2:
                            continue
                        p = opt.optimize_batch(im_s[idb], jo_s[idb], init, fl_s[idb[:-1]], va_s[idb[:-1]],
                                                  camera_lr=max(getattr(args, 'lr_min', 0.0) or 0.0, base_lr * (0.5 ** (epoch // halving))),
                                                  steps=steps, epoch=epoch, frame_ids=[si] + idb)
                        torch.cuda.empty_cache()
                print(f'[robomind] {cam} {tag} epoch {epoch} w2c:\n{p}')
                if tag == 'main':
                    np.save(os.path.join(cam_dir, 'extrinsics.npy'), p)
                    if track_epochs:
                        rec, prec = overlap_all(p)
                        epoch_track.append({'epoch': epoch, 'motion_recall': rec,
                                            'motion_precision': prec, 'w2c': p.tolist()})
                        print(f'[robomind] epoch {epoch}: motion recall {rec:.3f} precision {prec:.3f}')
            return p

        # ---- multi-start burn-in
        burnin = getattr(args, 'init_burnin_epochs', 0) or 0
        if burnin:
            topk = getattr(args, 'init_topk', None) or len(seed_best)
            trials = []
            for name, cand, score in sorted(seed_best, key=lambda t: t[2])[:topk]:
                w = run_schedule(cand, burnin, recipe.camera_lr, f'burn-in[{name}]', segs=segments[:1])
                rec, prec = overlap_all(w)
                trials.append((rec, prec, name, w))
                print(f'[robomind] burn-in {name}: motion recall {rec:.3f} precision {prec:.3f} '
                      f'(3-frame score {score:.4f})')
            trials.sort(key=lambda t: -t[0])
            _, _, best_name, best_w2c = trials[0]
            print(f'[robomind] burn-in winner: {best_name}')

        params = run_schedule(best_w2c, epochs, recipe.camera_lr, 'main')
        if track_epochs and epoch_track:
            best = max(epoch_track, key=lambda e: e['motion_recall'])
            if best['epoch'] != epoch_track[-1]['epoch']:
                print(f"[robomind] best epoch by motion recall: {best['epoch']} ({best['motion_recall']:.3f}) "
                      f"instead of the last one ({epoch_track[-1]['motion_recall']:.3f})")
            params = np.array(best['w2c'])
            np.save(os.path.join(cam_dir, 'extrinsics.npy'), params)

        overlap_ids = list(range(0, T - 1, max(1, (T - 1) // 24)))
        status = convergence_status(opt, joints_whole, best_w2c, params, T,
                                    fwd_flows_whole, fwd_valids_whole, overlap_ids, overlap_fn=overlap_all)
        status['n_trajectories'] = len(segments)
        status['n_frames'] = int(sum(len(sg[0]) for sg in segments))
        status['init'] = best_name
        status['recipe'] = recipe.name
        if epoch_track:
            status['epoch_curve'] = [(e['epoch'], round(e['motion_recall'], 4)) for e in epoch_track]
            status['selected_epoch'] = int(max(epoch_track, key=lambda e: e['motion_recall'])['epoch'])
        with open(os.path.join(cam_dir, 'calibration_status.json'), 'w') as f:
            json.dump(status, f, indent=1)
        print(f'[robomind] {cam} convergence check: {status}')

        results.setdefault(traj_name, {})[cam] = {'extrinsics': params.tolist(), 'status': status}
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f'[robomind] {cam} final w2c:\n{params}')
    return params
