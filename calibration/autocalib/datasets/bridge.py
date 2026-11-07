"""BridgeData (raw release layout): WidowX-250s, fixed camera ``images0``.

Scene layout (``--scene``)::

    <raw_root>/.../<date>/raw/traj_groupG/trajN/images0/im_K.jpg
    <raw_root>/.../<date>/raw/traj_groupG/trajN/obs_dict.pkl      keys 'qpos' (T,6), 'full_state' (T,7; last col = gripper)
    <raw_root>/.../<date>/raw/traj_groupG/trajN/moge_static.pkl   MoGe intrinsics (created on demand, see intrinsics.py)
    <raw_root>/.../<date>/raw/traj_groupG/trajN/intrinsics.npy    optional fixed K (one focal per session)
    <raw_root>/.../<date>/raw/traj_groupG/trajN/tracker_data.pth  flow cache (tools/sweep_flow.py, RAFT); CoTracker3 if missing

Outputs next to the data: ``extrinsics.npy`` (w2c 4x4), ``intrinsics.npy`` (3x3), ``calibration_status.json``,
and the trajectory's entry in ``--dict``.

Without ``--init_w2c``, ``traj0`` of a group starts from BRIDGE_INIT_W2C and ``trajN`` (N>0) warm-starts from
``traj0``'s result in ``--dict`` (one epoch at lr 1e-4). ``--extra_scenes`` adds trajectories of the same session
(same fixed camera) to one joint fit. ``--init_candidates`` / ``--init_pool_from_dict`` add start poses; every start
pose is expanded by the ``--init_grid_rot`` / ``--init_grid_n`` rotation grid and scored on the first frames.
``--init_burnin_epochs`` optimises a few epochs from the best candidate of every start pose and continues from the
one whose rendered robot covers the most real motion; ``--select_best_epoch`` keeps the epoch with the best
motion recall.
"""
import json
import os
import pickle

import numpy as np
import torch

from .. import config
from ..calib_dict import bridge_keys, get_nested_dict, load_calib_dict, save_calib_dict, set_nested_dict
from ..core import CameraOptimizer
from ..flow import cached_cotracker_flow
from ..intrinsics import ensure_moge_intrinsics, moge_intrinsics_matrix
from ..model import load_robot_model
from ..pose_grid import grid_from_ranges
from .common import load_images


def load_trajectory(scene_path, max_frames=config.MAX_FRAMES):
    obs = pickle.load(open(os.path.join(scene_path, 'obs_dict.pkl'), 'rb'))
    joints = np.asarray(obs['qpos'])
    grippers = np.asarray(obs['full_state'])[:, -1:]
    joints_whole = np.concatenate([joints, config.bridge_fingers(grippers)], axis=1)
    images = load_images(os.path.join(scene_path, 'images0'), n=len(joints_whole))
    n = min(max_frames, len(images))
    return images[:n], joints_whole[:n]


def run(args):
    scene, raw_root = os.path.abspath(args.scene), os.path.abspath(args.raw_root)
    keys = bridge_keys(scene, raw_root)
    traj_name = keys[-1]
    if not traj_name.startswith('traj'):
        raise ValueError(f'scene dir must be named trajN, got {traj_name}')
    n = int(traj_name[4:])
    calib_dict = load_calib_dict(args.dict)

    resume = False
    w2c = config.BRIDGE_INIT_W2C
    if args.init_w2c:
        w2c = np.load(args.init_w2c)
    elif n != 0:
        traj0 = get_nested_dict(calib_dict, keys[:-1] + ['traj0'])
        if traj0 is not None and 'w2c' in traj0:
            w2c = np.array(traj0['w2c'])
            resume = True
            print('[bridge] warm-starting from traj0 of the same group (1 epoch, lr 1e-4)')

    # optional pool of extra initial poses (results of scenes that converged)
    pool = []
    for f in (getattr(args, 'init_candidates', None) or []):
        pool.append((os.path.basename(f), np.load(f)))
    if getattr(args, 'init_pool_from_dict', False):
        for name, entry in iter_leaves(calib_dict):
            if name != '/'.join(keys) and entry.get('status', {}).get('converged', True):
                pool.append((name, np.array(entry['w2c'])))
    if pool:
        print(f'[bridge] {len(pool)} candidate initial poses from the pool (+ the default)')

    gaussians, background_color = load_robot_model(args.model_path, background='white')
    images, joints_whole = load_trajectory(scene)
    T, H, W, _ = images.shape

    # one fixed camera per session: use the session focal from intrinsics.npy when present
    fixed = os.path.join(scene, 'intrinsics.npy')
    if os.path.exists(fixed):
        intrinsics = np.load(fixed)
        print(f'[bridge] intrinsics from {fixed}: fx={intrinsics[0, 0]:.2f}')
    else:
        moge = ensure_moge_intrinsics(scene, os.path.join(scene, 'images0'), max_frames=config.MAX_FRAMES)
        intrinsics = moge_intrinsics_matrix(moge, W, H)
    flow = cached_cotracker_flow(images, os.path.join(scene, 'tracker_data.pth'), n_frame=config.DENSE.cotracker_n_frame)
    fwd_flows_whole, fwd_valids_whole = flow['fwd_flows'], flow['fwd_valids']
    # optional: more trajectories of the SAME session (same fixed camera) -> more motion evidence for one pose
    segments = [(images, joints_whole, fwd_flows_whole, fwd_valids_whole)]
    for extra in (getattr(args, 'extra_scenes', None) or []):
        im_e, jo_e = load_trajectory(extra)
        fl_e = cached_cotracker_flow(im_e, os.path.join(extra, 'tracker_data.pth'), n_frame=config.DENSE.cotracker_n_frame)
        if im_e.shape[1:3] != (H, W):
            raise ValueError(f'{extra}: image size differs from the primary trajectory')
        segments.append((im_e, jo_e, fl_e['fwd_flows'], fl_e['fwd_valids']))
    if len(segments) > 1:
        print(f'[bridge] calibrating on {len(segments)} trajectories of the session ({sum(len(s[0]) for s in segments)} frames)')

    recipe = config.RECIPES[args.recipe] if args.recipe else config.DENSE
    opt = CameraOptimizer(gaussians, background_color, recipe, W, H, intrinsics)

    # start-pose scoring: every candidate expanded by the rotation grid, scored on the first frames
    best_w2c = w2c
    best_name = 'default'
    seed_best = [('default', w2c, float('nan'))] + [(n, p, float('nan')) for n, p in pool]
    if not resume and not args.skip_init_scoring:
        seeds = [('default', w2c)] + pool
        best_score = float('inf')
        ids = [0, 1, 2]
        ranking = []
        seed_best = []  # (name, best candidate of this seed, its score)
        for name, seed in seeds:
            candidates = grid_from_ranges(seed, rot_xy=args.init_grid_rot, rot_z=args.init_grid_rot / 2, n_rot=args.init_grid_n)
            seed_score, seed_w2c = float('inf'), None
            for cand in candidates:
                _, score = opt.score_pose(images[ids], joints_whole[ids], cand.numpy(), fwd_flows_whole[ids[:-1]], fwd_valids_whole[ids[:-1]], mode=args.init_score)
                ranking.append((score, name))
                if score < seed_score:
                    seed_score, seed_w2c = score, cand.numpy()
                if score < best_score:
                    best_score, best_w2c, best_name = score, cand.numpy(), name
            seed_best.append((name, seed_w2c, seed_score))
        for score, name in sorted(ranking)[:10]:
            print(f'[bridge] init candidate {name}: score {score:.5f}')
        print(f'[bridge] best initial pose: {best_name} (score {best_score:.5f})')

    # --camera_lr, --lr_min, --batch_size and --frame_stride apply to RoboMIND only; Bridge keeps the recipe values
    halving = getattr(args, 'lr_halving_epochs', None) or 4
    steps = getattr(args, 'steps_per_batch', None) or recipe.steps_per_batch
    batch_size = step_size = recipe.batch_size

    def run_schedule(init, epochs, base_lr, tag, segs=None):
        """Epoch/batch loop from a fresh shared camera over every trajectory; returns the w2c after the last batch."""
        opt.reset()
        params = None
        for epoch in range(epochs):
            for si, (im_s, jo_s, fl_s, va_s) in enumerate(segs or segments):
                Ts = len(im_s)
                for batch_idx in range(Ts // step_size):
                    ids = [batch_idx * step_size + k for k in range(batch_size) if batch_idx * step_size + k < Ts]
                    if len(ids) < 1:
                        continue
                    params = opt.optimize_batch(im_s[ids], jo_s[ids], init, fl_s[ids[:-1]], va_s[ids[:-1]],
                                                   camera_lr=base_lr * (0.5 ** (epoch // halving)), steps=steps, epoch=epoch, frame_ids=[si] + ids)
                    torch.cuda.empty_cache()
            print(f'[bridge] {tag} epoch {epoch} w2c:\n{params}')
            if tag == 'main':
                np.save(os.path.join(scene, 'extrinsics.npy'), params)
                np.save(os.path.join(scene, 'intrinsics.npy'), intrinsics)
                if track_epochs:
                    rec, prec = overlap_all(params)
                    epoch_track.append({'epoch': epoch, 'motion_recall': rec, 'motion_precision': prec, 'w2c': params.tolist()})
                    print(f'[bridge] epoch {epoch}: motion recall {rec:.3f} precision {prec:.3f}')
        return params

    overlap_ids = list(range(0, T - 1, max(1, (T - 1) // 24)))  # <= 25 frame pairs for the motion-overlap metric

    def overlap_all(w):
        """Motion overlap (recall, precision) averaged over every trajectory of the session."""
        rs, ps = [], []
        for im_s, jo_s, fl_s, va_s in segments:
            ids_s = list(range(0, len(im_s) - 1, max(1, (len(im_s) - 1) // 24)))
            r, p_ = opt.motion_overlap(jo_s, w, fl_s, va_s, ids_s)
            rs.append(r); ps.append(p_)
        return float(np.mean(rs)), float(np.mean(ps))
    track_epochs = bool(getattr(args, 'select_best_epoch', False))
    epoch_track = []

    # multi-start burn-in: a few epochs from the best candidate of every start pose, keep the best motion recall
    burnin = getattr(args, 'init_burnin_epochs', 0) or 0
    if burnin and not resume:
        topk = getattr(args, 'init_topk', None) or len(seed_best)
        trials = []
        for name, cand, score in sorted(seed_best, key=lambda t: t[2])[:topk]:
            w = run_schedule(cand, burnin, recipe.camera_lr, f'burn-in[{name}]', segs=segments[:1])  # burn-in on the primary trajectory only
            rec, prec = overlap_all(w)
            trials.append((rec, prec, name, w))
            print(f'[bridge] burn-in {name}: motion recall {rec:.3f} precision {prec:.3f} (3-frame score {score:.4f})')
        trials.sort(key=lambda t: -t[0])
        _, _, best_name, best_w2c = trials[0]
        print(f'[bridge] burn-in winner: {best_name}')

    epochs = 1 if resume else (args.epochs or recipe.epochs)
    base_lr = 0.0001 if resume else recipe.camera_lr
    params = run_schedule(best_w2c, epochs, base_lr, 'main')
    if track_epochs and epoch_track:
        best = max(epoch_track, key=lambda e: e['motion_recall'])
        if best['epoch'] != epoch_track[-1]['epoch']:
            print(f"[bridge] best epoch by motion recall: {best['epoch']} ({best['motion_recall']:.3f}) instead of the last one ({epoch_track[-1]['motion_recall']:.3f})")
        params = np.array(best['w2c'])
        np.save(os.path.join(scene, 'extrinsics.npy'), params)

    status = convergence_status(opt, joints_whole, best_w2c, params, T, fwd_flows_whole, fwd_valids_whole, overlap_ids, overlap_fn=overlap_all)
    status['n_trajectories'] = len(segments)
    status['n_frames'] = int(sum(len(sg[0]) for sg in segments))
    status['init'] = best_name
    if epoch_track:
        status['epoch_curve'] = [(e['epoch'], round(e['motion_recall'], 4)) for e in epoch_track]
        status['selected_epoch'] = int(max(epoch_track, key=lambda e: e['motion_recall'])['epoch'])
    with open(os.path.join(scene, 'calibration_status.json'), 'w') as f:
        json.dump(status, f, indent=1)
    print(f"[bridge] convergence check: {status}")

    set_nested_dict(calib_dict, keys, {'w2c': params.tolist(), 'intrinsic': intrinsics.tolist(), 'status': status})
    if args.dict:
        save_calib_dict(calib_dict, args.dict)
    print('[bridge] final w2c:\n', params)
    return params


def convergence_status(opt, joints_whole, init_w2c, final_w2c, T, fwd_flows, fwd_valids, pair_ids, min_recall=0.4, min_fraction=0.25, min_pixels=500, overlap_fn=None):
    """Converged = the rendered robot is still in view and covers >= min_recall of the real motion. Recall does not
    separate a wrong basin from the right one; check overlays (tools/overlay.py) or a second, independently seeded run."""
    ids = [0, T // 2, T - 1]
    px_init = opt.robot_pixels(joints_whole[ids], init_w2c)
    px_final = opt.robot_pixels(joints_whole[ids], final_w2c)
    visible = sum(px_final) >= max(min_pixels, min_fraction * sum(px_init))
    fn = overlap_fn or (lambda w: opt.motion_overlap(joints_whole, w, fwd_flows, fwd_valids, pair_ids))
    rec_i, prec_i = fn(init_w2c)
    rec_f, prec_f = fn(final_w2c)
    return {'converged': bool(visible and rec_f >= min_recall), 'motion_recall_init': rec_i, 'motion_precision_init': prec_i,
            'motion_recall_final': rec_f, 'motion_precision_final': prec_f, 'robot_pixels_init': px_init, 'robot_pixels_final': px_final}


def iter_leaves(d, prefix=()):
    """Yield ('a/b/c', leaf) for every {'w2c': ...} leaf of a nested calib dict."""
    if isinstance(d, dict):
        if 'w2c' in d:
            yield '/'.join(prefix), d
        else:
            for k, v in d.items():
                yield from iter_leaves(v, prefix + (k,))
