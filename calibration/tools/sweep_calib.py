"""Calibrate every session of a sweep manifest (the settings of the released Bridge and RoboMIND sweeps).

    python tools/sweep_calib.py --dataset robomind --out_root <sweep> --model_path models/ur5e_2f85 --gpus 0 1 2 3
    python tools/sweep_calib.py --dataset bridge   --out_root <bridge_out> --model_path models/widowx --gpus 0 1

Per session the first trajectory is ``--scene`` and the others ``--extra_scenes``. Settings: recipe ``dense``,
5 Adam steps per batch, camera lr halved every 15 epochs, 35 epochs. 27 start poses (rotation grid +-0.15 rad
around the session's ``init_w2c.npy``) are scored on the first frames with the DINOv2 feature loss; the best one gets
2 burn-in epochs (with ``--init_candidates`` in calibrate.py, every candidate pose does and the one with the best
motion recall continues), then the main run keeps the epoch with the best motion recall. RoboMIND: camera
lr 4e-4 with a 1e-5 floor and the binary gripper. Bridge: the Bridge driver does not apply --camera_lr and --lr_min
and runs at the recipe's 1e-3 without a floor.

Motion recall does not tell a wrong basin from the right one. Run a second, independently started sweep
(``--tag seedB --init_w2c <other start pose>``) and compare the two poses per session (``tools/export_extrinsics.py``
does this); disagreement beyond a few degrees means the session needs a look at the overlay (``tools/overlay.py``).

``--flow raft`` first builds missing flow caches (tools/sweep_flow.py). Logs and results go to
``<out_root>/_runs/<tag>/<session>/``; a session whose log contains ``final w2c`` is skipped, so the command can be
re-run.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def episodes(out_root, sid, v):
    """Scene dirs of one session/rig, from either manifest layout."""
    if v['episodes'] and 'dir' in v['episodes'][0]:                      # bridge rig manifest
        return [os.path.join(out_root, e['dir']) for e in v['episodes']]
    return [os.path.join(out_root, sid, e['task'], e['ep']) for e in v['episodes']]


def cam(scene, dataset):
    """Where per-scene files live: RoboMIND keeps them under camera_top/, Bridge in the scene dir itself."""
    return scene if dataset == 'bridge' else os.path.join(scene, 'camera_top')


def cache(dataset):
    return 'tracker_data.pth' if dataset == 'bridge' else 'tracker_data_n5.pth'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--dataset', default='robomind', choices=['robomind', 'bridge'])
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--gpus', nargs='+', type=int, required=True)
    ap.add_argument('--per_gpu', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--tag', default='seedA')
    ap.add_argument('--init_w2c', default=None, help='.npy start pose for every session (second, independent run)')
    ap.add_argument('--flow', default='none', choices=['none', 'raft'], help='build missing caches first')
    ap.add_argument('--sessions', nargs='*', default=None)
    ap.add_argument('--skip', default=None, help='file with session ids to skip (one per line)')
    ap.add_argument('--frame_stride', type=int, default=None, help='[robomind] spacing between optimisation windows')
    ap.add_argument('--batch_size', type=int, default=None, help='frames per optimisation window (recipe default 3); window step = batch size')
    ap.add_argument('--camera_lr', type=float, default=0.0004, help='the effective step is about batch_size x this (see autocalib/core.py)')
    ap.add_argument('--burnin', type=int, default=2, help='--init_burnin_epochs')
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()

    m = json.load(open(os.path.join(a.out_root, 'manifest.json')))
    skip = set(l.strip() for l in open(a.skip)) if a.skip else set()
    sessions = [s for s in m if (not a.sessions or s in a.sessions) and s not in skip]
    runs = os.path.join(a.out_root, '_runs', a.tag)
    os.makedirs(runs, exist_ok=True)

    if a.flow == 'raft':
        cmd = [sys.executable, os.path.join(REPO, 'tools', 'sweep_flow.py'), '--out_root', a.out_root, '--stage', 'flow',
               '--backend', 'raft', '--dataset', a.dataset, '--gpus'] + [str(g) for g in a.gpus] + ['--per_gpu', '2']
        print('[calib] building missing flow caches with RAFT:', ' '.join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)

    def job(sid):
        eps = episodes(a.out_root, sid, m[sid])
        scene = eps[0]
        init = a.init_w2c or os.path.join(cam(scene, a.dataset), 'init_w2c.npy')
        wd = os.path.join(runs, sid)
        os.makedirs(wd, exist_ok=True)
        cmd = [sys.executable, os.path.join(REPO, 'calibrate.py'), '--dataset', a.dataset, '--scene', scene]
        if len(eps) > 1:
            cmd += ['--extra_scenes'] + eps[1:]
        cmd += ['--model_path', a.model_path, '--recipe', 'dense', '--camera_lr', str(a.camera_lr),
                '--epochs', str(a.epochs), '--steps_per_batch', '5', '--lr_halving_epochs', '15', '--lr_min', '0.00001',
                '--grip_mode', 'binary', '--init_w2c', init, '--init_grid_rot', '0.15', '--init_grid_n', '3',
                '--init_burnin_epochs', str(a.burnin), '--select_best_epoch']
        if a.dataset == 'robomind':
            cmd += ['--cameras', 'camera_top']
        else:   # bridge keys its result dict by the path relative to --raw_root
            cmd += ['--raw_root', a.out_root]
        if a.batch_size:
            cmd += ['--batch_size', str(a.batch_size)]
        if a.frame_stride:
            cmd += ['--frame_stride', str(a.frame_stride)]
        return cmd, wd, scene

    def done(sid):
        p = os.path.join(runs, sid, 'run.log')
        return os.path.exists(p) and 'final w2c' in open(p, errors='ignore').read()

    def cached(sid):
        return all(os.path.exists(os.path.join(cam(e, a.dataset), cache(a.dataset))) for e in episodes(a.out_root, sid, m[sid]))

    pending = [s for s in sessions if not done(s)]
    print(f'[calib] {len(sessions)} sessions, {len(pending)} to run, {len(a.gpus) * a.per_gpu} slots, {a.epochs} epochs, tag {a.tag}', flush=True)
    if a.dry_run:
        for s in pending[:2]:
            print('  ', ' '.join(job(s)[0]))
        return
    slots = {(g, k): None for g in a.gpus for k in range(a.per_gpu)}
    active = {}
    t0 = time.time()
    while pending or active:
        for key, p in list(slots.items()):
            if p is not None and p.poll() is not None:
                sid = active.pop(p)
                print(f'[calib] DONE   {sid:8s} gpu{key[0]} rc={p.returncode} +{(time.time() - t0) / 60:.0f} min', flush=True)
                slots[key] = None
        for key, p in slots.items():
            if p is None and pending:
                sid = next((s for s in pending if cached(s)), None)
                if sid is None:
                    break
                pending.remove(sid)
                cmd, wd, scene = job(sid)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(key[0]), 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'MUJOCO_GL': 'egl'}
                slots[key] = subprocess.Popen(cmd, cwd=wd, env=env, stdout=open(os.path.join(wd, 'run.log'), 'w'), stderr=subprocess.STDOUT)
                active[slots[key]] = sid
                print(f'[calib] LAUNCH {sid:8s} gpu{key[0]} ({len(episodes(a.out_root, sid, m[sid]))} traj) +{(time.time() - t0) / 60:.0f} min', flush=True)
        time.sleep(30)

    # ---- summary: pose + status per session, and agreement with the other tag if it exists
    rows, results = [], {}
    for sid in sessions:
        scene = episodes(a.out_root, sid, m[sid])[0]
        c = cam(scene, a.dataset)
        st, ex = os.path.join(c, 'calibration_status.json'), os.path.join(c, 'extrinsics.npy')
        if not os.path.exists(ex):
            rows.append((sid, 'MISSING', '', '', '')); continue
        w = np.load(ex); s = json.load(open(st)) if os.path.exists(st) else {}
        results[sid] = {'w2c': w.tolist(), 'status': s}
        px = s.get('robot_pixels_final', []); pxi = s.get('robot_pixels_init', [])
        vis = f'{sum(px) / max(1, sum(pxi)):.2f}' if px and pxi else ''
        rows.append((sid, 'ok', f"{s.get('motion_recall_final', float('nan')):.3f}", f"{s.get('motion_precision_final', float('nan')):.3f}", vis))
    json.dump(results, open(os.path.join(runs, 'results.json'), 'w'), indent=1)
    print(f'\n[calib] {a.tag}: %-8s %-8s %7s %7s %10s' % ('session', 'state', 'recall', 'prec', 'robot_px'))
    for r in rows:
        print('              %-8s %-8s %7s %7s %10s' % r)
    print(f'[calib] results -> {os.path.join(runs, "results.json")}')


if __name__ == '__main__':
    main()
