"""Prepare sessions for the calibration sweep: select episodes, convert them, and build the flow caches.

RoboMIND input (``--survey_dir``)::

    <survey_dir>/camera_check.json                 sessions and camera poses (tools/robomind_sessions.py)
    <survey_dir>/h5/<task>__<episode>.hdf5         downloaded episodes (tools/fetch_robomind.py)

Stages (``--stage all`` runs select, convert, flow; every stage is resumable):

    select   up to ``--per_session`` episodes per session with a single camera pose (sessions whose camera moved are
             dropped unless ``--keep_multipose``; then the most common pose is used), live follower joints only
             unless ``--joint_source auto`` -> <out_root>/manifest.json
    convert  h5 -> camera_top images, joints, grippers, intrinsics (``--k``) and init_w2c.npy
    flow     flow cache for every camera dir, ``--gpus`` x ``--per_gpu`` workers, ``--shard i/n`` across nodes
    status   caches present per session

Bridge (``--dataset bridge``): only the flow stage, on the manifest written by tools/bridge_sessions.py.

The released poses used ``--backend raft`` (torchvision RAFT-large). The cache keeps the file name the loaders read
(``tracker_data_n5.pth`` for RoboMIND, ``tracker_data.pth`` for Bridge) whatever the backend.

    python tools/sweep_flow.py --survey_dir <dir> --out_root <sweep> --stage all --backend raft --gpus 0 1 2 3
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

REALSENSE_K = [609.3246, 609.7464, 317.7893, 257.3246]   # fx fy cx cy of the RoboMIND UR5e camera_top (640x480)
CONTACT = re.compile(r'drawer|trash|lid|cap|open|close|cover|uncap|insert')


# ------------------------------------------------------------------------------------------------ select
def _scan(args):
    """(key, frames, 'puppet'|'master'|'none') for one survey h5."""
    h5, key, min_frames = args
    import h5py
    try:
        with h5py.File(h5, 'r') as f:
            p = f['puppet/joint_position'][:] if 'puppet/joint_position' in f else None
            m = f['master/joint_position'][:] if 'master/joint_position' in f else None
    except Exception:
        return key, 0, 'none'
    if p is None or len(p) < min_frames:
        return key, 0 if p is None else len(p), 'none'
    live_p = np.ptp(p[:, :6], axis=0).max() > 1e-6
    live_m = m is not None and np.ptp(m[:, :6], axis=0).max() > 1e-6
    return key, len(p), 'puppet' if live_p else 'master' if live_m else 'none'


def stage_select(a):
    cc = json.load(open(os.path.join(a.survey_dir, 'camera_check.json')))
    sessions, episodes = cc['sessions'], cc['episodes']
    keys = sorted(episodes)
    with ProcessPoolExecutor(a.workers) as ex:
        info = {k: (T, src) for k, T, src in ex.map(_scan, [(os.path.join(a.survey_dir, 'h5', k + '.hdf5'), k, a.min_frames) for k in keys], chunksize=32)}
    allowed = {'puppet'} if a.joint_source == 'puppet' else {'puppet', 'master'}
    manifest, rows = {}, []
    for sid, s in sessions.items():
        if a.sessions and sid not in a.sessions:
            continue
        if s['n_poses'] > 1 and not a.keep_multipose:
            rows.append((sid, 'DROPPED: camera moved (%d poses)' % s['n_poses']))
            continue
        pose = max(s['poses'], key=lambda p: s['poses'][p]['n'])
        ok = [k for k, v in episodes.items() if v['session'] == sid and str(v['pose']) == str(pose) and info[k][1] in allowed]
        # puppet first, non-contact tasks first, then time order; spread over tasks
        ok.sort(key=lambda k: (info[k][1] != 'puppet', bool(CONTACT.search(k.split('__')[0])), k.split('__')[1]))
        chosen, per_task = [], collections.Counter()
        for cap in (1, 2, a.per_session):
            for k in ok:
                t = k.split('__')[0]
                if k in chosen or per_task[t] >= cap:
                    continue
                chosen.append(k)
                per_task[t] += 1
                if len(chosen) == a.per_session:
                    break
            if len(chosen) == a.per_session:
                break
        if len(chosen) < a.min_per_session:
            rows.append((sid, 'DROPPED: only %d usable episodes' % len(chosen)))
            continue
        manifest[sid] = {'pose': pose, 'episodes': [
            {'key': k, 'task': k.split('__')[0], 'ep': k.split('__')[1], 'frames': min(a.max_frames, info[k][0]),
             'joint_source': info[k][1]} for k in chosen]}
        rows.append((sid, '%d episodes (%d puppet), %d frames' % (
            len(chosen), sum(info[k][1] == 'puppet' for k in chosen), sum(min(a.max_frames, info[k][0]) for k in chosen))))
    os.makedirs(a.out_root, exist_ok=True)
    json.dump(manifest, open(os.path.join(a.out_root, 'manifest.json'), 'w'), indent=1)
    for sid, msg in sorted(rows):
        print(f'[select] {sid:8s} {msg}')
    print(f'[select] {len(manifest)} sessions, {sum(len(v["episodes"]) for v in manifest.values())} episodes, '
          f'{sum(e["frames"] for v in manifest.values() for e in v["episodes"])} frames -> {a.out_root}/manifest.json')


# ------------------------------------------------------------------------------------------------ convert
def cam_dir(out_root, sid, e):
    return os.path.join(out_root, sid, e['task'], e['ep'], 'camera_top')


def _convert(job):
    a, sid, e = job
    cam = cam_dir(a.out_root, sid, e)
    out = os.path.dirname(cam)
    if not os.path.isdir(os.path.join(cam, 'images')):
        h5 = os.path.join(a.survey_dir, 'h5', e['key'] + '.hdf5')
        r = subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'convert_robomind_h5.py'), '--h5', h5, '--out', out,
                            '--cameras', 'camera_top', '--n_arm_joints', '6', '--no_moge', '--max_frames', str(a.max_frames),
                            '--joint_source', 'auto' if e['joint_source'] == 'master' else 'puppet', '--master_lag', str(a.master_lag)],
                           capture_output=True, text=True, env={**os.environ, 'OMP_NUM_THREADS': '2'})
        if r.returncode != 0 or not os.path.isdir(os.path.join(cam, 'images')):
            return sid, e['key'], 'FAIL ' + (r.stderr.strip().splitlines() or ['?'])[-1][:120]
    fx, fy, cx, cy = a.k
    np.save(os.path.join(cam, 'intrinsics.npy'), np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]))
    if a.init_w2c:
        init = np.load(a.init_w2c)
    else:
        from autocalib import config
        init = np.asarray(config.ROBOMIND_INIT_W2C['camera_top'], float)
    np.save(os.path.join(cam, 'init_w2c.npy'), init)
    return sid, e['key'], 'ok'


def stage_convert(a):
    m = json.load(open(os.path.join(a.out_root, 'manifest.json')))
    jobs = [(a, sid, e) for sid, v in m.items() for e in v['episodes']]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(_convert, jobs))
    fails = [r for r in res if r[2] != 'ok']
    for r in fails:
        print('[convert]', *r)
    print(f'[convert] {len(res) - len(fails)}/{len(res)} episodes ready ({len(fails)} failed)')


# ------------------------------------------------------------------------------------------------ flow
def cache_name(a):
    return 'tracker_data.pth' if a.dataset == 'bridge' else f'tracker_data_n{a.n_frame}.pth'


def scene_dirs(a):
    """Camera dirs to build flow for, from either manifest layout."""
    m = json.load(open(os.path.join(a.out_root, 'manifest.json')))
    first = next(iter(m.values()))
    if 'episodes' in first and first['episodes'] and 'dir' in first['episodes'][0]:   # bridge rig manifest
        return [os.path.join(a.out_root, e['dir']) for v in m.values() for e in v['episodes']]
    return [cam_dir(a.out_root, sid, e) for sid, v in m.items() for e in v['episodes']]


def _flow_worker(a, cams):
    import torch
    from autocalib.datasets.common import load_images
    from autocalib.flow import cotracker_flow, raft_flow
    for cam in cams:
        cache = os.path.join(cam, cache_name(a))
        if os.path.exists(cache):
            print(f'[flow] skip {cam}', flush=True)
            continue
        sub = 'images0' if os.path.isdir(os.path.join(cam, 'images0')) else 'images'
        images = load_images(os.path.join(cam, sub))
        tmp = cache + f'.tmp{os.getpid()}'
        if a.backend == 'raft':
            torch.save(raft_flow(images, 'cuda'), tmp)
        else:
            torch.save(cotracker_flow(images, 'cuda', a.n_frame, grid_step=a.grid_step, bf16=a.bf16), tmp)
        os.replace(tmp, cache)
        print(f'[flow] done {cam} ({len(images)} frames)', flush=True)
        torch.cuda.empty_cache()


def stage_flow(a):
    cams = scene_dirs(a)
    i, n = (int(x) for x in a.shard.split('/'))
    cams = cams[i::n]
    todo = [c for c in cams if not os.path.exists(os.path.join(c, cache_name(a)))]
    print(f'[flow] shard {a.shard}: {len(cams)} camera dirs, {len(todo)} without a cache; grid_step={a.grid_step} bf16={a.bf16} backend={a.backend}', flush=True)
    if not todo:
        return
    slots = [g for g in a.gpus for _ in range(a.per_gpu)]
    procs = []
    for k, gpu in enumerate(slots):
        shard = todo[k::len(slots)]
        if not shard:
            continue
        cmd = [sys.executable, os.path.abspath(__file__), '--out_root', a.out_root, '--stage', 'flow', '--n_frame', str(a.n_frame),
               '--grid_step', str(a.grid_step), '--backend', a.backend, '--dataset', a.dataset] + (['--bf16'] if a.bf16 else []) + ['--worker'] + shard
        procs.append(subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2', 'MUJOCO_GL': 'egl'}))
    rc = [p.wait() for p in procs]
    print(f'[flow] shard {a.shard} finished; worker exit codes {rc}', flush=True)


# ------------------------------------------------------------------------------------------------ status
def stage_status(a):
    m = json.load(open(os.path.join(a.out_root, 'manifest.json')))
    tot = done = 0
    dirs = scene_dirs(a)
    k = 0
    for sid, v in m.items():
        n_ep = len(v['episodes'])
        have = sum(os.path.exists(os.path.join(d, cache_name(a))) for d in dirs[k:k + n_ep])
        k += n_ep
        tot += len(v['episodes']); done += have
        print(f'[status] {sid:8s} {have}/{len(v["episodes"])} caches')
    print(f'[status] {done}/{tot} caches present')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--survey_dir', help='dir with camera_check.json and h5/')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--dataset', default='robomind', choices=['robomind', 'bridge'],
                    help="'bridge': scenes carry images0/ and the loader reads tracker_data.pth (no _n5 suffix)")
    ap.add_argument('--stage', default='all', choices=['all', 'select', 'convert', 'flow', 'status'])
    # select
    ap.add_argument('--per_session', type=int, default=5)
    ap.add_argument('--min_per_session', type=int, default=3, help='drop sessions with fewer usable episodes')
    ap.add_argument('--min_frames', type=int, default=45)
    ap.add_argument('--max_frames', type=int, default=200)
    ap.add_argument('--joint_source', default='puppet', choices=['puppet', 'auto'],
                    help="'puppet': episodes with live follower joints only; 'auto': also allow the leader-arm fallback")
    ap.add_argument('--master_lag', type=int, default=5)
    ap.add_argument('--keep_multipose', action='store_true', help='keep sessions whose camera moved (default: drop them)')
    ap.add_argument('--sessions', nargs='*', default=None, help='restrict to these session ids')
    # convert
    ap.add_argument('--k', type=float, nargs=4, default=REALSENSE_K, metavar=('FX', 'FY', 'CX', 'CY'))
    ap.add_argument('--init_w2c', default=None, help='.npy written as init_w2c.npy into every camera dir (default: config.ROBOMIND_INIT_W2C["camera_top"])')
    ap.add_argument('--workers', type=int, default=16, help='CPU workers for select/convert')
    # flow
    ap.add_argument('--gpus', nargs='*', type=int, default=[0])
    ap.add_argument('--per_gpu', type=int, default=4)
    ap.add_argument('--shard', default='0/1', help='i/n: this node takes every n-th camera dir starting at i')
    ap.add_argument('--n_frame', type=int, default=5)
    ap.add_argument('--grid_step', type=int, default=1, help='[cotracker] 1 = every pixel; k > 1 = one point every k px')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--backend', default='raft', choices=['cotracker', 'raft'], help='raft: torchvision RAFT-large (used for the released poses); cotracker: CoTracker3 dense tracks')
    ap.add_argument('--worker', nargs='*', default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker is not None:
        _flow_worker(a, a.worker)
        return
    if a.stage in ('all', 'select'):
        assert a.survey_dir, '--survey_dir is required for select'
        stage_select(a)
    if a.stage in ('all', 'convert'):
        assert a.survey_dir, '--survey_dir is required for convert'
        stage_convert(a)
    if a.stage in ('all', 'flow'):
        stage_flow(a)
    if a.stage == 'status':
        stage_status(a)


if __name__ == '__main__':
    main()
