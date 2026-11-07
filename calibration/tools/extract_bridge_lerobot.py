"""Extract BridgeData V2 trajectories from the `jnogga/bridge_data_v2_teleop` LeRobot-v3 mirror into the
raw BridgeData layout read by `autocalib.datasets.bridge`:

    <raw_root>/<original raw path ...>/raw/traj_groupG/trajN/images0/im_K.jpg
    <raw_root>/<original raw path ...>/raw/traj_groupG/trajN/obs_dict.pkl   {'qpos': (T,6), 'full_state': (T,7), ...}
    <raw_root>/<...>/lang.txt, source.json

The mirror is a full-resolution port of the raw `demos_8_17` release and keeps `observation.joint_position`
(the 6 WidowX-250s arm joints = raw `qpos`) and `observation.gripper_position` (= raw `full_state[:, 6]`,
1 = open). `full_state` is rebuilt as [x, y, z, roll, pitch, yaw, gripper] from `observation.cartesian`
(the calibration only reads `qpos` and `full_state[:, -1]`). Frames come from the `camera_k` video
mapped to `images{k}` by the episode's `cameras` field; only `images0` (the fixed primary camera) is written.

Usage:
    python tools/extract_bridge_lerobot.py --raw_root <dir> --episodes 123 456 ... [--meta_dir <dir with meta/>]
"""
import argparse
import json
import os
import pickle

import av
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

REPO = 'jnogga/bridge_data_v2_teleop'


def dl(rel, cache_dir):
    return hf_hub_download(REPO, rel, repo_type='dataset', local_dir=cache_dir)


def decode_segment(mp4, t_from, expected_ts, fps):
    """Decode the frames of one episode from a LeRobot-v3 concatenated video.

    Frame i of the episode sits at presentation time `t_from + expected_ts[i]` (the parquet `timestamp`
    column is episode-relative). The mirror's `to_timestamp` can include one boundary frame beyond the
    episode, so frames are matched by nearest timestamp rather than by slicing [from, to).
    """
    want = t_from + np.asarray(expected_ts, dtype=np.float64)
    tol = 0.5 / fps
    got = {}
    with av.open(mp4) as c:
        s = c.streams.video[0]
        s.thread_type = 'AUTO'
        tb = float(s.time_base)
        c.seek(int(max(want[0] - 1.0, 0) / tb), stream=s, backward=True, any_frame=False)
        for fr in c.decode(s):
            t = fr.pts * tb
            if t < want[0] - tol:
                continue
            if t > want[-1] + tol:
                break
            j = int(np.argmin(np.abs(want - t)))
            if abs(want[j] - t) < tol and j not in got:
                got[j] = fr.to_ndarray(format='rgb24')
    missing = [i for i in range(len(want)) if i not in got]
    if missing:
        raise RuntimeError(f'{mp4}: missing frames {missing[:5]}... for times starting {want[0]:.2f}')
    return [got[i] for i in range(len(want))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_root', required=True)
    ap.add_argument('--episodes', type=int, nargs='+', required=True)
    ap.add_argument('--meta_dir', default=None, help='dir containing meta/ (downloaded if missing)')
    ap.add_argument('--cache_dir', default=None, help='where HF files are downloaded (default <raw_root>/_hf_cache)')
    ap.add_argument('--camera', default='images0')
    args = ap.parse_args()
    cache = args.cache_dir or os.path.join(args.raw_root, '_hf_cache')
    meta_dir = args.meta_dir or cache
    ep_pq = os.path.join(meta_dir, 'meta/episodes/chunk-000/file_000.parquet')
    if not os.path.exists(ep_pq):
        ep_pq = dl('meta/episodes/chunk-000/file_000.parquet', cache)
    info = json.load(open(os.path.join(meta_dir, 'meta/info.json'))) if os.path.exists(os.path.join(meta_dir, 'meta/info.json')) else json.load(open(dl('meta/info.json', cache)))
    fps = info['fps']
    ep = pd.read_parquet(ep_pq).set_index('episode_index')

    for e in args.episodes:
        row = ep.loc[e]
        cams = list(row['cameras'])
        k = cams.index(args.camera)
        vkey = f'observation.images.camera_{k}'
        uuid = row['uuid']  # e.g. raw/bridge_data_v1/berkeley/<kitchen>/<task>/<date>/raw/traj_group0/traj38
        out = os.path.join(args.raw_root, uuid)
        if os.path.exists(os.path.join(out, 'source.json')) and os.path.exists(os.path.join(out, 'obs_dict.pkl')):
            print(f'[extract] ep {e}: already extracted -> {out}')
            continue
        os.makedirs(os.path.join(out, 'images0'), exist_ok=True)
        data_rel = info['data_path'].format(chunk_index=int(row['data/chunk_index']), file_index=int(row['data/file_index']))
        vid_rel = info['video_path'].format(video_key=vkey, chunk_index=int(row[f'videos/{vkey}/chunk_index']), file_index=int(row[f'videos/{vkey}/file_index']))
        data_pq = dl(data_rel, cache)
        mp4 = dl(vid_rel, cache)
        df = pd.read_parquet(data_pq)
        df = df[df['episode_index'] == e].sort_values('frame_index')
        T = int(row['length'])
        assert len(df) == T, (len(df), T)
        avail = np.stack(df[f'{vkey}_available'].values) if f'{vkey}_available' in df else np.ones(T, bool)
        assert avail.all(), f'episode {e}: {args.camera} not available on all frames'
        frames = decode_segment(mp4, float(row[f'videos/{vkey}/from_timestamp']), df['timestamp'].values, fps)
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(out, 'images0', f'im_{i}.jpg'), quality=95)
        qpos = np.stack(df['observation.joint_position'].values).astype(np.float64)
        cart = np.stack(df['observation.cartesian'].values).astype(np.float64)  # x y z qw qx qy qz
        grip = np.stack(df['observation.gripper_position'].values).astype(np.float64).reshape(T, 1)
        rpy = Rot.from_quat(cart[:, [4, 5, 6, 3]]).as_euler('xyz')
        full_state = np.concatenate([cart[:, :3], rpy, grip], axis=1)
        eef = np.stack(df['observation.eef_transform'].values).astype(np.float64)
        obs = {'qpos': qpos, 'full_state': full_state, 'state': full_state.copy(), 'qvel': np.stack(df['observation.joint_velocity'].values).astype(np.float64),
               'eef_pose_quat': eef, 'time_stamp': df['timestamp'].values.astype(np.float64)}
        with open(os.path.join(out, 'obs_dict.pkl'), 'wb') as f:
            pickle.dump(obs, f)
        with open(os.path.join(out, 'lang.txt'), 'w') as f:
            f.write(str(row['task']) + '\n')
        with open(os.path.join(out, 'source.json'), 'w') as f:
            json.dump({'repo': REPO, 'episode_index': int(e), 'uuid': uuid, 'video': vid_rel, 'data': data_rel, 'camera_key': vkey,
                       'metadata': row['metadata'], 'task': str(row['task']), 'length': T}, f, indent=1)
        print(f'[extract] ep {e}: {T} frames -> {out}')


if __name__ == '__main__':
    main()
