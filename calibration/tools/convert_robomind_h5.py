"""Convert a RoboMIND ``trajectory.hdf5`` episode into the per-camera layout read by ``autocalib.datasets.robomind``:

    <out>/<camera>/images/{000000..}.jpg
    <out>/<camera>/joints.npy      (T, n_arm_joints)
    <out>/<camera>/grippers.npy    (T, 1)  closedness, 0 = open, 1 = closed
    <out>/<camera>/intrinsics.npy  (3, 3)  MoGe estimate unless --no_moge (write the real K yourself then)

HDF5 keys: observations/rgb_images/<camera> (JPEG), puppet/joint_position (arm joints + gripper closedness),
master/joint_position (leader arm). Many UR5e episodes log a frozen puppet arm; ``--joint_source auto`` then takes
the leader arm shifted by ``--master_lag`` frames.

    python tools/convert_robomind_h5.py --h5 <trajectory.hdf5> --out <root>/<task>/<episode> \
        --cameras camera_top --n_arm_joints 6 --no_moge
"""
import argparse
import os
import pickle

import cv2
import h5py
import numpy as np


def find_keys(f):
    keys = []
    f.visit(lambda k: keys.append(k) if isinstance(f[k], h5py.Dataset) else None)
    return keys


def decode_frame(x):
    """RoboMIND stores frames as JPEG byte strings (variable-length uint8 arrays)."""
    if isinstance(x, (bytes, np.bytes_)):
        buf = np.frombuffer(x, np.uint8)
    else:
        x = np.asarray(x)
        if x.ndim == 3:  # already decoded HxWx3
            return x
        buf = x.astype(np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('could not decode frame')
    return img  # BGR (cv2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', required=True, help='episode output dir (camera subdirs are created inside)')
    ap.add_argument('--cameras', nargs='+', default=['camera_left', 'camera_right'])
    ap.add_argument('--max_frames', type=int, default=200)
    ap.add_argument('--bgr', action='store_true', default=True, help='frames stored as BGR (franka/ur embodiments)')
    ap.add_argument('--n_arm_joints', type=int, default=7,
                    help='arm DOF: 7 for franka (h5_franka_*), 6 for UR5e (h5_ur_1rgb). The gripper is read from '
                         'the next joint_position column when present.')
    ap.add_argument('--joint_source', default='puppet', choices=['puppet', 'master', 'auto'],
                    help="'auto': when the puppet arm stream is static, take the arm from master/joint_position "
                         "shifted by --master_lag; the gripper still comes from puppet")
    ap.add_argument('--master_lag', type=int, default=5,
                    help='leader -> follower lag in frames (4-5 measured on episodes that log both streams)')
    ap.add_argument('--allow_static_joints', action='store_true',
                    help='do not abort when the arm joints never move')
    ap.add_argument('--no_moge', action='store_true', help='skip MoGe intrinsics (write nothing)')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        keys = find_keys(f)
        print(f'[convert] {len(keys)} datasets in {args.h5}')
        jp_key = next((k for k in ['puppet/joint_position', 'franka/joint_position'] if k in f), None)
        if jp_key is None:
            jp_key = next((k for k in keys if k.endswith('joint_position') and 'master' not in k), None)
        assert jp_key, f'no joint_position dataset; keys: {keys}'
        jp = np.asarray(f[jp_key])
        print(f'[convert] joints from {jp_key} shape {jp.shape} (using {args.n_arm_joints} arm joints)')
        n_arm = args.n_arm_joints
        assert jp.shape[1] >= n_arm, f'{jp_key} has {jp.shape[1]} columns, need >= --n_arm_joints ({n_arm})'
        joints = jp[:, :n_arm].astype(np.float64)
        if args.joint_source != 'puppet' and len(np.unique(joints, axis=0)) < 2:
            mk = next((k for k in keys if k.endswith('master/joint_position')), None)
            if args.joint_source == 'master' or (args.joint_source == 'auto' and mk is not None):
                assert mk, 'no master/joint_position dataset to fall back to'
                mj = np.asarray(f[mk])[:, :n_arm].astype(np.float64)
                idx = np.clip(np.arange(len(mj)) - args.master_lag, 0, len(mj) - 1)  # joints[t] = master[t - lag]
                joints = mj[idx]
                print(f'[convert] puppet arm static -> joints from {mk} shifted +{args.master_lag} frames')
        # a frozen joint series would calibrate against a motionless robot
        n_unique = len(np.unique(joints, axis=0))
        if n_unique < 2:
            msg = (f'arm joints from {jp_key} are STATIC ({n_unique} unique pose over {len(joints)} frames) '
                   f'- this episode did not log follower joint states; pick another episode')
            if not args.allow_static_joints:
                raise SystemExit(f'[convert] ERROR {msg}')
            print(f'[convert] WARNING {msg}')
        if jp.shape[1] >= n_arm + 1:
            grip = jp[:, n_arm:n_arm + 1].astype(np.float64)
            grip_src = f'{jp_key}[:, {n_arm}]'
        else:
            gk = next((k for k in keys if 'gripper' in k.lower() and 'master' not in k), None)
            assert gk, 'no gripper column/dataset found'
            g = np.asarray(f[gk]).astype(np.float64).reshape(len(jp), -1)
            grip = g[:, :1]
            grip_src = gk
        # gripper column = closedness in [0, 1] (0 = open)
        grippers = np.clip(grip, 0, 1)
        print(f'[convert] gripper from {grip_src}: raw range [{grip.min():.4f}, {grip.max():.4f}] used as closedness (1 = closed)')

        T = min(len(joints), args.max_frames)
        for cam in args.cameras:
            ck = next((k for k in keys if k.endswith(f'rgb_images/{cam}')), None) or next((k for k in keys if k.endswith(f'/{cam}') and 'depth' not in k), None)
            if ck is None:
                print(f'[convert] WARNING camera {cam} not found (candidates: {[k for k in keys if "image" in k.lower()]})')
                continue
            d = f[ck]
            n = min(len(d), T)
            cam_dir = os.path.join(args.out, cam)
            os.makedirs(os.path.join(cam_dir, 'images'), exist_ok=True)
            for i in range(n):
                img = decode_frame(d[i])
                if not args.bgr:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(cam_dir, 'images', f'{i:06d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.save(os.path.join(cam_dir, 'joints.npy'), joints[:n])
            np.save(os.path.join(cam_dir, 'grippers.npy'), grippers[:n])
            print(f'[convert] {cam}: {n} frames from {ck} -> {cam_dir}')
            if not args.no_moge:
                import subprocess
                import sys
                tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools', 'moge_intrinsics.py')
                pkl = os.path.join(cam_dir, 'moge_static.pkl')
                subprocess.run([sys.executable, tool, '--scene_path', cam_dir, '--image_dir', 'images', '--out', pkl], check=True)
                K = pickle.load(open(pkl, 'rb'))['intrinsics']
                h, w = cv2.imread(os.path.join(cam_dir, 'images', '000000.jpg')).shape[:2]
                intr = np.array([[K[:, 0, 0].mean(), 0, w / 2], [0, K[:, 1, 1].mean(), h / 2], [0, 0, 1]])
                np.save(os.path.join(cam_dir, 'intrinsics.npy'), intr)
                print(f'[convert] {cam}: intrinsics fx={intr[0,0]:.2f} fy={intr[1,1]:.2f}')


if __name__ == '__main__':
    main()
