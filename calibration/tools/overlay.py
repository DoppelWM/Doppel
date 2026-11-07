"""Check a calibration by rendering the robot over the real frames.

    python tools/overlay.py --dataset bridge   --model_path models/widowx   --scene <trajN>
    python tools/overlay.py --dataset robomind --model_path models/ur5e_2f85 --scene <traj>/camera_top
    python tools/overlay.py --dataset robomind --model_path models/ur5e_2f85 --scene <traj>/camera_top \\
        --T_base_cam <16 numbers from the released table>

The pose is ``<scene>/extrinsics.npy`` (world-to-camera) unless ``--extrinsics`` or ``--T_base_cam`` is given.
Writes ``<out>/overlay_XXXX.png`` (real | real with the rendered robot tinted red | render) and ``overlay.mp4``.
The rendered silhouette should sit on the arm in every frame, including the gripper.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np  # noqa: E402


def load(dataset, scene, grip_mode='binary'):
    """(images (T,H,W,3) uint8, joints (T,J) incl. fingers, K (3,3), default extrinsics path)."""
    from autocalib.datasets import bridge, robomind
    from autocalib.intrinsics import ensure_moge_intrinsics, moge_intrinsics_matrix
    if dataset == 'bridge':
        images, joints = bridge.load_trajectory(scene)
        k_file = os.path.join(scene, 'intrinsics.npy')
        if os.path.exists(k_file):
            K = np.load(k_file)
        else:
            K = moge_intrinsics_matrix(ensure_moge_intrinsics(scene, os.path.join(scene, 'images0')), images.shape[2], images.shape[1])
    else:
        images, joints, K = robomind.load_camera(scene, grip_mode=grip_mode)
    return images, joints, K, os.path.join(scene, 'extrinsics.npy')


def render_frames(gaussians, bg, w2c, K, joints, W, H):
    """Yield (rgb uint8 (H,W,3), alpha float (H,W)) per joint vector."""
    import torch
    from drrobot.cameras import Camera_Pose
    from drrobot.renderer import render_gradio
    fovx, fovy = 2 * np.arctan(W / (2 * K[0, 0])), 2 * np.arctan(H / (2 * K[1, 1]))
    white, black = torch.ones(3, device='cuda'), torch.zeros(3, device='cuda')
    for q in joints:
        cam = Camera_Pose(torch.tensor(w2c).float().cuda(), fovx, fovy, W, H, joint_pose=torch.tensor(q)).cuda()
        with torch.no_grad():
            r_w = render_gradio(cam, gaussians, white)['render'].clamp(0, 1)
            r_b = render_gradio(cam, gaussians, black)['render'].clamp(0, 1)
        alpha = (1 - (r_w - r_b).abs().max(0)[0]).cpu().numpy()
        yield (r_w * 255).permute(1, 2, 0).byte().cpu().numpy(), alpha


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['bridge', 'robomind'])
    ap.add_argument('--scene', required=True, help='bridge: trajN dir; robomind: <traj>/<camera> dir')
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--extrinsics', default=None, help='world-to-camera .npy (default <scene>/extrinsics.npy)')
    ap.add_argument('--T_base_cam', type=float, nargs=16, default=None, help='camera-to-base 4x4, row-major')
    ap.add_argument('--grip_mode', default='binary', choices=['legacy', 'binary'], help='[robomind]')
    ap.add_argument('--out', default=None, help='default <scene>/overlay')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--fps', type=int, default=5)
    args = ap.parse_args()

    from PIL import Image
    from autocalib.model import load_robot_model
    images, joints, K, ext_default = load(args.dataset, args.scene, args.grip_mode)
    if args.T_base_cam is not None:
        w2c = np.linalg.inv(np.asarray(args.T_base_cam, np.float64).reshape(4, 4))
    else:
        w2c = np.load(args.extrinsics or ext_default)
    out = args.out or os.path.join(args.scene, 'overlay')
    os.makedirs(out, exist_ok=True)
    gaussians, bg = load_robot_model(args.model_path)
    T, H, W, _ = images.shape
    idx = list(range(0, T, args.stride))
    panels = []
    for i, (rgb, alpha) in zip(idx, render_frames(gaussians, bg, w2c, K, joints[idx], W, H)):
        a = (alpha > 0.15)[..., None] * 0.6
        tinted = (images[i] * (1 - a) + np.array([255, 40, 40]) * a).astype(np.uint8)
        panel = np.concatenate([images[i], tinted, rgb], axis=1)
        Image.fromarray(panel).save(os.path.join(out, f'overlay_{i:04d}.png'))
        panels.append(panel)
    try:
        import imageio
        imageio.mimwrite(os.path.join(out, 'overlay.mp4'), panels, fps=args.fps, codec='libx264')
    except Exception as e:  # noqa: BLE001
        print(f'video not written ({e}); frames are in {out}')
    print(f'{len(panels)} frames -> {out}')


if __name__ == '__main__':
    main()
