"""Interactive start pose: align the rendered robot onto the real frames by hand (Gradio, GPU).

The optimisation needs a start pose in the right basin. For a new camera mount, orbit the camera around the
workspace with the sliders until the render roughly covers the arm, then "Save init_w2c" writes ``init_w2c.npy``
into the scene directory; pass it as ``--init_w2c``. X / Y / Z translate along the camera's own axes.

    python tools/align_gui.py --dataset robomind --model_path <model> --glob '<root>/*/*/*/camera_top'
    python tools/align_gui.py --dataset bridge --model_path <model> --scenes <trajN> ...

One alignment serves every trajectory recorded with the same camera pose.
"""
import argparse
import glob as globmod
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np  # noqa: E402

STATE = {}


def lookat_w2c(C, target, up=np.array([0.0, 0.0, 1.0])):
    """World-to-camera for a camera at C looking at target; rows of R are [right, down, forward]."""
    C, target = np.asarray(C, float), np.asarray(target, float)
    f = target - C
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return T


def orbit_pose(az, el, dist, target, tx=0.0, ty=0.0, tz=0.0):
    """Orbit pose around ``target``, then translated along the camera's own axes."""
    a, e = np.radians(az), np.radians(el)
    target = np.asarray(target, float)
    C = target + np.array([dist * np.cos(e) * np.cos(a), dist * np.cos(e) * np.sin(a), dist * np.sin(e)])
    R = lookat_w2c(C, target)[:3, :3]
    C2 = C + tx * R[0] + ty * R[1] + tz * R[2]
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = -R @ C2
    return out, C2


def params_from_w2c(w2c, target):
    """Inverse of ``orbit_pose``: recover (az, el, dist, tx, ty, tz) from a world-to-camera matrix.

    ``R`` depends only on the view direction, so ``u = -R[2]`` gives azimuth/elevation; the orbit
    distance is taken as |C - target| and the residual is absorbed by the camera-axis offsets.
    Exact for any pose this tool produced (round-trips to ~1e-16).
    """
    w2c = np.asarray(w2c, float)
    R, t = w2c[:3, :3], w2c[:3, 3]
    target = np.asarray(target, float)
    C2 = -R.T @ t
    u = -R[2]
    el = np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0)))
    az = np.degrees(np.arctan2(u[1], u[0]))
    dist = float(np.linalg.norm(C2 - target))
    d = C2 - (target + dist * u)
    return az, el, dist, float(d @ R[0]), float(d @ R[1]), float(d @ R[2])


def saved_params(key, target, default=(20.0, 20.0, 1.6, 0.0, 0.0, 0.0)):
    """Slider values for a scene's existing ``init_w2c.npy``, or ``default`` when there is none."""
    p = os.path.join(STATE['scenes'][key], 'init_w2c.npy')
    if not os.path.exists(p):
        return default
    try:
        w2c = np.load(p)
        az, el, dist, tx, ty, tz = params_from_w2c(w2c, target)
        # snap to the slider grids and re-solve the offsets for the snapped distance
        az, el = round(az), round(el)
        dist = round(dist / 0.02) * 0.02
        R = np.asarray(w2c, float)[:3, :3]
        C2 = -R.T @ np.asarray(w2c, float)[:3, 3]
        d = C2 - (np.asarray(target, float) + dist * (-R[2]))
        tx, ty, tz = (round(float(d @ R[k]), 2) for k in range(3))
        return az, el, dist, tx, ty, tz
    except Exception as e:  # noqa: BLE001
        print(f'[align] could not read {p}: {e}')
        return default


def load_scene(key):
    if key in STATE['cache']:
        return STATE['cache'][key]
    from tools.overlay import load
    images, joints, K, _ = load(STATE['dataset'], STATE['scenes'][key])
    STATE['cache'][key] = dict(images=images, joints=joints, K=K,
                               n=len(images), H=images.shape[1], W=images.shape[2])
    return STATE['cache'][key]


def render(key, frame, az, el, dist, tx, ty, tz, opacity, show_real):
    import torch
    from drrobot.cameras import Camera_Pose
    from drrobot.renderer import render_gradio
    d = load_scene(key)
    i = int(np.clip(frame, 0, d['n'] - 1))
    real, H, W, K = d['images'][i], d['H'], d['W'], d['K']
    if show_real:
        return real, STATE.get('last_text', '')
    w2c, C = orbit_pose(az, el, dist, STATE['target'], tx, ty, tz)
    fovx, fovy = 2 * np.arctan(W / (2 * K[0, 0])), 2 * np.arctan(H / (2 * K[1, 1]))
    cam = Camera_Pose(torch.tensor(w2c).float().cuda(), fovx, fovy, W, H,
                      joint_pose=torch.tensor(d['joints'][i]), zero_init=True).cuda()
    with torch.no_grad():
        r = render_gradio(cam, STATE['gaussians'], STATE['bg'])['render'].clamp(0, 1)
    rend = (r * 255).permute(1, 2, 0).byte().cpu().numpy()
    m = (rend < 245).any(axis=2, keepdims=True).astype(np.float32) * float(opacity)
    blend = (real * (1 - m) + rend * m).astype(np.uint8)
    STATE['last_w2c'] = w2c
    txt = ('world-to-camera (4x4)\n[' + ',\n '.join(
        '[' + ', '.join(f'{v: .6f}' for v in row) + ']' for row in w2c) + ']'
        + f'\n\ncamera centre  [{C[0]: .3f}, {C[1]: .3f}, {C[2]: .3f}]'
        + f'\nparameters     az={az:.0f} el={el:.0f} dist={dist:.2f} '
          f'tx={tx:.3f} ty={ty:.3f} tz={tz:.3f}')
    STATE['last_text'] = txt
    return blend, txt


def save_init(key):
    if STATE.get('last_w2c') is None:
        return 'nothing to save yet - move a slider first'
    p = os.path.join(STATE['scenes'][key], 'init_w2c.npy')
    np.save(p, STATE['last_w2c'])
    return f'saved {p}\n\nuse it with:\n  --init_w2c {p}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['bridge', 'robomind'])
    ap.add_argument('--scenes', nargs='*', default=None, help='scene dirs (robomind: <traj>/<camera>)')
    ap.add_argument('--glob', default=None, help="glob for scene dirs, e.g. '<root>/*/*/camera_top'")
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--target', type=float, nargs=3, default=[-0.05, 0.35, 0.12],
                    help='world point the orbit looks at (roughly the robot workspace centre)')
    ap.add_argument('--share', action='store_true',
                    help='create a public gradio.live link')
    ap.add_argument('--port', type=int, default=7860)
    args = ap.parse_args()

    scenes = list(args.scenes or [])
    if args.glob:
        scenes += sorted(globmod.glob(args.glob))
    if not scenes:
        ap.error('give --scenes or --glob')

    from autocalib.model import load_robot_model
    base = os.path.commonpath(scenes) if len(scenes) > 1 else os.path.dirname(scenes[0])
    STATE['dataset'] = args.dataset
    STATE['scenes'] = {os.path.relpath(s, base): s for s in scenes}
    STATE['cache'] = {}
    STATE['target'] = args.target
    STATE['last_w2c'] = None
    gaussians, bg = load_robot_model(args.model_path)
    STATE['gaussians'], STATE['bg'] = gaussians, bg
    print(f'[align] model loaded; {len(scenes)} scenes')

    import gradio as gr
    keys = list(STATE['scenes'])
    with gr.Blocks(title='Initial-pose aligner', theme=gr.themes.Soft()) as demo:
        gr.Markdown('## Initial-pose aligner\n'
                    'Align the rendered robot onto the real frame, then **Save init_w2c** and pass the file '
                    'to `calibrate.py --init_w2c`. X / Y / Z translate along the camera axes, so they behave '
                    'the same at any azimuth.')
        with gr.Row():
            with gr.Column(scale=3):
                view = gr.Image(label='overlay', height=520, type='numpy')
            with gr.Column(scale=1):
                scene = gr.Dropdown(keys, value=keys[0], label='scene')
                frame = gr.Slider(0, 200, value=30, step=1, label='frame')
                gr.Markdown('**Orientation**')
                p0 = saved_params(keys[0], STATE['target'])
                az = gr.Slider(-180, 180, value=p0[0], step=1, label='azimuth (deg)')
                el = gr.Slider(-20, 89, value=p0[1], step=1, label='elevation (deg)')
                dist = gr.Slider(0.3, 4.0, value=p0[2], step=0.02, label='orbit distance (m)')
                gr.Markdown('**Translation** (camera axes)')
                tx = gr.Slider(-1.5, 1.5, value=p0[3], step=0.01, label='X - right (m)')
                ty = gr.Slider(-1.5, 1.5, value=p0[4], step=0.01, label='Y - down (m)')
                tz = gr.Slider(-1.5, 1.5, value=p0[5], step=0.01, label='Z - forward (m)')
                opacity = gr.Slider(0.2, 1.0, value=0.7, step=0.05, label='overlay opacity')
                show_real = gr.Checkbox(False, label='show real frame only')
                b_save = gr.Button('Save init_w2c', variant='primary')
        out = gr.Textbox(label='pose', lines=9)

        ctrls = [scene, frame, az, el, dist, tx, ty, tz, opacity, show_real]
        for c in ctrls:
            c.change(render, ctrls, [view, out], show_progress='hidden')

        def on_scene(k):
            d = load_scene(k)
            p = saved_params(k, STATE['target'])
            return (gr.update(maximum=d['n'] - 1, value=min(30, d['n'] // 2)),
                    gr.update(value=p[0]), gr.update(value=p[1]), gr.update(value=p[2]),
                    gr.update(value=p[3]), gr.update(value=p[4]), gr.update(value=p[5]))
        scene.change(on_scene, scene, [frame, az, el, dist, tx, ty, tz])
        b_save.click(save_init, scene, out)
        demo.load(render, ctrls, [view, out])

    demo.launch(share=args.share, server_name='127.0.0.1', server_port=args.port, show_error=True)


if __name__ == '__main__':
    main()
