"""Fit the fixed top camera of an ABC-130K episode to the robot masks (soft IoU) or photometrically.

Unknown: the camera pose (6 DoF) in the station frame (see station.py). K is the episode's calibration (the frames
are undistorted with it) and the joints are the recorded ones.

Mask loss: soft IoU of the rendered two-arm alpha and the robot masks (masks.py; uint8 png: >= 200 robot,
50..199 ignored, < 50 background), both Gaussian-blurred, coarse to fine over image scales.
Photometric loss: 1 - local NCC between the render composited on grey and the real grayscale (+ 0.5 x the same on
gradient magnitudes), inside the dilated render silhouette at the stage's start pose (kept fixed per stage).
Start pose: a camera-frame grid around the simulator top camera plus look-at cameras aimed at the bases and end
effectors, ranked by hard mask IoU on 4 frames at 1/4 resolution; the best mutually distinct seeds get a burn-in and
the best one continues through the stages. Frames are split into fit / held-out frames (``--holdout even``).

Per-video recipe (what run_videos.py runs):

    python -m yam.fit --model M --episodes EP --out EP/fit   --holdout even --max_frames 12 --k_mode auto
    python -m yam.fit --model M --episodes EP --out EP/photo --holdout even --max_frames 12 --k_mode auto \\
        --loss photo --init_pose EP/fit/c2w.npy --no_grid --n_seeds 1 --burnin_steps 20 \\
        --stages 0.5:2:300:1e-3,1.0:1:400:4e-4

Outputs in --out: c2w.npy / w2c.npy (OpenCV, station frame), K.npy, status.json, *_sheet.jpg, log.txt.
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import station

DEV = "cuda"
# Gaussians below this canonical opacity are dropped (75 % of the model; silhouette and pose gradient unchanged).
# The keep set comes from the canonical opacities so that every frame keeps the same Gaussians (batched frames).
MIN_OPACITY = [0.05]
_KEEP = {}
LOSS = {"mode": "mask", "w_mask": 1.0, "w_photo": 1.0, "ncc_win": 9, "dilate": 3, "w_grad": 0.5}


def _rasterize(**kw):
    from gsplat import rasterization
    return rasterization(**kw)


def keep_index(pc):
    k = (id(pc), MIN_OPACITY[0])
    if k not in _KEEP:
        _KEEP[k] = torch.nonzero(pc.get_opacity.detach().reshape(-1) > MIN_OPACITY[0])[:, 0]
    return _KEEP[k]


# ----------------------------------------------------------------------------------------------- data
def load_episode(ep_dir, masks_sub, k_mode, skip_first=True, max_frames=None):
    st = json.load(open(os.path.join(ep_dir, "states.json")))
    calib = st["calib"]
    K = np.array(calib["K"], np.float64).reshape(3, 3)
    img_sub = "undist"
    if k_mode == "auto":
        hyp = st.get("cx_shift_hypothesis")
        k_mode = "cxfix" if (hyp and hyp.get("K_hypothesis") and os.path.isdir(os.path.join(ep_dir, "undist_cxfix"))) else "calib"
    if k_mode == "cxfix":
        hyp = st.get("cx_shift_hypothesis")
        assert hyp and hyp.get("K_hypothesis"), f"{ep_dir}: no cx_shift_hypothesis in states.json"
        K = np.array(hyp["K_hypothesis"], np.float64).reshape(3, 3)
        img_sub = "undist_cxfix"
    elif k_mode == "raw":
        img_sub = "."
    recs = st["records"][1:] if skip_first else st["records"]   # the first frame's state is often stale
    frames = []
    for r in recs:
        name = r["png"]
        img_p = os.path.join(ep_dir, img_sub, name)
        msk_p = os.path.join(ep_dir, masks_sub, name) if masks_sub not in ("none", "") else None
        if not os.path.exists(img_p) or (msk_p is not None and not os.path.exists(msk_p)):
            continue
        gl, gr = r["left_grip"], r["right_grip"]
        gl = gl[0] if isinstance(gl, list) else gl
        gr = gr[0] if isinstance(gr, list) else gr
        frames.append(dict(name=name, img=img_p, mask=msk_p, t_s=r["t_s"],
                           ql=station.abc_state_to_q(r["left_q"], gl), qr=station.abc_state_to_q(r["right_q"], gr)))
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx.tolist()))]
    return dict(dir=ep_dir, task=st.get("task", os.path.basename(ep_dir.rstrip("/"))), K=K,
                w=int(calib["w"]), h=int(calib["h"]), frames=frames, k_mode=k_mode)


def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert m is not None, path
    return (m >= 200).astype(np.float32), 1.0 - ((m >= 50) & (m < 200)).astype(np.float32)


def read_img(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    assert im is not None, path
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------------------------- render
class Frame:
    """A real frame and the deformed, placed Gaussians of both arms at its joints."""

    def __init__(self, pc, fr, ep, base_sep, device=DEV):
        """device='cpu' keeps the Gaussians in host memory (held-out frames are evaluated only a few times)."""
        self.name, self.t_s, self.ep = fr["name"], fr["t_s"], ep["task"]
        self.img = read_img(fr["img"])
        self.has_mask = bool(fr.get("mask")) and os.path.exists(fr["mask"])
        if self.has_mask:
            self.mask, self.weight = read_mask(fr["mask"])
        else:
            self.mask = np.zeros(self.img.shape[:2], np.float32)
            self.weight = np.ones(self.img.shape[:2], np.float32)
        assert self.img.shape[:2] == (ep["h"], ep["w"]) == self.mask.shape, (fr["img"], self.img.shape, self.mask.shape)
        self._gray, self._cache = {}, {}
        self.K, self.w, self.h = ep["K"], ep["w"], ep["h"]
        with torch.no_grad():
            self.arms = {}
            keep = keep_index(pc)
            for name, q, T in (("left", fr["ql"], station.T_ABC_LEFT), ("right", fr["qr"], station.right_base(base_sep))):
                m, r, s, o, c = station.deform_arm(pc, q)
                m, r = station.place(m, r, T)
                d = dict(means=m.detach()[keep].contiguous(), quats=r.detach()[keep].contiguous(), scales=s.detach()[keep].contiguous(),
                         opacities=o.detach().squeeze(-1)[keep].contiguous(), colors=c.detach()[keep].contiguous())
                if device == "cpu":
                    d = {k: v.cpu().pin_memory() for k, v in d.items()}
                self.arms[name] = d
            self.cat = {k: torch.cat([self.arms["left"][k], self.arms["right"][k]]) for k in self.arms["left"]}
        self.sh_degree = int(pc.active_sh_degree)

    def gray(self, scale):
        """Real frame as grayscale (H, W) in [0, 1] at the given scale (area downsampling)."""
        if scale not in self._gray:
            g = torch.as_tensor(cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0, device=DEV)[None, None]
            if scale != 1.0:
                g = F.interpolate(g, size=(int(round(self.h * scale)), int(round(self.w * scale))), mode="area")
            self._gray[scale] = g[0, 0]
        return self._gray[scale]

    def target(self, scale):
        """(mask, weight) at the given scale (area downsampling)."""
        if scale not in self._cache:
            m = torch.as_tensor(self.mask, device=DEV)[None, None]
            wgt = torch.as_tensor(self.weight, device=DEV)[None, None]
            if scale != 1.0:
                size = (int(round(self.h * scale)), int(round(self.w * scale)))
                m = F.interpolate(m, size=size, mode="area")
                wgt = F.interpolate(wgt, size=size, mode="area")
            self._cache[scale] = (m[0, 0], wgt[0, 0])
        return self._cache[scale]

    def gather(self, arms=("left", "right")):
        parts = [self.cat] if tuple(arms) == ("left", "right") else [self.arms[a] for a in arms]
        out = []
        for k in ("means", "quats", "scales", "opacities", "colors"):
            v = torch.cat([p[k] for p in parts]) if len(parts) > 1 else parts[0][k]
            out.append(v.to(DEV, non_blocking=True) if v.device.type != "cuda" else v)
        return out


def K_scaled(K, scale):
    return torch.as_tensor(np.diag([scale, scale, 1.0]) @ K, device=DEV, dtype=torch.float32)


_ONES = {}


def _ones(n, b=None):
    k = (n, b)
    if k not in _ONES:
        _ONES[k] = torch.ones((n, 1) if b is None else (b, n, 1), device=DEV)
    return _ONES[k]


def render_alpha(frame, viewmats, scale, arms=("left", "right")):
    """viewmats (C, 4, 4) -> alpha (C, h, w), differentiable w.r.t. viewmats."""
    hs, ws = int(round(frame.h * scale)), int(round(frame.w * scale))
    means, quats, scales, opac, _ = frame.gather(arms)
    C = viewmats.shape[0]
    _, alphas, _ = _rasterize(means=means, quats=quats, scales=scales, opacities=opac, colors=_ones(means.shape[0]),
                              viewmats=viewmats, Ks=K_scaled(frame.K, scale)[None].expand(C, 3, 3), width=ws, height=hs, packed=False)
    return alphas[..., 0]


def render_rgba(frame, viewmat, scale, bg=0.0):
    """rgb (h, w, 3) + alpha (h, w) for one camera, differentiable."""
    hs, ws = int(round(frame.h * scale)), int(round(frame.w * scale))
    means, quats, scales, opac, colors = frame.gather()
    rgb, alphas, _ = _rasterize(means=means, quats=quats, scales=scales, opacities=opac, colors=colors, viewmats=viewmat[None],
                                Ks=K_scaled(frame.K, scale)[None], width=ws, height=hs, packed=False, sh_degree=frame.sh_degree,
                                backgrounds=torch.full((1, 3), bg, device=DEV))
    return rgb[0], alphas[0, ..., 0]


class Batch:
    """All fit frames stacked ([B, N, .], same keep set): one rasterisation per step."""

    def __init__(self, frames, need_rgb):
        self.B, self.h, self.w = len(frames), frames[0].h, frames[0].w
        assert all(f.cat["means"].shape[0] == frames[0].cat["means"].shape[0] for f in frames)
        assert all((f.h, f.w) == (self.h, self.w) for f in frames), "batched frames need one image size"
        self.Ks = {}
        self._K_list = [f.K for f in frames]
        keys = ("means", "quats", "scales", "opacities") + (("colors",) if need_rgb else ())
        self.g = {k: torch.stack([f.cat[k].to(DEV) for f in frames]) for k in keys}
        self.sh_degree = frames[0].sh_degree

    def K(self, scale):
        if scale not in self.Ks:
            self.Ks[scale] = torch.stack([K_scaled(k, scale) for k in self._K_list])[:, None]
        return self.Ks[scale]

    def render(self, vm, scale, rgb=False):
        """vm (4, 4) -> alpha (B, h, w) [, rgb (B, h, w, 3) on black]."""
        hs, ws = int(round(self.h * scale)), int(round(self.w * scale))
        g, B = self.g, self.B
        vms = vm[None, None].expand(B, 1, 4, 4)
        if rgb:
            out, al, _ = _rasterize(means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"], colors=g["colors"],
                                    viewmats=vms, Ks=self.K(scale), width=ws, height=hs, packed=False, sh_degree=self.sh_degree,
                                    backgrounds=torch.zeros(B, 1, 3, device=DEV))
            return al[:, 0, ..., 0], out[:, 0]
        _, al, _ = _rasterize(means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"],
                              colors=_ones(g["means"].shape[1], B), viewmats=vms, Ks=self.K(scale), width=ws, height=hs, packed=False)
        return al[:, 0, ..., 0]


# ----------------------------------------------------------------------------------------------- losses
def gauss_blur(x, sigma):
    """Separable Gaussian blur of (B, 1, H, W) or (H, W) with replicate padding."""
    if sigma <= 0:
        return x
    sq = x.dim() == 2
    if sq:
        x = x[None, None]
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-(torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return x[0, 0] if sq else x


def local_ncc(a, b, win=9, eps=1e-4):
    """Per-pixel normalised cross-correlation of two (H, W) images over win x win windows."""
    A, B = a[None, None], b[None, None]
    mu_a, mu_b = F.avg_pool2d(A, win, 1, win // 2), F.avg_pool2d(B, win, 1, win // 2)
    va = F.avg_pool2d(A * A, win, 1, win // 2) - mu_a ** 2
    vb = F.avg_pool2d(B * B, win, 1, win // 2) - mu_b ** 2
    cov = F.avg_pool2d(A * B, win, 1, win // 2) - mu_a * mu_b
    return (cov / torch.sqrt(va.clamp(min=0) + eps) / torch.sqrt(vb.clamp(min=0) + eps))[0, 0]


def grad_mag(g):
    gx = F.pad(g[:, 2:] - g[:, :-2], (1, 1, 0, 0))
    gy = F.pad(g[2:, :] - g[:-2, :], (0, 0, 1, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def photo_region(alpha, dilate=3):
    """Comparison region of a stage: the dilated render silhouette at the stage's start pose. It is kept fixed so
    the loss cannot be lowered by moving the render away from the compared area."""
    return (F.max_pool2d(alpha.detach()[None, None], 2 * dilate + 1, 1, dilate)[0, 0] > 0.3).float()


def photo_loss(rgb, alpha, gray_real, region, win=9, w_grad=0.5):
    """Returns (loss, mean NCC inside alpha > 0.5). An absent render (flat grey) scores ~1 everywhere."""
    g_r = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    g_r = g_r + (1 - alpha) * 0.5
    ncc = local_ncc(g_r, gray_real, win)
    loss = ((1 - ncc) * region).sum() / region.sum().clamp(min=1)
    if w_grad > 0:
        ncc_g = local_ncc(grad_mag(g_r), grad_mag(gray_real), win)
        loss = loss + w_grad * ((1 - ncc_g) * region).sum() / region.sum().clamp(min=1)
    inside = alpha.detach() > 0.5
    return loss, (float(ncc.detach()[inside].mean()) if int(inside.sum()) > 50 else float("nan"))


def soft_iou(alpha, target, weight):
    inter = (alpha * target * weight).sum(dim=(-2, -1))
    union = ((alpha + target - alpha * target) * weight).sum(dim=(-2, -1))
    return inter / (union + 1e-6)


def hard_iou(alpha, target, weight, thr=0.5):
    a, t = (alpha > thr).float(), (target > 0.5).float()
    inter = (a * t * weight).sum(dim=(-2, -1))
    union = ((a + t - a * t) * weight).sum(dim=(-2, -1))
    return inter / (union + 1e-6)


# ----------------------------------------------------------------------------------------------- poses
def rot_xyz(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def camera_frame_grid(w2c0, rot_x, rot_y, rot_z, trans, n_rx, n_ry, n_rz, n_t):
    """Candidates D @ w2c0 (rotation about the camera centre, translation along the camera axes)."""
    cands, meta = [], []
    for rx in np.linspace(-rot_x, rot_x, n_rx):
        for ry in np.linspace(-rot_y, rot_y, n_ry):
            for rz in np.linspace(-rot_z, rot_z, n_rz):
                for tx in np.linspace(-trans, trans, n_t):
                    for ty in np.linspace(-trans, trans, n_t):
                        for tz in np.linspace(-trans, trans, n_t):
                            D = np.eye(4)
                            D[:3, :3] = rot_xyz(rx, ry, rz)
                            D[:3, 3] = [tx, ty, tz]
                            cands.append(D @ w2c0)
                            meta.append((rx, ry, rz, tx, ty, tz))
    return np.stack(cands), meta


def fk_point_world(pc, q8, T_base, frame_key="link_6"):
    """World position of a chain frame (default link_6) at raw joints q8."""
    names = pc.chain.get_frame_names()
    cand = [n for n in names if frame_key in n]
    name = cand[0] if cand else names[-1]
    tf = pc.chain.forward_kinematics(torch.as_tensor(np.asarray(q8, np.float32), device=DEV)[None])
    return (T_base @ tf[name].get_matrix()[0].detach().cpu().numpy().astype(np.float64))[:3, 3]


def lookat_w2c(cam_pos, target, up_hint=(0.0, 0.0, 1.0)):
    """OpenCV camera at cam_pos looking at target; image up follows up_hint."""
    f = np.asarray(target, np.float64) - np.asarray(cam_pos, np.float64)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up_hint, np.float64))
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([r, d, f], 1)
    c2w[:3, 3] = cam_pos
    return np.linalg.inv(c2w)


def lookat_seeds(target, dists, elevs_deg, azims_deg, rolls_deg=(0.0,)):
    """Cameras on spherical caps above ``target``, behind the arm bases (azimuth 0 = from -x)."""
    out, meta = [], []
    for dd in dists:
        for el in elevs_deg:
            for az in azims_deg:
                e, z = math.radians(el), math.radians(az)
                direction = np.array([-math.cos(e) * math.cos(z), -math.cos(e) * math.sin(z), math.sin(e)])
                w2c = lookat_w2c(np.asarray(target) + dd * direction, target)
                for roll in rolls_deg:
                    Rr = np.eye(4)
                    Rr[:3, :3] = rot_xyz(0, 0, math.radians(roll))
                    out.append(Rr @ w2c)
                    meta.append((dd, el, az, roll))
    return np.stack(out), meta


def pose_delta(w2c_a, w2c_b):
    """Rotation (deg) and camera-centre distance (cm) between two world-to-camera poses."""
    Ra, Rb = w2c_a[:3, :3], w2c_b[:3, :3]
    ang = math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
    ca, cb = -Ra.T @ w2c_a[:3, 3], -Rb.T @ w2c_b[:3, 3]
    return ang, float(np.linalg.norm(ca - cb) * 100)


# ----------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(frames, w2c, per_arm=True):
    """Full-resolution hard IoU per frame, photometric NCC, and per-arm precision (rendered pixels inside the mask)."""
    vm = torch.as_tensor(w2c, device=DEV, dtype=torch.float32)[None]
    out = []
    for f in frames:
        m, wgt = f.target(1.0)
        a = render_alpha(f, vm, 1.0)[0]
        rgb_f, a_f = render_rgba(f, vm[0], 1.0)
        _, ncc = photo_loss(rgb_f, a_f, f.gray(1.0), photo_region(a_f, LOSS["dilate"]), LOSS["ncc_win"], 0.0)
        rec = dict(ep=f.ep, name=f.name, iou=float(hard_iou(a, m, wgt)), soft_iou=float(soft_iou(a, m, wgt)), ncc=ncc,
                   robot_px=int((a > 0.5).sum()), mask_px=int((m > 0.5).sum()), has_mask=bool(f.has_mask))
        if per_arm:
            for arm in ("left", "right"):
                aa = (render_alpha(f, vm, 1.0, arms=(arm,))[0] > 0.5).float()
                rec[f"{arm}_prec"] = float((aa * (m > 0.5).float() * wgt).sum() / (aa * wgt).sum().clamp(min=1))
                rec[f"{arm}_px"] = int(aa.sum())
        out.append(rec)
    return out


def contour_overlay(img, alpha, mask, weight):
    """Real frame with the render contour (green), mask contour (red) and ignore band (blue tint)."""
    vis = img.copy()
    ign = weight < 0.5
    vis[ign] = (0.6 * vis[ign] + 0.4 * np.array([80, 80, 255])).astype(np.uint8)
    for m, col in (((mask > 0.5).astype(np.uint8), (255, 40, 40)), ((alpha > 0.5).astype(np.uint8), (40, 255, 40))):
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, col, 1)
    return vis


@torch.no_grad()
def sheet(frames, w2c, path, max_rows=8):
    """real | render | blend | contours for up to max_rows frames."""
    vm = torch.as_tensor(w2c, device=DEV, dtype=torch.float32)
    rows = []
    for i in sorted(set(np.linspace(0, len(frames) - 1, min(max_rows, len(frames))).round().astype(int).tolist())):
        f = frames[i]
        rgb, alpha = render_rgba(f, vm, 1.0, bg=1.0)
        rgb, alpha = rgb.clamp(0, 1).cpu().numpy(), alpha.cpu().numpy()
        row = np.concatenate([f.img, (rgb * 255).astype(np.uint8), (station.overlay(f.img, rgb, alpha, 0.65) * 255).astype(np.uint8),
                              contour_overlay(f.img, alpha, f.mask, f.weight)], 1)
        cv2.putText(row, f"{f.ep[:28]} {f.name}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        rows.append(row)
    im = np.concatenate(rows, 0)
    if im.shape[0] > 3000:
        s = 3000 / im.shape[0]
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ----------------------------------------------------------------------------------------------- optimisation
def optimise(frames, w2c0, scale, sigma, steps, lr, log, tag=""):
    """Adam on a camera-frame se(3) delta, cosine lr to 5 %. Returns (best w2c, best score)."""
    vm0 = torch.as_tensor(w2c0, device=DEV, dtype=torch.float32)
    xi = torch.zeros(6, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([{"params": [xi], "lr": lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    tb = [(gauss_blur(m, sigma), w) for m, w in (f.target(scale) for f in frames)]
    regions = None
    if LOSS["mode"] != "mask":
        with torch.no_grad():
            regions = [photo_region(render_alpha(f, vm0[None], scale)[0], LOSS["dilate"]) for f in frames]
    best = (-1.0, None)
    batch = Batch(frames, need_rgb=LOSS["mode"] != "mask")
    MB, WB = torch.stack([mb for mb, _ in tb]), torch.stack([w for _, w in tb])
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        vm = station.se3_exp(xi) @ vm0
        if LOSS["mode"] == "mask":
            al = batch.render(vm, scale)
            ious = soft_iou(gauss_blur(al[:, None], sigma)[:, 0], MB, WB)
            loss = (1 - ious).mean()
        else:
            al, rgb = batch.render(vm, scale, rgb=True)
            terms, ious = [], []
            for fi, f in enumerate(frames):
                lp, _ = photo_loss(rgb[fi], al[fi], f.gray(scale), regions[fi], LOSS["ncc_win"], LOSS["w_grad"])
                term = LOSS["w_photo"] * lp
                if LOSS["mode"] == "both" and f.has_mask:
                    iou = soft_iou(gauss_blur(al[fi], sigma), MB[fi], WB[fi])
                    term = term + LOSS["w_mask"] * (1 - iou)
                else:
                    iou = 1 - lp      # reported in the score slot
                terms.append(term)
                ious.append(iou)
            loss = torch.stack(terms).mean()
            ious = torch.stack(ious)
        loss.backward()
        ious = ious.detach()
        if not torch.isfinite(xi.grad).all():
            log(f"  [{tag}] non-finite gradient at step {step}; stopping the stage")
            break
        opt.step()
        sched.step()
        cur = float(ious.mean())
        if cur > best[0]:
            best = (cur, (station.se3_exp(xi.detach()) @ vm0).cpu().numpy().astype(np.float64))
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log(f"  [{tag}] step {step:4d} score {cur:.4f} (min {float(ious.min()):.4f}) |xi| {float(xi.detach().norm()):.4f}")
    del batch
    torch.cuda.empty_cache()
    return best[1], best[0]


def optimise_seeds(frames, w2c_list, scale, sigma, steps, lr, log, tag="burnin"):
    """Mask-mode Adam on S seeds at once (one rasterisation renders every frame from every seed). The loss is the
    sum over seeds, so every seed follows exactly its own path. Returns [(best w2c, best soft IoU)]."""
    S = len(w2c_list)
    b = Batch(frames, need_rgb=False)
    vm0 = torch.as_tensor(np.stack(w2c_list), device=DEV, dtype=torch.float32)
    xi = torch.zeros(S, 6, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    tb = [f.target(scale) for f in frames]
    MB = torch.stack([gauss_blur(m, sigma) for m, _ in tb])[:, None]
    WB = torch.stack([w for _, w in tb])[:, None]
    hs, ws = int(round(b.h * scale)), int(round(b.w * scale))
    g, Bn = b.g, b.B
    best = [(-1.0, None)] * S
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        vms = torch.stack([station.se3_exp(xi[i]) @ vm0[i] for i in range(S)])
        _, al, _ = _rasterize(means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"],
                              colors=_ones(g["means"].shape[1], Bn), viewmats=vms[None].expand(Bn, S, 4, 4),
                              Ks=b.K(scale).expand(Bn, S, 3, 3), width=ws, height=hs, packed=False)
        bl = gauss_blur(al[..., 0].reshape(Bn * S, 1, hs, ws), sigma).reshape(Bn, S, hs, ws)
        ious = soft_iou(bl, MB, WB)                                   # (B, S)
        (1 - ious).mean(0).sum().backward()
        if not torch.isfinite(xi.grad).all():
            log(f"  [{tag}] non-finite gradient at step {step}; stopping")
            break
        opt.step()
        sched.step()
        cur = ious.detach().mean(0).cpu().numpy()
        with torch.no_grad():
            poses = torch.stack([station.se3_exp(xi[i]) @ vm0[i] for i in range(S)]).cpu().numpy()
        for i in range(S):
            if cur[i] > best[i][0]:
                best[i] = (float(cur[i]), poses[i].astype(np.float64))
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log(f"  [{tag}] step {step:4d} soft-IoU per seed {np.round(cur, 4).tolist()}")
    del b
    torch.cuda.empty_cache()
    return [(best[i][1], best[i][0]) for i in range(S)]


@torch.no_grad()
def score_candidates(frames, cands, scale=0.25, chunk=48, keep_frac=1.0):
    """Mean hard IoU per candidate w2c (C, 4, 4), all frames x ``chunk`` candidates per rasterisation.
    keep_frac < 1 ranks with a fixed random subset of the Gaussians (ranking only; the fit uses all)."""
    C = cands.shape[0]
    tot = np.zeros(C)
    b = Batch(frames, need_rgb=False)
    if keep_frac < 1.0:
        N = b.g["means"].shape[1]
        gen = torch.Generator(device=DEV).manual_seed(0)
        idx = torch.randperm(N, generator=gen, device=DEV)[: int(N * keep_frac)]
        b.g = {k: v[:, idx].contiguous() for k, v in b.g.items()}
    vms = torch.as_tensor(cands, device=DEV, dtype=torch.float32)
    M = torch.stack([f.target(scale)[0] for f in frames])[:, None]
    Wt = torch.stack([f.target(scale)[1] for f in frames])[:, None]
    hs, ws = int(round(b.h * scale)), int(round(b.w * scale))
    g, Fn = b.g, b.B
    for i in range(0, C, chunk):
        v = vms[i:i + chunk]
        c = v.shape[0]
        _, al, _ = _rasterize(means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"],
                              colors=_ones(g["means"].shape[1], Fn), viewmats=v[None].expand(Fn, c, 4, 4),
                              Ks=b.K(scale).expand(Fn, c, 3, 3), width=ws, height=hs, packed=False)
        tot[i:i + c] = hard_iou(al[..., 0], M, Wt).sum(0).cpu().numpy()
    return tot / len(frames)


# ----------------------------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="+", required=True, help="episode dirs (yam.extract + masks/) of ONE camera")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="YAM Dr.Robot model directory")
    ap.add_argument("--iteration", type=int, default=80000)
    ap.add_argument("--masks_sub", default="masks")
    ap.add_argument("--k_mode", default="calib", choices=["calib", "cxfix", "raw", "auto"],
                    help="auto: cxfix when the extraction wrote undist_cxfix (640 stream with an 848-profile K)")
    ap.add_argument("--min_opacity", type=float, default=0.05, help="drop Gaussians below this canonical opacity (0 = keep all)")
    ap.add_argument("--search_keep", type=float, default=0.1, help="fraction of Gaussians used to rank the start candidates")
    ap.add_argument("--base_sep", type=float, default=station.BASE_SEP)
    ap.add_argument("--max_frames", type=int, default=40, help="per episode, spread over time")
    ap.add_argument("--holdout", default="none", choices=["none", "even", "odd"], help="held-out frames (by index)")
    ap.add_argument("--init_pose", default=None, help="c2w.npy to start from instead of the simulator camera")
    ap.add_argument("--eval_pose", default=None, help="c2w.npy: evaluate only")
    ap.add_argument("--grid_rot", type=float, nargs=3, default=[0.4, 0.3, 0.15], help="+- rad about camera x, y, z")
    ap.add_argument("--grid_n", type=int, nargs=3, default=[9, 7, 3])
    ap.add_argument("--grid_trans", type=float, default=0.2)
    ap.add_argument("--grid_nt", type=int, default=3)
    ap.add_argument("--n_init_frames", type=int, default=4)
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--stages", default="0.25:3:150:3e-3,0.5:2:150:1.5e-3,1.0:1.5:150:6e-4", help="scale:blur_sigma_px:steps:lr per stage")
    ap.add_argument("--burnin_steps", type=int, default=80)
    ap.add_argument("--no_grid", action="store_true", help="start straight from the start pose (no search)")
    ap.add_argument("--init", default="both", choices=["sim", "lookat", "both"], help="start candidates")
    ap.add_argument("--lookat_dists", type=float, nargs="+", default=[0.8, 1.05, 1.3, 1.6])
    ap.add_argument("--lookat_elevs", type=float, nargs="+", default=[40, 55, 70, 85], help="deg above horizontal")
    ap.add_argument("--lookat_azims", type=float, nargs="+", default=[-30, -15, 0, 15, 30], help="deg from -x")
    ap.add_argument("--lookat_rolls", type=float, nargs="+", default=[-8, 0, 8])
    ap.add_argument("--lookat_expand", type=float, nargs=2, default=[0.12, 3], help="+- rad and n per axis around every look-at seed")
    ap.add_argument("--loss", default="mask", choices=["mask", "photo", "both"])
    ap.add_argument("--w_mask", type=float, default=1.0)
    ap.add_argument("--w_photo", type=float, default=1.0)
    ap.add_argument("--ncc_win", type=int, default=9)
    ap.add_argument("--photo_dilate", type=int, default=3)
    ap.add_argument("--w_grad", type=float, default=0.5)
    ap.add_argument("--no_sheets", action="store_true")
    return ap


def main(argv=None, pc=None):
    a = build_parser().parse_args(argv)
    MIN_OPACITY[0] = a.min_opacity
    LOSS.update(mode=a.loss, w_mask=a.w_mask, w_photo=a.w_photo, ncc_win=a.ncc_win, dilate=a.photo_dilate, w_grad=a.w_grad)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "a")
    t_start = time.time()

    def log(s):
        s = f"[{time.time() - t_start:6.1f}s] {s}"
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"fit {vars(a)}")
    if pc is None:
        pc = station.load_model(a.model, a.iteration)
    eps = [load_episode(e, a.masks_sub, a.k_mode, max_frames=a.max_frames) for e in a.episodes]
    for e in eps:
        log(f"episode {e['task']}: {len(e['frames'])} frames, {e['w']}x{e['h']}, K fx {e['K'][0, 0]:.3f} fy {e['K'][1, 1]:.3f} "
            f"cx {e['K'][0, 2]:.3f} cy {e['K'][1, 2]:.3f} ({e['k_mode']})")
        assert e["frames"], f"no frames with masks in {e['dir']}/{a.masks_sub}"
    specs = [(fr, e) for e in eps for fr in e["frames"]]
    fit_idx = [i for i in range(len(specs)) if a.holdout == "none" or (i % 2 == 0) != (a.holdout == "even")]
    hold_idx = [i for i in range(len(specs)) if i not in fit_idx]
    frames = [Frame(pc, fr, e, a.base_sep, device=DEV if i in set(fit_idx) else "cpu") for i, (fr, e) in enumerate(specs)]
    fit_frames, hold_frames = [frames[i] for i in fit_idx], [frames[i] for i in hold_idx]
    log(f"fit frames {len(fit_frames)}, held-out {len(hold_frames)}")
    K0 = eps[0]["K"]

    if a.eval_pose or a.init_pose:
        c2w0 = np.load(a.eval_pose or a.init_pose)
    else:
        _, c2w0 = station.sim_top_camera(eps[0]["w"], eps[0]["h"])
    w2c0 = np.linalg.inv(c2w0)
    ev0 = evaluate(frames, w2c0)
    log(f"start pose IoU: mean {np.mean([r['iou'] for r in ev0]):.4f}; photo NCC {np.nanmean([r['ncc'] for r in ev0]):.4f}")
    if not a.no_sheets:
        sheet(frames, w2c0, os.path.join(a.out, "start_sheet.jpg"))
    status = dict(args=vars(a), episodes=[dict(task=e["task"], dir=e["dir"], n_frames=len(e["frames"]), w=e["w"], h=e["h"],
                                               K=e["K"].tolist(), k_mode=e["k_mode"]) for e in eps],
                  base_sep=a.base_sep, start=dict(c2w=c2w0.tolist(), eval=ev0, mean_iou=float(np.mean([r["iou"] for r in ev0]))))
    if a.eval_pose:
        status["mode"] = "eval"
        json.dump(status, open(os.path.join(a.out, "status.json"), "w"), indent=1)
        return status

    # ---- start-pose search
    seeds = [w2c0]
    if not a.no_grid:
        t0 = time.time()
        init_frames = [fit_frames[i] for i in np.linspace(0, len(fit_frames) - 1, min(a.n_init_frames, len(fit_frames))).round().astype(int)]
        cand_list, meta = [], []
        if a.init in ("sim", "both"):
            cg, mg = camera_frame_grid(w2c0, *a.grid_rot, a.grid_trans, *a.grid_n, a.grid_nt)
            cand_list.append(cg)
            meta += [("grid",) + tuple(float(x) for x in m) for m in mg]
        if a.init in ("lookat", "both"):
            # aim point = centroid of the two bases and the end effectors of the init frames
            T_right = station.right_base(a.base_sep)
            pts = [station.T_ABC_LEFT[:3, 3], T_right[:3, 3]]
            for e in eps:
                for fr in e["frames"][:: max(1, len(e["frames"]) // a.n_init_frames)]:
                    pts.append(fk_point_world(pc, fr["ql"], station.T_ABC_LEFT))
                    pts.append(fk_point_world(pc, fr["qr"], T_right))
            la, ml = lookat_seeds(np.mean(np.stack(pts), 0), a.lookat_dists, a.lookat_elevs, a.lookat_azims, a.lookat_rolls)
            ex_r, ex_n = a.lookat_expand[0], int(a.lookat_expand[1])
            for w2c_l, m in zip(la, ml):
                cg, mg = camera_frame_grid(w2c_l, ex_r, ex_r, ex_r, 0.0, ex_n, ex_n, ex_n, 1)
                cand_list.append(cg)
                meta += [("lookat",) + tuple(float(x) for x in m) + tuple(float(x) for x in mm[:3]) for mm in mg]
        cands = np.concatenate(cand_list)
        sc = score_candidates(init_frames, np.asarray(cands), keep_frac=a.search_keep)
        order = np.argsort(-sc)
        log(f"search: {len(cands)} candidates scored in {time.time() - t0:.1f}s; best IoU {sc[order[0]]:.4f} ({meta[order[0]][0]})")
        seeds = []   # top seeds that are mutually > 3 deg or > 5 cm apart
        for j in order:
            if all(max(pose_delta(cands[j], s)[0] / 3.0, pose_delta(cands[j], s)[1] / 5.0) > 1.0 for s in seeds):
                seeds.append(cands[j])
            if len(seeds) >= a.n_seeds:
                break
        status["grid"] = dict(n=len(cands), top=[dict(score=float(sc[j]), kind=meta[j][0], meta=[float(x) for x in meta[j][1:]]) for j in order[:10]])

    stages = [tuple(float(x) for x in s.split(":")) for s in a.stages.split(",")]
    # ---- burn-in of every seed at the first stage, keep the best
    sc0, sig0, _, lr0 = stages[0]
    if LOSS["mode"] == "mask" and len(seeds) > 1:
        burn = [(iou_b, w2c_b) for w2c_b, iou_b in optimise_seeds(fit_frames, seeds, sc0, sig0, a.burnin_steps, lr0, log)]
    else:
        burn = []
        for si, s in enumerate(seeds):
            w2c_b, iou_b = optimise(fit_frames, s, sc0, sig0, a.burnin_steps, lr0, log, tag=f"burnin seed {si}")
            burn.append((iou_b, w2c_b))
    order_b = sorted(range(len(burn)), key=lambda i: -burn[i][0])
    best_i = order_b[0]
    w2c = burn[best_i][1]
    spread = [pose_delta(w2c, burn[i][1]) for i in order_b[1:3]]
    status["burnin"] = dict(seeds=[dict(score=float(b[0]), w2c=b[1].tolist()) for b in burn], best=int(best_i),
                            runner_up_delta_deg_cm=[[float(x) for x in d] for d in spread])
    log(f"burn-in scores {[round(b[0], 4) for b in burn]}; best seed {best_i}")

    # ---- coarse to fine
    for k, (sc, sig, steps, lr) in enumerate(stages):
        w2c, _ = optimise(fit_frames, w2c, sc, sig, int(steps), lr, log, tag=f"stage {k} scale {sc}")

    ev_fit = evaluate(fit_frames, w2c)
    ev_hold = evaluate(hold_frames, w2c) if hold_frames else []
    c2w = np.linalg.inv(w2c)
    d0 = pose_delta(w2c, w2c0)
    log(f"final fit IoU {np.mean([r['iou'] for r in ev_fit]):.4f}"
        + (f", held-out IoU {np.mean([r['iou'] for r in ev_hold]):.4f}" if ev_hold else "")
        + f"; moved {d0[0]:.2f} deg / {d0[1]:.1f} cm from the start pose; camera centre {np.round(c2w[:3, 3], 3).tolist()}")
    np.save(os.path.join(a.out, "c2w.npy"), c2w)
    np.save(os.path.join(a.out, "w2c.npy"), w2c)
    np.save(os.path.join(a.out, "K.npy"), K0)
    status.update(mode="fit", c2w=c2w.tolist(), w2c=w2c.tolist(),
                  final=dict(fit=ev_fit, holdout=ev_hold, fit_mean_iou=float(np.mean([r["iou"] for r in ev_fit])),
                             holdout_mean_iou=float(np.mean([r["iou"] for r in ev_hold])) if ev_hold else None,
                             moved_deg_cm=[float(x) for x in d0], camera_centre=c2w[:3, 3].tolist(),
                             fit_names=[f"{f.ep}/{f.name}" for f in fit_frames], hold_names=[f"{f.ep}/{f.name}" for f in hold_frames]))
    json.dump(status, open(os.path.join(a.out, "status.json"), "w"), indent=1)
    if not a.no_sheets:
        sheet(fit_frames, w2c, os.path.join(a.out, "final_fit_sheet.jpg"))
        if hold_frames:
            sheet(hold_frames, w2c, os.path.join(a.out, "final_holdout_sheet.jpg"))
    log(f"wrote {a.out}")
    return status


if __name__ == "__main__":
    main()
