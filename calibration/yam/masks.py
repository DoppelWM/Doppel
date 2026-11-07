"""Robot masks (both YAM arms, cables excluded) for the extracted frames of one episode, without extrinsics.

1. render both arms at the recorded joints from the simulator top camera with the episode's K,
2. one 2-D similarity (dx, dy, scale, rotation) aligning the render to all frames of the episode
   (masked NCC of render vs real grayscale, +-130 px search),
3. SAM 2 per frame and arm, one query per positive point inside the aligned render (white-link and black-motor
   points chosen from the render's colours); negatives = far background, the other arm's points, purple cable and
   table-wood pixels next to the arm; per query the largest candidate that lies mostly inside the dilated render
   is kept; orange (wood) and purple (cable) pixels and thin cable-like strands are removed.

Output: <out>/<frame>.png (uint8, 255 = robot), counts.json, align.json.

    python -m yam.masks <episode_dir> --model <yam model dir> [--sam2_ckpt sam2.1_hiera_small.pt]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from . import station

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def sam2_checkpoint(path=None):
    if path:
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt")


def load_sam2(ckpt=None, device="cuda"):
    from sam2.build_sam import build_sam2
    return build_sam2(SAM2_CFG, sam2_checkpoint(ckpt), device=device)


# ------------------------------------------------------------------------------------------------ inputs
def stage_episode(ep_dir):
    """Frames (JPEG, quality 95, as the masks were made) and states with list-valued grippers. For cx-104
    episodes the undist_cxfix frames and the corrected K are used. Returns (states dict, frames dir)."""
    st = json.load(open(os.path.join(ep_dir, "states.json")))
    hyp = st.get("cx_shift_hypothesis")
    use_fix = bool(hyp and hyp.get("K_hypothesis"))
    sub = "undist_cxfix" if use_fix else "undist"
    calib = dict(st["calib"])
    if use_fix:
        calib["K"] = list(np.asarray(hyp["K_hypothesis"], np.float64).reshape(-1))
    fdir = os.path.join(ep_dir, "mask_frames")
    os.makedirs(fdir, exist_ok=True)
    recs = []
    for r in st["records"]:
        src = os.path.join(ep_dir, sub, r["png"])
        if not os.path.exists(src):
            continue
        Image.open(src).convert("RGB").save(os.path.join(fdir, r["png"].replace(".png", ".jpg")), quality=95)
        g = lambda v: v if isinstance(v, list) else [v]
        recs.append(dict(frame=r["frame"], left_q=r["left_q"], left_grip=g(r["left_grip"]),
                         right_q=r["right_q"], right_grip=g(r["right_grip"])))
    return dict(calib=calib, records=recs), fdir


# ------------------------------------------------------------------------------------------------ helpers
def fps_points(mask, k, first=None):
    """Farthest-point sampling of k pixel coords (x, y) inside a boolean mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 2), np.float32)
    P = np.stack([xs, ys], 1).astype(np.float32)
    if len(P) > 4000:
        P = P[np.random.RandomState(0).choice(len(P), 4000, replace=False)]
    sel = [0 if first is None else first]
    d = np.linalg.norm(P - P[sel[0]], axis=1)
    for _ in range(1, min(k, len(P))):
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return P[sel]


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def warp_mask(m, M):
    return cv2.warpAffine(m.astype(np.float32), M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0.0)


def purple(hsv):
    return (hsv[..., 0] >= 120) & (hsv[..., 0] <= 170) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 60)


def wood(hsv):
    return (hsv[..., 0] >= 5) & (hsv[..., 0] <= 32) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 90)


def wood_any(hsv):
    """Lit or shadowed table wood (orange hue, saturated, brighter than the black robot parts)."""
    return (hsv[..., 0] >= 5) & (hsv[..., 0] <= 32) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 40)


def ring_points(mask, hsv, sel_fn, r_in, r_out, k):
    d = ndimage.distance_transform_edt(~mask)
    cand = ndimage.binary_opening((d >= r_in) & (d <= r_out) & sel_fn(hsv), iterations=1)
    return fps_points(cand, k) if cand.sum() > 30 else np.zeros((0, 2), np.float32)


def eroded(m, px, min_px=40):
    while px > 0:
        e = ndimage.binary_erosion(m, iterations=px)
        if e.sum() >= min_px:
            return e
        px -= 2
    return m


def material_prompts(arm_mask, render_gray, hsv, k_bright=4, k_dark=3):
    """Positive points inside one arm's aligned render: white links (render bright) + black motors (render dark)."""
    bright = eroded(arm_mask & (render_gray > 150), 6)
    dark = eroded(arm_mask & (render_gray < 110), 2)
    P = np.concatenate([fps_points(bright, k_bright), fps_points(dark, k_dark)]) if (bright.any() or dark.any()) else np.zeros((0, 2), np.float32)
    if len(P) < 3:
        P = fps_points(eroded(arm_mask, 8), 6)
    if len(P):
        ok = hsv[..., 1][P[:, 1].astype(int), P[:, 0].astype(int)] < 70   # drop points on wood / cable
        if ok.sum() >= 3:
            P = P[ok]
    return P


# ------------------------------------------------------------------------------------------------ 2-D alignment
def _align_prep(items, n_px, dev):
    h, w = items[0][2].shape
    cx, cy = w / 2.0, h / 2.0
    pre = []
    for alpha, rgb_render, gray in items:
        m = ndimage.binary_erosion(alpha > 0.5, iterations=2)
        ys, xs = np.nonzero(m)
        if len(ys) < 200:
            continue
        rs = np.random.RandomState(0)
        if len(ys) > n_px:
            i = rs.choice(len(ys), n_px, replace=False)
            ys, xs = ys[i], xs[i]
        tv = cv2.cvtColor(rgb_render, cv2.COLOR_BGR2GRAY).astype(np.float32)[ys, xs]
        pre.append(dict(X=torch.as_tensor(xs.astype(np.float32) - cx, device=dev, dtype=torch.float64),
                        Y=torch.as_tensor(ys.astype(np.float32) - cy, device=dev, dtype=torch.float64),
                        tv=torch.as_tensor(tv, device=dev, dtype=torch.float64),
                        g=torch.as_tensor(gray.astype(np.float32), device=dev, dtype=torch.float64)))
    return pre, h, w, cx, cy


def _align_scores(pre, P, h, w, cx, cy, min_valid, chunk=512):
    """P (n, 4) [dx, dy, s, th] -> mean masked NCC over frames (n,); -1 for a frame with too few valid points."""
    out = []
    for i in range(0, P.shape[0], chunk):
        p = P[i:i + chunk]
        dx, dy, s, th = (p[:, k:k + 1] for k in range(4))
        c, sn = torch.cos(th), torch.sin(th)
        acc = torch.zeros(p.shape[0], device=P.device, dtype=torch.float64)
        for f in pre:
            X, Y = f["X"][None], f["Y"][None]
            xx = torch.round(s * (c * X - sn * Y) + cx + dx).long()
            yy = torch.round(s * (sn * X + c * Y) + cy + dy).long()
            v = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            nv = v.sum(1).double()
            rv = f["g"][yy.clamp(0, h - 1), xx.clamp(0, w - 1)]
            tv = f["tv"][None].expand_as(rv)
            vf = v.double()
            n = nv.clamp(min=1)
            mt = (tv * vf).sum(1) / n
            mr = (rv * vf).sum(1) / n
            st = torch.sqrt((((tv - mt[:, None]) ** 2) * vf).sum(1) / n)
            sr = torch.sqrt((((rv - mr[:, None]) ** 2) * vf).sum(1) / n)
            ncc = ((((tv - mt[:, None]) / (st[:, None] + 1e-6)) * ((rv - mr[:, None]) / (sr[:, None] + 1e-6))) * vf).sum(1) / n
            acc += torch.where(nv < min_valid * X.shape[1], torch.full_like(ncc, -1.0), ncc)
        out.append(acc / len(pre))
    return torch.cat(out)


def _align_best(pre, cands, best, h, w, cx, cy, min_valid, dev):
    sc = _align_scores(pre, torch.as_tensor(np.asarray(cands, np.float64), device=dev), h, w, cx, cy, min_valid)
    j = int(torch.argmax(sc))
    return (*cands[j], float(sc[j])) if float(sc[j]) > best[4] else best


def align_render_multi(items, search=130, n_px=4000, min_valid=0.6, dev="cuda"):
    """One 2-D similarity shared by all frames of a fixed camera: items = [(alpha, render_bgr, gray_real), ...].
    Coarse shift grid, then scale x rotation x shift, then a fine pass; first maximum in scan order wins.
    Returns (M 2x3, (dx, dy, scale, rot_deg), score, per-frame scores)."""
    pre, h, w, cx, cy = _align_prep(items, n_px, dev)
    if not pre:
        return np.float32([[1, 0, 0], [0, 1, 0]]), (0, 0, 1.0, 0.0), 0.0, []
    args = (h, w, cx, cy, min_valid)
    s0 = float(_align_scores(pre, torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=dev, dtype=torch.float64), *args)[0])
    best = (0, 0, 1.0, 0.0, s0)
    best = _align_best(pre, [(dx, dy, 1.0, 0.0) for dy in range(-search, search + 1, 5) for dx in range(-search, search + 1, 5)],
                       best, *args, dev)
    dx0, dy0 = best[:2]
    best = _align_best(pre, [(dx, dy, float(s), float(th)) for s in np.arange(0.8, 1.21, 0.05) for th in np.radians(np.arange(-10, 10.1, 2.5))
                             for dy in range(dy0 - 20, dy0 + 21, 5) for dx in range(dx0 - 20, dx0 + 21, 5)], best, *args, dev)
    dx0, dy0, s0_, th0 = best[:4]
    best = _align_best(pre, [(dx, dy, float(s), float(th)) for s in np.arange(s0_ - 0.04, s0_ + 0.041, 0.02)
                             for th in th0 + np.radians(np.arange(-2, 2.1, 1.0))
                             for dy in range(dy0 - 4, dy0 + 5, 2) for dx in range(dx0 - 4, dx0 + 5, 2)], best, *args, dev)
    dx, dy, s, th, sc = best
    c, sn = np.cos(th), np.sin(th)
    M = np.float32([[s * c, -s * sn, cx + dx - s * (c * cx - sn * cy)], [s * sn, s * c, cy + dy - s * (sn * cx + c * cy)]])
    per = [round(float(_align_scores([p], torch.tensor([[dx, dy, s, th]], device=dev, dtype=torch.float64), *args)[0]), 3) for p in pre]
    return M, (int(dx), int(dy), float(s), float(np.degrees(th))), sc, per


# ------------------------------------------------------------------------------------------------ main
class DeformCache:
    """Each frame is rendered three times (both arms, left, right): keep the last two arm deformations."""

    def __init__(self, size=2):
        self.size, self.items = size, []

    def __call__(self, pc, q):
        k = np.asarray(q, np.float32).tobytes()
        for kk, v in self.items:
            if kk == k:
                return v
        v = station.deform_arm(pc, q)
        self.items.append((k, v))
        if len(self.items) > self.size:
            self.items.pop(0)
        return v


def robot_masks(pc, sam_model, ep_dir, out=None, search=130, k_bright=6, k_dark=8, inside_frac=0.6, wood_veto=0.2):
    """Write <out>/<frame>.png masks for one extracted episode (out defaults to <ep_dir>/masks)."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    out = out or os.path.join(ep_dir, "masks")
    os.makedirs(out, exist_ok=True)
    st, fdir = stage_episode(ep_dir)
    recs = st["records"]
    K = np.array(st["calib"]["K"], np.float64).reshape(3, 3)
    W, H = int(st["calib"]["w"]), int(st["calib"]["h"])
    _, T = station.sim_top_camera(W, H)
    deform = DeformCache()

    # 1. coarse render at the simulator camera (the colour render goes through JPEG, as in the original runs)
    data, names = {}, []
    for r in recs:
        n = f"frame_{r['frame']:05d}"
        path = f"{fdir}/{n}.jpg"
        if not os.path.exists(path):
            continue
        ql = station.abc_state_to_q(r["left_q"], r["left_grip"][0])
        qr = station.abc_state_to_q(r["right_q"], r["right_grip"][0])
        rgb, _ = station.render_two_arms(pc, K, W, H, T, ql, qr, deform=deform)
        _, aL = station.render_two_arms(pc, K, W, H, T, ql, qr, arms=("left",), deform=deform)
        _, aR = station.render_two_arms(pc, K, W, H, T, ql, qr, arms=("right",), deform=deform)
        rgbr = cv2.imdecode(cv2.imencode(".jpg", (rgb[..., ::-1] * 255).astype(np.uint8))[1], cv2.IMREAD_COLOR)
        img = cv2.imread(path)
        data[n] = dict(img=img, aL=aL.astype(np.float32), aR=aR.astype(np.float32), rgbr=rgbr,
                       gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        names.append(n)

    # 2. shared alignment
    t0 = time.time()
    M, params, sc, per = align_render_multi([(np.maximum(d["aL"], d["aR"]), d["rgbr"], d["gray"]) for d in data.values()], search=search)
    json.dump(dict(dx=params[0], dy=params[1], scale=round(params[2], 3), rot_deg=round(params[3], 2), ncc_mean=round(sc, 3),
                   ncc_per_frame=dict(zip(names, per)), M=M.tolist()), open(f"{out}/align.json", "w"), indent=1)
    print(f"[masks] alignment dx {params[0]} dy {params[1]} scale {params[2]:.3f} rot {params[3]:.2f} deg, "
          f"NCC {sc:.3f} ({time.time() - t0:.1f}s)", flush=True)

    # 3. SAM 2
    pred = SAM2ImagePredictor(sam_model)
    counts = {}
    for n in names:
        d = data[n]
        img = d["img"]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        aligned = {"left": warp_mask(d["aL"] > 0.5, M) > 0.5, "right": warp_mask(d["aR"] > 0.5, M) > 0.5}
        rgray = cv2.warpAffine(cv2.cvtColor(d["rgbr"], cv2.COLOR_BGR2GRAY), M, (W, H), borderValue=255)
        union = aligned["left"] | aligned["right"]
        pos = {k: material_prompts(m, rgray, hsv, k_bright, k_dark) for k, m in aligned.items()}
        extra = {k: np.concatenate([ring_points(m, hsv, purple, 3, 90, 4), ring_points(m, hsv, wood, 15, 60, 4)]) for k, m in aligned.items()}
        neg_bg = fps_points(ndimage.distance_transform_edt(~union) > 90, 8)
        masks, info = {}, {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred.set_image(img[..., ::-1].copy())
            for k in aligned:
                if len(pos[k]) == 0:
                    masks[k] = np.zeros((H, W), bool)
                    info[k] = dict(px=0)
                    continue
                others = np.concatenate([pos[o] for o in pos if o != k])
                negs = np.concatenate([neg_bg, others, extra[k]])
                big = ndimage.binary_dilation(aligned[k], iterations=12)
                big20 = ndimage.binary_dilation(aligned[k], iterations=20)
                wood_px = wood_any(hsv)
                m = np.zeros((H, W), bool)
                for p in pos[k]:   # one query per positive point
                    ms, scs, _ = pred.predict(point_coords=np.concatenate([p[None], negs]), point_labels=np.r_[1, np.zeros(len(negs))],
                                              multimask_output=True)
                    best = None
                    for mm in ms:
                        mm = mm > 0
                        area = mm.sum()
                        if area < 30 or (mm & big20).sum() < inside_frac * area or wood_px[mm].mean() > wood_veto:
                            continue
                        if best is None or area > best[0]:
                            best = (area, mm)
                    if best is not None:
                        m |= best[1]
                m = m & ~wood_px                       # the robot is achromatic
                lab, nc = ndimage.label(m)
                keep = np.zeros_like(m)
                for i in range(1, nc + 1):             # components that touch the render
                    c = lab == i
                    if (c & big).sum() > 0.3 * c.sum():
                        keep |= c
                holes = ndimage.binary_fill_holes(keep) & ~keep
                lh, nh = ndimage.label(holes)
                for i in range(1, nh + 1):             # small holes only
                    c = lh == i
                    if c.sum() <= 400:
                        keep |= c
                m = ndimage.binary_closing(keep, iterations=2) & ~ndimage.binary_dilation(purple(hsv), iterations=1)
                thin = m & ~ndimage.binary_opening(m, structure=np.ones((7, 7), bool))
                cable_like = thin & (hsv[..., 2] >= 70) & (hsv[..., 2] <= 215) & ~((hsv[..., 1] < 15) & (hsv[..., 2] > 200))
                m = m & ~ndimage.binary_dilation(cable_like, iterations=1)
                lab2, n2 = ndimage.label(m)
                keep2 = np.zeros_like(m)
                for i in range(1, n2 + 1):
                    c = lab2 == i
                    if c.sum() >= 150 and (c & big).sum() > 0.3 * c.sum():
                        keep2 |= c
                masks[k] = keep2
                info[k] = dict(px=int(keep2.sum()), iou_render=round(iou(m, aligned[k]), 3), n_pos=int(len(pos[k])))
        final = masks["left"] | masks["right"]
        cv2.imwrite(f"{out}/{n}.png", (final * 255).astype(np.uint8))
        counts[n] = dict(px=int(final.sum()), left=info["left"], right=info["right"], iou_render_union=round(iou(final, union), 3))
    json.dump(counts, open(f"{out}/counts.json", "w"), indent=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="directory written by yam.extract")
    ap.add_argument("--model", required=True, help="YAM Dr.Robot model directory")
    ap.add_argument("--iteration", type=int, default=None)
    ap.add_argument("--sam2_ckpt", default=None, help="sam2.1_hiera_small.pt (default: download from Hugging Face)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    robot_masks(station.load_model(a.model, a.iteration), load_sam2(a.sam2_ckpt), a.episode, a.out)


if __name__ == "__main__":
    main()
