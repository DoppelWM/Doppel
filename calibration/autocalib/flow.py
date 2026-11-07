"""Dense 2-D flow of the real video, the supervision signal.

* ``raft_flow``        torchvision RAFT-large between consecutive frames, validity from a forward-backward check.
                       Used for the released Bridge and RoboMIND poses.
* ``cotracker_flow``   CoTracker3 (offline) dense tracks over sliding windows, as in RobotArena∞.

Both return ``fwd_flows`` (T-1, 2, H, W) and ``fwd_valids`` (T-1, 1, H, W) float tensors on CPU. The calibration
reads them from a per-trajectory cache file (``tracker_data*.pth``); ``tools/sweep_flow.py`` writes the caches.
"""
import os

import numpy as np
import torch

_COTRACKER = None


def load_cotracker(device='cuda'):
    global _COTRACKER
    if _COTRACKER is None:
        _COTRACKER = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device, non_blocking=True)
    return _COTRACKER


def run_tracker_on_images(model, images, grid_size=0, n_frame=5, grid_step=1, bf16=False):
    """Dense tracks -> forward flow between consecutive frames.

    images: (T, C, H, W) float tensor on GPU (0-255). Returns flows (T-1, 2, H, W), valid masks (T-1, H, W).

    grid_step=1 tracks every pixel; grid_step=k > 1 tracks one point every k pixels in one sparse pass (untracked
    pixels stay invalid). bf16 runs the tracker under bfloat16 autocast.
    """
    assert grid_size == 0  # only dense track is supported so far
    model.eval()

    T, C, H, W = images.shape
    flows = torch.zeros(T - 1, H, W, 2).cuda()
    valid_masks = torch.zeros(T - 1, H, W).cuda()

    def check_pos(x, y):
        return (x >= 0) & (x < W) & (y >= 0) & (y < H)

    pos = 0
    while pos < T - n_frame:
        if (T - pos) < 2 * n_frame or n_frame == 1:  # include last few frames into the last track
            start, end = pos, T
        else:
            start, end = pos, pos + n_frame
        if grid_step > 1:  # sparse grid queries on the window's first frame
            xs = torch.arange(0, W, grid_step, device=images.device)
            ys = torch.arange(0, H, grid_step, device=images.device)
            g = torch.stack(torch.meshgrid(ys, xs, indexing='ij'), -1).reshape(-1, 2).float()
            queries = torch.cat([torch.zeros(len(g), 1, device=g.device), g[:, 1:2], g[:, 0:1]], 1).unsqueeze(0)
        else:
            queries = None
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(bf16)):
            pred_tracks, pred_visibility = model(images[start:end].unsqueeze(0), queries=queries, grid_size=grid_size)
        pred_tracks, pred_visibility = pred_tracks.float(), pred_visibility.float()
        pred_tracks, pred_visibility = pred_tracks.squeeze(0), pred_visibility.squeeze(0)  # (end-start) N 2, (end-start) N 1

        for t in range(start, end - 1):
            pred_track, pred_track_next, visibilities = pred_tracks[t - start], pred_tracks[t + 1 - start], pred_visibility[t - start]
            x1 = torch.round(pred_track[:, 0]).long()
            y1 = torch.round(pred_track[:, 1]).long()
            x2 = pred_track_next[:, 0]
            y2 = pred_track_next[:, 1]
            valid = check_pos(x1, y1)
            x1 = x1[valid]
            y1 = y1[valid]
            x2 = x2[valid]
            y2 = y2[valid]
            visibilities = visibilities[valid]
            flows[t, y1, x1, 0] = x2 - x1.float()
            flows[t, y1, x1, 1] = y2 - y1.float()
            valid_masks[t, y1, x1] = visibilities.float()

        pos = end - 1  # one-frame buffer

    return flows.permute(0, 3, 1, 2), valid_masks  # T-1 2 H W, T-1 H W


def cotracker_flow(images_thwc_uint8, device='cuda', n_frame=5, grid_step=1, bf16=False):
    """images: (T, H, W, 3) uint8 numpy. Returns dict(fwd_flows=(T-1,2,H,W), fwd_valids=(T-1,1,H,W), meta=...) on CPU.
    ``meta`` records how the flow was made (grid_step, bf16, n_frame) so a cache file carries its provenance."""
    imgs = torch.stack([torch.from_numpy(im).float().cuda().permute(2, 0, 1).to(device, non_blocking=True) for im in images_thwc_uint8])
    model = load_cotracker(device)
    fwd_flows, fwd_valids = run_tracker_on_images(model, imgs, n_frame=n_frame, grid_step=grid_step, bf16=bf16)  # T-1 2 H W, T-1 H W
    fwd_valids = fwd_valids.unsqueeze(1)
    return {'fwd_flows': fwd_flows.detach().cpu(), 'fwd_valids': fwd_valids.detach().cpu(),
            'meta': {'backend': 'cotracker3_offline', 'n_frame': n_frame, 'grid_step': grid_step, 'bf16': bool(bf16)}}


def cached_cotracker_flow(images_thwc_uint8, cache_file, device='cuda', n_frame=5):
    """Compute (or load) the CoTracker flow cache ``cache_file`` (torch.save of the dict above)."""
    if not os.path.exists(cache_file):
        torch.save(cotracker_flow(images_thwc_uint8, device, n_frame), cache_file)
    return torch.load(cache_file)


_RAFT = None


def raft_flow(images_thwc_uint8, device='cuda', batch=8, fb_thresh=1.0, fb_rel=0.05):
    """Dense optical flow with torchvision RAFT-large (default weights, C_T_SKHT_V2).

    Returns the same dict as ``cotracker_flow``: fwd_flows (T-1, 2, H, W), the displacement of every pixel of frame t
    to frame t+1, and fwd_valids (T-1, 1, H, W) from the forward-backward check
    |fwd(p) + bwd(p + fwd(p))| < fb_thresh + fb_rel * |fwd(p)|, with the destination inside the frame.
    """
    global _RAFT
    import torch.nn.functional as F
    if _RAFT is None:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _RAFT = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    imgs = torch.from_numpy(np.ascontiguousarray(images_thwc_uint8)).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    T, _, H, W = imgs.shape
    fwd, bwd = [], []
    with torch.no_grad():
        for i in range(0, T - 1, batch):
            a, b = imgs[i:i + batch], imgs[i + 1:i + 1 + batch]
            n = min(len(a), len(b))
            fwd.append(_RAFT(a[:n], b[:n])[-1])
            bwd.append(_RAFT(b[:n], a[:n])[-1])
        fwd, bwd = torch.cat(fwd), torch.cat(bwd)                       # (T-1, 2, H, W)
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        qx, qy = xs[None] + fwd[:, 0], ys[None] + fwd[:, 1]             # destination of every pixel
        grid = torch.stack([2 * qx / (W - 1) - 1, 2 * qy / (H - 1) - 1], -1)
        bwd_at_q = F.grid_sample(bwd, grid, mode='bilinear', padding_mode='border', align_corners=True)
        err = (fwd + bwd_at_q).norm(dim=1)
        inside = (qx >= 0) & (qx <= W - 1) & (qy >= 0) & (qy <= H - 1)
        valid = (err < fb_thresh + fb_rel * fwd.norm(dim=1)) & inside
    return {'fwd_flows': fwd.cpu(), 'fwd_valids': valid.float().unsqueeze(1).cpu(),
            'meta': {'backend': 'raft_large', 'fb_thresh': fb_thresh, 'fb_rel': fb_rel}}
