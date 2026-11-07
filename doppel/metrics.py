"""
Evaluation metrics: PSNR / SSIM / LPIPS over the predicted frames and the Structural
Consistency Score (SCS) with SAM 2.

* Image metrics are computed per frame on uint8 RGB at 240x432 and averaged over the
  predicted frames 1..16. Frame 0 is the conditioning frame and is excluded from the
  reported numbers (it is still recorded per frame).
  PSNR: data range 255. SSIM: scikit-image, 7x7 window, RGB (channel_axis=-1).
  LPIPS: AlexNet (torchmetrics, inputs in [0, 1]).
* SCS (Bagchi et al., EgoWM, Eq. 7): structures are segmented in frame 0 (SAM 2 automatic
  masks, up to 6 objects; object 0 is the embodiment's mask from the rendering when one is
  found), tracked through the ground-truth and the predicted video with the SAM 2 video
  predictor, and SCS is the mean IoU between the two tracks over objects and frames 1..16.
  ``scs_dynamic`` restricts the mean to non-embodiment objects that move in the ground truth
  (max centroid displacement >= ``motion_thresh`` x sqrt(frame-0 area)); ``scs_static`` to the rest.
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SAM2_ID = "facebook/sam2.1-hiera-small"
K_OBJECTS = 6
MOTION_THRESH = 0.25

_LPIPS = None


def lpips_model():
    """AlexNet LPIPS (torchmetrics), on GPU when available; ``None`` if unavailable."""
    global _LPIPS
    if _LPIPS is None:
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).eval()
            _LPIPS = m.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] LPIPS unavailable ({exc})")
            _LPIPS = False
    return _LPIPS or None


def frame_metrics(pred: np.ndarray, gt: np.ndarray, first: int = 1, lpips: bool = True) -> Dict[str, Any]:
    """Per-frame PSNR / SSIM / LPIPS of uint8 ``(T, H, W, 3)`` arrays; means over frames ``first..T-1``."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    if pred.shape != gt.shape or pred.dtype != np.uint8 or gt.dtype != np.uint8:
        raise ValueError(f"need two uint8 arrays of one shape, got {pred.shape}/{pred.dtype} {gt.shape}/{gt.dtype}")
    T = pred.shape[0]
    psnr = [float(peak_signal_noise_ratio(gt[t], pred[t], data_range=255)) for t in range(T)]
    ssim = [float(structural_similarity(gt[t], pred[t], data_range=255, channel_axis=-1)) for t in range(T)]
    out: Dict[str, Any] = {"psnr_per_frame": psnr, "ssim_per_frame": ssim,
                           "psnr": float(np.mean(psnr[first:])), "ssim": float(np.mean(ssim[first:]))}
    lp = lpips_model() if lpips else None
    if lp is not None:
        dev = next(lp.parameters()).device
        with torch.no_grad():
            a = torch.from_numpy(pred).to(dev).permute(0, 3, 1, 2).float() / 255.0
            b = torch.from_numpy(gt).to(dev).permute(0, 3, 1, 2).float() / 255.0
            lps = [float(lp(a[t:t + 1], b[t:t + 1])) for t in range(T)]
        out.update(lpips_per_frame=lps, lpips=float(np.mean(lps[first:])))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def embodiment_mask(render0: np.ndarray, thresh: int = 25, min_frac: float = 0.003) -> Optional[np.ndarray]:
    """Frame-0 mask of the embodiment from its rendering (non-black pixels), or None if too small."""
    from scipy import ndimage
    m = render0.max(axis=2) > thresh
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    m = ndimage.binary_closing(m, np.ones((5, 5)))
    return m if m.sum() >= min_frac * m.size else None


def pick_masks(amg, frame0: np.ndarray, k: int, min_frac: float = 0.005, max_frac: float = 0.40,
               overlap_thresh: float = 0.4) -> List[np.ndarray]:
    """Up to ``k`` SAM 2 automatic masks of frame 0, ranked by quality and size, mutually non-overlapping."""
    anns = amg.generate(frame0)
    hw = frame0.shape[0] * frame0.shape[1]
    anns = [a for a in anns if min_frac * hw <= a["area"] <= max_frac * hw]
    anns.sort(key=lambda a: a["predicted_iou"] * a["stability_score"] * np.sqrt(a["area"]), reverse=True)
    picked: List[np.ndarray] = []
    for a in anns:
        m = a["segmentation"]
        if any((m & p).sum() / min(m.sum(), p.sum()) > overlap_thresh for p in picked):
            continue
        picked.append(m)
        if len(picked) == k:
            break
    return picked


def object_motion(tracks: np.ndarray) -> List[Optional[float]]:
    """Per object: max_t ||centroid_t - centroid_0|| / sqrt(area_0) of ``(T, J, H, W)`` bool tracks."""
    out: List[Optional[float]] = []
    for j in range(tracks.shape[1]):
        cen = []
        for t in range(tracks.shape[0]):
            ys, xs = np.nonzero(tracks[t, j])
            cen.append([xs.mean(), ys.mean()] if len(xs) else [np.nan, np.nan])
        c = np.asarray(cen, np.float64)
        a0 = int(tracks[0, j].sum())
        if a0 == 0 or not np.isfinite(c[0]).all():
            out.append(None)
            continue
        d = np.linalg.norm(c - c[0], axis=-1)
        d = d[np.isfinite(d)]
        out.append(float(d.max() if d.size else 0.0) / float(np.sqrt(a0)))
    return out


class SCS:
    """Structural Consistency Score with SAM 2 (``pip install sam2``; weights from the HF Hub)."""

    def __init__(self, model_id: str = SAM2_ID, k_objects: int = K_OBJECTS):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        self.amg = SAM2AutomaticMaskGenerator(build_sam2_hf(model_id), points_per_side=16,
                                              pred_iou_thresh=0.7, stability_score_thresh=0.9)
        self.pred = SAM2VideoPredictor.from_pretrained(model_id)
        self.k = int(k_objects)

    def masks0(self, gt0: np.ndarray, render0: Optional[np.ndarray]) -> Tuple[List[np.ndarray], bool]:
        ms = pick_masks(self.amg, gt0, self.k)
        arm = embodiment_mask(render0) if render0 is not None else None
        if arm is not None:
            ms = [m for m in ms if (m & arm).sum() / max(m.sum(), 1) < 0.5]
            ms = [arm] + ms[: self.k - 1]
        return ms, arm is not None

    def track(self, frames: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """Propagate frame-0 masks through ``frames``; returns ``(T, J, H, W)`` bool."""
        import imageio.v3 as iio
        tmp = tempfile.mkdtemp(prefix="scs_")
        try:
            for i, f in enumerate(frames):
                iio.imwrite(f"{tmp}/{i:05d}.jpg", f, quality=95)
            state = self.pred.init_state(video_path=tmp)
            for oid, m in enumerate(masks):
                self.pred.add_new_mask(state, frame_idx=0, obj_id=oid, mask=torch.from_numpy(m))
            out = np.zeros((len(frames), len(masks), *frames[0].shape[:2]), dtype=bool)
            for t, oids, logits in self.pred.propagate_in_video(state):
                for j, oid in enumerate(oids):
                    out[t, oid] = (logits[j, 0] > 0).cpu().numpy()
            self.pred.reset_state(state)
            return out
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def __call__(self, pred: np.ndarray, gt: np.ndarray, render: Optional[np.ndarray] = None,
                 motion_thresh: float = MOTION_THRESH) -> Dict[str, Any]:
        """SCS of a prediction against its ground truth (uint8 ``(T, H, W, 3)``); ``render`` gives the embodiment mask."""
        masks, has_arm = self.masks0(gt[0], render[0] if render is not None else None)
        if not masks:
            return {}
        ac = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else _null()
        with ac:
            gtr = self.track(gt, masks)
            prr = self.track(pred, masks)
        T = min(len(gt), len(pred))
        io = np.array([[iou(prr[t, j], gtr[t, j]) for j in range(len(masks))] for t in range(T)])
        rec: Dict[str, Any] = {"scs": float(io[1:].mean()), "scs_n_objects": len(masks), "scs_has_arm": bool(has_arm)}
        first = 1 if has_arm else 0
        if has_arm:
            rec["scs_arm"] = float(io[1:, 0].mean())
        if len(masks) > first:
            rec["scs_objects"] = float(io[1:, first:].mean())
        mot = object_motion(gtr[:T])
        per = [float(io[1:, j].mean()) for j in range(len(masks))]
        dyn = [per[j] for j in range(first, len(masks)) if mot[j] is not None and mot[j] >= motion_thresh]
        sta = [per[j] for j in range(first, len(masks)) if mot[j] is not None and mot[j] < motion_thresh]
        rec.update(n_dynamic=len(dyn), n_static=len(sta), motion_thresh=motion_thresh)
        if dyn:
            rec["scs_dynamic"] = float(np.mean(dyn))
        if sta:
            rec["scs_static"] = float(np.mean(sta))
        return rec


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
