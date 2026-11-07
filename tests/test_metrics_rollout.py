import numpy as np
from PIL import Image

from doppel.metrics import frame_metrics, object_motion
from doppel.rollout import chained_rollout


def test_frame_metrics_excludes_frame0():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 256, (17, 24, 32, 3), dtype=np.uint8)
    pred = gt.copy()
    pred[1:] = np.clip(gt[1:].astype(int) + rng.integers(-8, 9, gt[1:].shape), 0, 255).astype(np.uint8)
    m = frame_metrics(pred, gt, lpips=False)
    mse = ((pred[1:].astype(float) - gt[1:]) ** 2).mean((1, 2, 3))
    assert np.isclose(m["psnr"], np.mean(10 * np.log10(255 ** 2 / mse)))
    assert np.isinf(m["psnr_per_frame"][0]) and len(m["ssim_per_frame"]) == 17


def test_object_motion():
    tr = np.zeros((5, 2, 40, 40), bool)
    for t in range(5):
        tr[t, 0, 10:20, 10 + 2 * t:20 + 2 * t] = True               # moves 8 px; sqrt(area) = 10
        tr[t, 1, 25:35, 25:35] = True                              # still
    mo = object_motion(tr)
    assert np.isclose(mo[0], 0.8) and mo[1] == 0.0


class _FakeWM:
    """predict() returns 17 frames whose frame 0 is the conditioning image and value = call index."""
    def __init__(self):
        self.calls = []

    def predict(self, image, render, actions, robot_type, seed, num_inference_steps):
        self.calls.append(np.asarray(image).copy())
        k = len(self.calls)
        out = np.full((17, 4, 4, 3), k, np.uint8)
        out[0] = np.asarray(image)
        return out


def test_chained_rollout():
    wm = _FakeWM()
    first = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
    wins = [{"render": None, "actions": None}] * 3
    out = chained_rollout(wm, first, wins, "panda", seeds=[1, 2, 3])
    assert out.shape[0] == 1 + 16 * 3
    assert (wm.calls[1] == 1).all() and (wm.calls[2] == 2).all()     # each window starts from the last prediction
    assert (out[0] == 0).all() and (out[-1] == 3).all()
