"""Frames and joint states of one ABC-130K episode (MCAP), for the camera fit.

    python -m yam.extract <episode.mcap> <out_dir> [--every_s 3] [--max 30]

Writes ``frame_XXXXX.png`` (decoded frame, index = message index in the top-camera stream),
``undist/frame_XXXXX.png`` (undistorted to the same K), ``undist_cxfix/`` (only for a 640-wide stream whose
calibration has the 848-wide principal point: the stream is the centre crop of the 848 profile, so cx - 104) and
``states.json`` (calibration, per-frame joints and grippers of both arms).

Frame k of the decoder output belongs to message k (the decoder delay does not shift the association). States are
matched to frames by nearest log time. The published distortion is applied as OpenCV's forward Brown-Conrady /
rational model (RealSense ``inverse_brown_conrady`` coefficients are used by the SDK the same way).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image


def pb_ns(dec):
    return int(dec.timestamp.seconds) * 10 ** 9 + int(dec.timestamp.nanos)


def read_mcap(path):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    frames, calib = [], {}
    arm = {"left": [], "right": []}
    ee = {"left": [], "right": []}
    with open(path, "rb") as f:
        rd = make_reader(f, decoder_factories=[DecoderFactory()])
        summ = rd.get_summary()
        topics = {ch.topic for ch in summ.channels.values()}
        top = "/top-camera" if "/top-camera" in topics else "/top-left-camera"
        want = [top, top + "-info", "/left-arm-state", "/right-arm-state", "/left-ee-state", "/right-ee-state"]
        for _, ch, msg, dec in rd.iter_decoded_messages(topics=want):
            t = ch.topic
            if t == top:
                frames.append((int(msg.log_time), pb_ns(dec), dec.format, bytes(dec.data)))
            elif t == top + "-info":
                calib = dict(w=int(dec.width), h=int(dec.height), K=[float(v) for v in dec.K], D=[float(v) for v in dec.D],
                             model=str(dec.distortion_model))
            elif t in ("/left-arm-state", "/right-arm-state"):
                arm[t.split("-")[0][1:]].append((int(msg.log_time), [float(v) for v in dec.position]))
            elif t in ("/left-ee-state", "/right-ee-state"):
                v = dec.position
                ee[t.split("-")[0][1:]].append((int(msg.log_time), float(v[0]) if len(v) else float("nan")))
    return dict(top=top, frames=frames, calib=calib, arm=arm, ee=ee)


def decode_wanted(frames, want_ts, threads):
    """Decode the stream in order; return [(frame_idx, rgb)] for the first frame at/after each wanted time."""
    import av
    fmt = frames[0][2]
    codec = av.CodecContext.create("hevc" if fmt in ("h265", "hevc") else "h264", "r")
    if threads > 1:
        codec.thread_type = "AUTO"
        codec.thread_count = threads
    ts = np.array([f[0] for f in frames], np.int64)
    keep_idx = []
    for wt in want_ts:
        j = int(np.searchsorted(ts, wt, side="left"))
        if j >= len(ts):
            break
        if keep_idx and j <= keep_idx[-1]:
            j = keep_idx[-1] + 1
            if j >= len(ts):
                break
        keep_idx.append(j)
    keep_set, last_needed = set(keep_idx), (keep_idx[-1] if keep_idx else -1)
    out, nout, done = {}, 0, False
    for fr in frames:
        for pkt in codec.parse(fr[3]):
            for f in codec.decode(pkt):
                if nout in keep_set:
                    out[nout] = f.to_ndarray(format="rgb24")
                if nout >= last_needed:
                    done = True
                nout += 1
        if done:
            break
    if not done:  # flush for the trailing frames
        for pkt in list(codec.parse(None)) + [None]:
            try:
                fl = codec.decode(pkt)
            except Exception:
                fl = []
            for f in fl:
                if nout in keep_set:
                    out[nout] = f.to_ndarray(format="rgb24")
                nout += 1
    return [(k, out[k]) for k in keep_idx if k in out]


def forward_distort(x, y, D):
    """Ideal normalised (x, y) -> distorted, D = k1 k2 p1 p2 k3 [k4 k5 k6]; empty / zero D = identity."""
    D = list(D) + [0.0] * (8 - len(D)) if len(D) < 8 else list(D[:8])
    k1, k2, p1, p2, k3, k4, k5, k6 = D
    r2 = x * x + y * y
    rad = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    return xd, yd


def undistort_maps(K, D, w, h, K_new):
    """Sampling maps (cv2.initUndistortRectifyMap semantics) from the pinhole K_new image into the raw image."""
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    nfx, nfy, ncx, ncy = K_new[0], K_new[4], K_new[2], K_new[5]
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    xd, yd = forward_distort((u - ncx) / nfx, (v - ncy) / nfy, D)
    return (xd * fx + cx).astype(np.float32), (yd * fy + cy).astype(np.float32)


def remap_bilinear(img, map_x, map_y):
    """cv2.remap(INTER_LINEAR, BORDER_CONSTANT 0) in numpy."""
    h, w = img.shape[:2]
    x0 = np.floor(map_x).astype(np.int64)
    y0 = np.floor(map_y).astype(np.int64)
    wx = (map_x - x0)[..., None]
    wy = (map_y - y0)[..., None]
    out = np.zeros_like(img, dtype=np.float32)
    for dy, wy_ in ((0, 1 - wy), (1, wy)):
        for dx, wx_ in ((0, 1 - wx), (1, wx)):
            xx, yy = x0 + dx, y0 + dy
            valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            src = img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)].astype(np.float32)
            out += (wx_ * wy_) * src * valid[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def nearest(ts_arr, ts):
    j = int(np.argmin(np.abs(ts_arr - ts)))
    return j, float((ts_arr[j] - ts) / 1e6)


def extract(mcap_path, out, every_s=3.0, max_frames=30, start_s=0.0, threads=4, task=None):
    t_start = time.time()
    os.makedirs(out, exist_ok=True)
    R = read_mcap(mcap_path)
    frames, calib = R["frames"], R["calib"]
    if not frames:
        raise RuntimeError(f"{mcap_path}: no top-camera frames")
    ts = np.array([f[0] for f in frames], np.int64)
    if np.any(np.diff(ts) < 0):
        order = np.argsort(ts, kind="stable")
        frames = [frames[i] for i in order]
        ts = ts[order]
    t0 = int(ts[0])
    want = [t0 + int(round((start_s + k * every_s) * 1e9)) for k in range(max_frames)]
    kept = decode_wanted(frames, want, threads)
    if not kept:
        raise RuntimeError(f"{mcap_path}: nothing decoded")
    dh, dw = kept[0][1].shape[:2]

    # 1280x720 streams that publish the 848x480-profile K: the 848 profile is the 1280 sensor scaled by 2/3 and
    # centre-cropped to 848 (2.67 px a side), so K_1280 = 1.5 * (K_848 + crop)
    if calib.get("w") == 1280 and calib.get("h") == 720 and calib["K"][2] < 0.4 * 1280:
        Kp = list(calib["K"])
        crop = (1280 / 1.5 - 848) / 2
        calib["K_published_848"] = Kp
        calib["K"] = [1.5 * Kp[0], 0.0, 1.5 * (Kp[2] + crop), 0.0, 1.5 * Kp[4], 1.5 * Kp[5], 0.0, 0.0, 1.0]
        calib["K_rule"] = "1280x720 stream with an 848x480 K: K = 1.5 * (K_848 + 2.67 px crop)"
    K, D = calib["K"], calib["D"]
    os.makedirs(f"{out}/undist", exist_ok=True)
    map_x, map_y = undistort_maps(K, D, dw, dh, K)
    cxfix = None
    if calib.get("w") == 640 and abs(K[2] - 640 / 2) > 0.1 * 640:
        # 640-wide stream with the principal point of an 848-wide image: centre crop of the 848 profile
        K_fix = list(K)
        K_fix[2] = K[2] - (848 - 640) / 2
        fmx, fmy = undistort_maps(K_fix, D, dw, dh, K_fix)
        cxfix = dict(K_hypothesis=K_fix, cx_shift_px=-(848 - 640) / 2)
        os.makedirs(f"{out}/undist_cxfix", exist_ok=True)

    arm_ts = {s: np.array([r[0] for r in R["arm"][s]], np.int64) for s in ("left", "right")}
    ee_ts = {s: np.array([r[0] for r in R["ee"][s]], np.int64) for s in ("left", "right")}
    records = []
    for k, img in kept:
        fts = int(ts[k])
        name = f"frame_{k:05d}.png"
        Image.fromarray(img).save(f"{out}/{name}", compress_level=3)
        Image.fromarray(remap_bilinear(img, map_x, map_y)).save(f"{out}/undist/{name}", compress_level=3)
        rec = dict(frame=int(k), t_s=(fts - t0) / 1e9, timestamp_ns=fts, png=name)
        if cxfix:
            Image.fromarray(remap_bilinear(img, fmx, fmy)).save(f"{out}/undist_cxfix/{name}", compress_level=3)
        for s in ("left", "right"):
            if len(arm_ts[s]):
                j, dt = nearest(arm_ts[s], fts)
                rec[f"{s}_q"], rec[f"{s}_dt_ms"] = R["arm"][s][j][1], dt
            else:
                rec[f"{s}_q"], rec[f"{s}_dt_ms"] = None, None
            if len(ee_ts[s]):
                j, dt = nearest(ee_ts[s], fts)
                rec[f"{s}_grip"], rec[f"{s}_grip_dt_ms"] = R["ee"][s][j][1], dt
            else:
                rec[f"{s}_grip"], rec[f"{s}_grip_dt_ms"] = None, None
        records.append(rec)
    task = task or os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(mcap_path))))
    json.dump(dict(source_mcap=os.path.abspath(mcap_path), task=task, top=R["top"], calib=calib,
                   decoded_w=int(dw), decoded_h=int(dh), cx_shift_hypothesis=cxfix,
                   video=dict(n_msgs=len(frames), duration_s=float((ts[-1] - t0) / 1e9)),
                   n_records=len(records), records=records), open(f"{out}/states.json", "w"), indent=1)
    print(f"[extract] {task}: {len(records)} frames {dw}x{dh} (calib {calib.get('w')}x{calib.get('h')}, "
          f"{calib.get('model')}{', cx - 104' if cxfix else ''}) -> {out} ({time.time() - t_start:.1f}s)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mcap")
    ap.add_argument("out")
    ap.add_argument("--every_s", type=float, default=3.0, help="one frame every this many seconds")
    ap.add_argument("--max", type=int, default=30, help="max frames")
    ap.add_argument("--start_s", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=4, help="decoder threads")
    a = ap.parse_args(argv)
    extract(a.mcap, a.out, a.every_s, a.max, a.start_s, a.threads)


if __name__ == "__main__":
    sys.exit(main())
