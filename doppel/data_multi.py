"""
Multi-embodiment training data read from the latent store.

Embodiments (``robot_type``, dataset, action width D):

* ``panda`` (DROID, 8), ``ur5e_2f85`` (RoboMIND, 8), ``widowx`` (Bridge, 8): joints + gripper;
* ``yam_bimanual`` (ABC, 14), ``vega1_sharpa`` (T-Rex, 58), ``mano_hands`` (GigaHands, 108),
  ``smplx_human`` (OakInk2, 159), ``smplh_human`` (Ego-Exo4D, 159).

Embodiments of different widths share one action encoder: every action is z-scored in its own
width and then zero-extended to ``action_pad_dim`` slots (:func:`pad_actions`). There is no
embodiment label; the rendering tells the model which body it is looking at.

* Frozen-encoder outputs (VAE latents of the video and of the robot rendering,
  CLIP image embedding, UMT5 prompt embedding) are read from the
  content-addressed :mod:`latent_store`. Their keys depend only on the encoder
  inputs (video, render, prompt) and the encoder/geometry spec, never on action
  statistics or sampling.
* Actions are loaded from each clip's ``npz`` in the DataLoader workers on every
  access and z-scored with the per-embodiment statistics given to this run.
* :class:`MultiEmbodimentCache` is list-like (``len`` / ``[i]``) and returns one
  per-sample dict with leading batch dims, consumed by the training loop and
  the validation pass.

Frame contract: every stream of a clip (video, render, actions) has exactly 17
entries sampled on the same source frame indices. Nothing here resamples in
time; any other length is a data error.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Constants / embodiment registry
# --------------------------------------------------------------------------- #
NUM_FRAMES = 17
CANVAS_HW = (240, 432)                   # training canvas (non-aspect-preserving resize)
VAE_T = 4
LATENT_SHAPE = (16, (NUM_FRAMES - 1) // VAE_T + 1, CANVAS_HW[0] // 8, CANVAS_HW[1] // 8)  # (16,5,30,54)
TEXT_CTX = 512                           # Wan I2V splits context at shape[1]-512 (diffusers)


def _smplx_human_dof_mask() -> tuple:
    """159-D SMPL-X action of the OakInk2 human: [global orient 3 | pelvis xyz 3 | 21 body joints x3 |
    15 left-hand joints x3 | 15 right-hand joints x3] (axis-angle, camera frame). 42 slots are
    structurally constant in OakInk2's fits (std < 1e-3 rad over the training frames) and carry 0:
    the unobserved lower body (hips, knees, ankles, feet: 8 joints) and the finger PIP/DIP hinge axes
    (y of joints 2-3 of index/middle/pinky/ring, plus x of index 3, per hand)."""
    m = [1] * 159
    for j in (0, 1, 3, 4, 6, 7, 9, 10):                     # L/R hip, knee, ankle, foot among the 21 body joints
        m[6 + 3 * j: 6 + 3 * j + 3] = [0, 0, 0]
    for base in (69, 114):                                  # left hand, right hand (MANO order: index, middle, pinky, ring, thumb)
        for f in (0, 3, 6, 9):
            for k in (1, 2):
                m[base + 3 * (f + k) + 1] = 0               # PIP / DIP y
        m[base + 3 * 2] = 0                                 # index DIP x
    return tuple(m)


#: Embodiment registry: integer id, source dataset, native frame rate of the
#: source video (Hz) and the action dof mask (its length is the embodiment's
#: action width D; 0 = a slot that is always zero). DROID video content is
#: 15 Hz even where the mp4 container reports 60 fps. ``root_rotvec`` names the
#: axis-angle root-rotation slots that :func:`load_actions` unwraps per clip.
#: Per-slot layouts are documented in ``doppel/assets/embodiments.json``.
EMBODIMENTS: Dict[str, Dict[str, Any]] = {
    "panda":     dict(id=0, dataset="droid",    fps=15.0, dof_mask=(1, 1, 1, 1, 1, 1, 1, 1)),
    "ur5e_2f85": dict(id=1, dataset="robomind", fps=10.0, dof_mask=(1, 1, 1, 1, 1, 1, 0, 1)),
    "widowx":    dict(id=2, dataset="bridge",   fps=5.0,  dof_mask=(1, 1, 1, 1, 1, 1, 0, 1)),
    # OakInk2: SMPL-X, axis-angle in the camera frame.
    "smplx_human": dict(id=3, dataset="oakink2", fps=30.0, dof_mask=_smplx_human_dof_mask(), root_rotvec=(0, 3)),
    # ABC: bimanual YAM, [left arm 6 | left gripper | right arm 6 | right gripper].
    "yam_bimanual": dict(id=4, dataset="abc", fps=30.0, dof_mask=(1,) * 14),
    # GigaHands: per hand, camera-frame wrist rotation (first two columns), wrist xyz, 15 joint rotvecs.
    "mano_hands": dict(id=5, dataset="gigahands", fps=30.0, dof_mask=(1,) * 108),
    # Ego-Exo4D: SMPL-H in the OakInk2 159-D layout; every slot is live.
    "smplh_human": dict(id=6, dataset="egoexo4d", fps=30.0, dof_mask=(1,) * 159, root_rotvec=(0, 3)),
    # T-Rex: Vega-1 + two Sharpa hands, [L arm 7 | L hand 22 | R arm 7 | R hand 22] joint positions (rad).
    "vega1_sharpa": dict(id=7, dataset="trex", fps=30.0, dof_mask=(1,) * 58),
}
DATASET_IDS = {"droid": 0, "robomind": 1, "bridge": 2, "oakink2": 3, "abc": 4, "gigahands": 5, "egoexo4d": 6, "trex": 7}
#: Widest action in the registry: the shared action encoder of the 8-embodiment model takes this many slots.
MAX_ACTION_DIM = max(len(e["dof_mask"]) for e in EMBODIMENTS.values())


def embodiment(robot_type: str) -> Dict[str, Any]:
    rt = str(robot_type or "").strip()
    if rt not in EMBODIMENTS:
        raise KeyError(f"unknown robot_type {rt!r}; registry: {sorted(EMBODIMENTS)}")
    return EMBODIMENTS[rt]


def action_dim(robot_type: str) -> int:
    """Width D of an embodiment's per-frame action vector (= len(dof_mask))."""
    return len(embodiment(robot_type)["dof_mask"])


def pad_actions(actions, dof_mask, width: int):
    """Zero-extend ``actions (..., D)`` and ``dof_mask (D,)`` to ``width`` slots (numpy or torch).

    The mixed-embodiment contract: an embodiment's own D slots come first (z-scored, its pad slots 0),
    slots D..width-1 are exactly 0 and their mask is 0. Training and inference must pad the same way;
    a checkpoint trained with ``action_pad_dim`` records it in its meta.json."""
    D = int(actions.shape[-1])
    if int(dof_mask.shape[-1]) != D:
        raise ValueError(f"actions width {D} != dof_mask width {int(dof_mask.shape[-1])}")
    if D > int(width):
        raise ValueError(f"action width {D} exceeds action_pad_dim {width}")
    if D == int(width):
        return actions, dof_mask
    if isinstance(actions, torch.Tensor):
        a = actions.new_zeros(*actions.shape[:-1], int(width))
        m = dof_mask.new_zeros(*dof_mask.shape[:-1], int(width))
    else:
        a = np.zeros(actions.shape[:-1] + (int(width),), dtype=actions.dtype)
        m = np.zeros(dof_mask.shape[:-1] + (int(width),), dtype=dof_mask.dtype)
    a[..., :D] = actions
    m[..., :D] = dof_mask
    return a, m


def row_key(row: Mapping[str, Any]) -> str:
    """Stable identity of an index row (independent of row order and split)."""
    s = json.dumps([str(row.get("dataset", "")), str(row.get("video", ""))])
    return hashlib.blake2b(s.encode(), digest_size=10).hexdigest()


# --------------------------------------------------------------------------- #
# Action stats + actions
# --------------------------------------------------------------------------- #
#: Directory of this package; the shipped statistics are in ``<PACKAGE_DIR>/assets/action_stats``.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")


def resolve_stats_path(p: str, base_path: str = "") -> str:
    """A stats path as given in a config: absolute; else relative to ``base_path``; else, if that
    file does not exist, relative to the package (``assets/action_stats/<robot_type>.json`` = the
    statistics shipped with the released models, ``doppel/assets/action_stats``)."""
    p = str(p)
    if os.path.isabs(p):
        return p
    cand = os.path.join(base_path, p)
    if os.path.isfile(cand):
        return cand
    alt = os.path.join(PACKAGE_DIR, p)
    return alt if os.path.isfile(alt) else cand


def load_stats_map(stats: Mapping[str, str], base_path: str = "") -> Dict[str, Dict[str, Any]]:
    """``{robot_type: json path}`` -> ``{robot_type: {mean, std, dof_mask, sha256, path}}``.

    Relative paths are resolved by :func:`resolve_stats_path`. The dof_mask of each
    stats file must match the registry's mask for that embodiment."""
    out: Dict[str, Dict[str, Any]] = {}
    for rt, p in stats.items():
        path = resolve_stats_path(p, base_path)
        raw = open(path, "rb").read()
        blob = json.loads(raw)
        mean = np.asarray(blob["mean"], dtype=np.float32)
        std = np.asarray(blob["std"], dtype=np.float32)
        mask = np.asarray(blob.get("dof_mask", [1] * len(mean)), dtype=np.float32)
        reg = np.asarray(embodiment(rt)["dof_mask"], dtype=np.float32)
        if mean.shape != reg.shape or std.shape != reg.shape or not np.array_equal(mask, reg):
            raise ValueError(f"{path}: stats for {rt!r} must be {reg.shape[0]}-d with dof_mask {reg.tolist()}, "
                             f"got mean{mean.shape} std{std.shape} mask {mask.tolist()}")
        if not (np.all(np.isfinite(mean)) and np.all(std > 1e-6)):
            raise ValueError(f"{path}: non-finite mean or non-positive std")
        if np.any(mean[mask == 0] != 0) or np.any(std[mask == 0] != 1):
            raise ValueError(f"{path}: masked slots must be mean 0 / std 1")
        out[str(rt)] = dict(mean=mean, std=std, dof_mask=mask, path=path,
                            sha256=hashlib.sha256(raw).hexdigest())
    return out


def unwrap_rotvec(seq) -> np.ndarray:
    """Make a per-frame axis-angle sequence ``(T, 3)`` continuous without changing any rotation.

    ``theta * u`` and ``u * (theta + 2*pi*k)`` are the same rotation; k = -1 is the antipodal form an
    exporter switches to when theta crosses pi. Frame 0 is kept; every later frame takes the
    representative with k in {-1, 0, 1} closest to the already-unwrapped previous frame. A sequence
    without a flip is returned unchanged (the same object)."""
    a = np.asarray(seq)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"unwrap_rotvec expects (T, 3), got {a.shape}")
    r = a.astype(np.float64)
    out = r.copy()
    changed = False
    for t in range(1, r.shape[0]):
        prev = out[t - 1]
        theta = float(np.linalg.norm(r[t]))
        if theta > 1e-12:
            u = r[t] / theta
        else:
            pn = float(np.linalg.norm(prev))
            if pn <= 1e-12:
                continue
            u = prev / pn
        cands = [u * (theta + 2.0 * np.pi * k) for k in (-1, 0, 1)]
        dist = [float(np.linalg.norm(c - prev)) for c in cands]
        best = int(np.argmin(dist))
        if best != 1 and dist[best] < dist[1]:
            out[t] = cands[best]
            changed = True
    if not changed:
        return a
    return out.astype(a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64)


def unwrap_root(actions: np.ndarray, robot_type: str) -> np.ndarray:
    """Apply :func:`unwrap_rotvec` to the embodiment's ``root_rotvec`` slots of a RAW ``(T, D)`` action."""
    sl = embodiment(robot_type).get("root_rotvec")
    if sl is None:
        return actions
    lo, hi = sl
    view = actions[:, lo:hi]
    fixed = unwrap_rotvec(view)
    if fixed is view:
        return actions
    out = np.array(actions, copy=True)
    out[:, lo:hi] = fixed
    return out


def normalize_actions(a: np.ndarray, robot_type: str, stats_map: Mapping[str, Mapping[str, Any]],
                      root_unwrap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """RAW ``(T, D)`` actions -> ``(z-scored actions (T, D) f32, dof_mask (D,) f32)``.

    Root axis-angle slots are unwrapped per clip first (``root_unwrap``; the SMPL-X/H statistics
    were computed the same way), then the per-embodiment z-score, then pad slots (dof_mask == 0)
    are set to exactly 0. Shared by training (:func:`load_actions`) and inference."""
    if robot_type not in stats_map:
        raise KeyError(f"no action stats for robot_type {robot_type!r}; have {sorted(stats_map)}")
    st = stats_map[robot_type]
    a = np.asarray(a, dtype=np.float32)
    D = action_dim(robot_type)
    if a.ndim != 2 or a.shape[1] != D:
        raise ValueError(f"{robot_type} actions must be (T, {D}), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite actions")
    mask = np.asarray(st["dof_mask"], dtype=np.float32)
    if np.any(a[:, mask == 0] != 0):
        raise ValueError(f"pad slot(s) {np.where(mask == 0)[0].tolist()} of {robot_type} are not zero")
    if root_unwrap:
        a = unwrap_root(a, robot_type)
    out = (a - st["mean"]) / st["std"]
    out[:, mask == 0] = 0.0
    return out.astype(np.float32), mask


def load_actions(path: str, robot_type: str, stats_map: Mapping[str, Mapping[str, Any]],
                 field: str = "state", root_unwrap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load one clip's actions -> ``(actions (17, D) f32 z-scored, dof_mask (D,) f32)``, D = action_dim(robot_type).

    * ``robot_type`` comes from the index row (required); a ``robot_type`` field
      stored in the npz, if present, must agree.
    * No temporal resampling: actions are frame-aligned with the clip, exactly 17 rows.
    * Root axis-angle unwrap (SMPL-X/H), per-embodiment z-score, then pad slots
      (dof_mask == 0) are set to exactly 0 (:func:`normalize_actions`).
    """
    embodiment(robot_type)
    if robot_type not in stats_map:
        raise KeyError(f"no action stats for robot_type {robot_type!r}; have {sorted(stats_map)}")
    st = stats_map[robot_type]
    z = np.load(path)
    if field not in z.files:
        raise KeyError(f"{path}: missing {field!r}; have {z.files}")
    a = np.asarray(z[field], dtype=np.float32)
    D = action_dim(robot_type)
    if a.shape != (NUM_FRAMES, D):
        raise ValueError(f"{path}: {field} shape {a.shape} != ({NUM_FRAMES}, {D}) (frame-aligned clip)")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{path}: non-finite {field}")
    if "robot_type" in z.files:
        rt_npz = str(np.asarray(z["robot_type"]).reshape(-1)[0])
        if rt_npz != robot_type:
            raise ValueError(f"{path}: npz robot_type {rt_npz!r} != index {robot_type!r}")
    mask = np.asarray(st["dof_mask"], dtype=np.float32)
    if "dof_mask" in z.files and not np.array_equal(np.asarray(z["dof_mask"], np.float32), mask):
        raise ValueError(f"{path}: npz dof_mask {z['dof_mask'].tolist()} != {mask.tolist()} ({robot_type})")
    if np.any(a[:, mask == 0] != 0):
        raise ValueError(f"{path}: pad slot(s) {np.where(mask == 0)[0].tolist()} are not zero-padded")
    return normalize_actions(a, robot_type, stats_map, root_unwrap=root_unwrap)


# --------------------------------------------------------------------------- #
# Wan I2V condition mask (as built by the diffusers Wan I2V pipeline)
# --------------------------------------------------------------------------- #
def i2v_condition_mask(num_frames: int = NUM_FRAMES, latent_h: int = LATENT_SHAPE[2],
                       latent_w: int = LATENT_SHAPE[3], vae_t: int = VAE_T) -> torch.Tensor:
    m = torch.ones(1, 1, num_frames, latent_h, latent_w)
    m[:, :, 1:] = 0
    first = m[:, :, 0:1].repeat_interleave(vae_t, dim=2)
    m = torch.cat([first, m[:, :, 1:]], dim=2)
    return m.view(1, -1, vae_t, latent_h, latent_w).transpose(1, 2)   # (1, 4, T_lat, h, w)


# --------------------------------------------------------------------------- #
# Store-backed dataset
# --------------------------------------------------------------------------- #
class MultiEmbodimentCache(torch.utils.data.Dataset):
    """List-like dataset over index rows, backed by the latent store.

    ``cache[i]`` returns::

        clean_latents  (1,16,5,30,54)  store dtype (bf16)   VAE posterior mean, Wan-normalised
        render_latents (1,16,5,30,54)  (render-conditioned stores only)
        condition      (1,20,5,30,54)  = [I2V mask (4ch), first-frame latent (16ch)]
        image_embeds   (1,257,1280)    CLIP penultimate layer, frame 0
        prompt_embeds  (1,512,4096)    UMT5, zero-padded to 512 (Wan context split)
        actions        (17,D) f32      per-embodiment z-score, pad slots 0 (D = action_pad_dim
                                        if set, else the index's single action width)

    plus metadata: ``dof_mask (D,)``, ``embodiment_id``, ``dataset_id`` (used
    for per-dataset validation losses) and ``row_key``.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], base_path: str, store: Any,
                 stats_map: Optional[Mapping[str, Mapping[str, Any]]], *,
                 with_render: bool, ignore_prompts: bool = False, action_field: str = "state",
                 action_pad_dim: Optional[int] = None):
        self.rows = [dict(r) for r in rows]
        self.base_path = os.path.abspath(base_path)
        self.store = store
        self.stats_map = stats_map
        self.with_render = bool(with_render)
        self.ignore_prompts = bool(ignore_prompts)
        self.action_field = action_field
        self._mask = i2v_condition_mask()
        for r in self.rows:
            embodiment(r["robot_type"])            # fail fast on unknown embodiments
            r.setdefault("row_key", row_key(r))
        widths = sorted({action_dim(r["robot_type"]) for r in self.rows}) if stats_map is not None else []
        self.action_pad_dim = int(action_pad_dim) if action_pad_dim else None
        if self.action_pad_dim is not None:        # every sample is zero-extended to this width
            if widths and widths[-1] > self.action_pad_dim:
                raise ValueError(f"index has action widths {widths}: action_pad_dim {self.action_pad_dim} "
                                 f"must be >= {widths[-1]}")
        elif len(widths) > 1:                      # actions are stacked into a batch: one width per run
            raise ValueError(f"index mixes action widths {widths}; set action_pad_dim (>= {widths[-1]}) "
                             "to zero-pad them into one encoder, or index one width per run")

    def __len__(self) -> int:
        return len(self.rows)

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.base_path, p)

    def __getitem__(self, i) -> Dict[str, Any]:
        r = self.rows[int(i)]
        emb = embodiment(r["robot_type"])
        enc = self.store.get(r, with_render=self.with_render)       # raises if missing
        prompt = "" if self.ignore_prompts else str(r.get("prompt", "") or "")
        pe = self.store.get_prompt(prompt)                           # (L, 4096)
        prompt_embeds = pe.new_zeros(1, TEXT_CTX, pe.shape[-1])
        prompt_embeds[0, : pe.shape[0]] = pe
        s: Dict[str, Any] = {
            "clean_latents": enc["clean_latents"].unsqueeze(0),
            "image_embeds": enc["image_embeds"].unsqueeze(0),
            "prompt_embeds": prompt_embeds,
            "condition": torch.cat([self._mask.to(enc["cond_latents"].dtype),
                                    enc["cond_latents"].unsqueeze(0)], dim=1),
            "embodiment_id": torch.tensor(emb["id"], dtype=torch.int64),
            "dataset_id": torch.tensor(DATASET_IDS[emb["dataset"]], dtype=torch.int64),
            "row_key": r["row_key"],
        }
        if self.with_render:
            s["render_latents"] = enc["render_latents"].unsqueeze(0)
        if self.stats_map is not None:
            a, m = load_actions(self._abs(r["actions"]), r["robot_type"], self.stats_map,
                                field=self.action_field)
            if self.action_pad_dim is not None:
                a, m = pad_actions(a, m, self.action_pad_dim)
            s["actions"] = torch.from_numpy(a)
            s["dof_mask"] = torch.from_numpy(m)
        return s


class IndexView:
    """List-like reordering/subset of a cache (e.g. dataset-interleaved val)."""

    def __init__(self, base: Any, order: Sequence[int]):
        self.base, self.order = base, list(order)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.base[self.order[int(i)]]


def stale_rows(rows: Sequence[Mapping[str, Any]], base_path: str, threads: int = 16,
               missing_ok: bool = False) -> List[str]:
    """Rows whose video/render changed (size or mtime) since the index
    (scripts/data/build_index.py) hashed them, i.e. whose store keys would be
    wrong. Must be empty to train.

    ``missing_ok`` skips files that are absent entirely, for corpora that carry
    the latent store but not the source media. A file that is present and has
    changed is still reported."""
    from concurrent.futures import ThreadPoolExecutor

    def _chk(r):
        bad = []
        for col in ("video", "render"):
            if not r.get(f"{col}_size"):
                continue
            p = r[col] if os.path.isabs(r[col]) else os.path.join(base_path, r[col])
            try:
                st = os.stat(p)
            except OSError:
                if not missing_ok:
                    bad.append(f"{col} missing: {p}")
                continue
            if st.st_size != int(r[f"{col}_size"]) or int(st.st_mtime) != int(r[f"{col}_mtime"]):
                bad.append(f"{col} changed since indexing: {p}")
        return bad

    with ThreadPoolExecutor(threads) as ex:
        return [b for bb in ex.map(_chk, rows, chunksize=64) for b in bb]


def list_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """DataLoader collate that keeps the per-sample dicts (the loop stacks them)."""
    return batch


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def row_weights(rows: Sequence[Mapping[str, Any]],
                dataset_weights: Optional[Mapping[str, float]] = None) -> np.ndarray:
    """Per-row sampling probability (sums to 1): uniform within each dataset,
    ``p(row) = p(dataset) / n(dataset)``, with ``p(dataset)`` from
    ``dataset_weights`` (default: proportional to the row counts).
    """
    ds = np.array([embodiment(r["robot_type"])["dataset"] for r in rows])
    w = np.zeros(len(rows), np.float64)
    names = sorted(set(ds))
    n_ds = {d: int((ds == d).sum()) for d in names}
    pd_ = ({d: float(dataset_weights.get(d, 0.0)) for d in names} if dataset_weights
           else {d: float(n_ds[d]) for d in names})
    tot = sum(pd_.values())
    for d in names:
        w[ds == d] = (pd_[d] / tot) / n_ds[d]
    return w / w.sum()


def epoch_batches(n_rows: int, epoch: int, seed: int, world_size: int, rank: int, batch_size: int,
                  weights: Optional[np.ndarray] = None, epoch_len: Optional[int] = None) -> List[List[int]]:
    """This rank's micro-batches for one epoch; identical on every rank without
    communication. ``weights=None`` gives a seeded permutation of all rows;
    otherwise ``epoch_len`` (default ``n_rows``) rows are drawn with
    replacement from ``weights``. The plan is padded to a multiple of
    ``world_size * batch_size`` so every rank runs the same number of steps."""
    g = torch.Generator()
    g.manual_seed(int(seed) + int(epoch))
    if weights is None:
        full = torch.randperm(n_rows, generator=g).tolist()
    else:
        wt = torch.as_tensor(np.asarray(weights, np.float64))
        full = torch.multinomial(wt, int(epoch_len or n_rows), replacement=True, generator=g).tolist()
    bs = max(1, int(batch_size))
    multiple = world_size * bs
    pad = (-len(full)) % multiple
    if pad:
        n_have = len(full)
        full = full + [full[i % n_have] for i in range(pad)]
    mine = full[rank::world_size]
    return [mine[i:i + bs] for i in range(0, len(mine), bs)]


class EpochBatchSampler:
    """Stateful batch sampler for a single persistent-worker DataLoader.

    With ``persistent_workers=True`` the DataLoader forks its workers once and
    calls ``iter(batch_sampler)`` again on every ``iter(loader)``; the trainer
    hands each epoch's micro-batch plan (from :func:`epoch_batches`) to this
    object before iterating.
    """

    def __init__(self) -> None:
        self.epoch = 0
        self.batches: List[List[int]] = []

    def set_epoch(self, epoch: int, batches: Sequence[Sequence[int]]) -> None:
        self.epoch = int(epoch)
        self.batches = [[int(j) for j in b] for b in batches]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def interleave_by_dataset(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    """Deterministic val order: each dataset sorted by a hash of its row key, then
    round-robin, so any prefix (``val_n_samples``) is dataset-balanced."""
    by: Dict[str, List[Tuple[str, int]]] = {}
    for i, r in enumerate(rows):
        d = embodiment(r["robot_type"])["dataset"]
        h = hashlib.blake2b(str(r.get("row_key") or row_key(r)).encode(), digest_size=8).hexdigest()
        by.setdefault(d, []).append((h, i))
    lists = [[i for _, i in sorted(v)] for _, v in sorted(by.items())]
    out: List[int] = []
    for k in range(max((len(l) for l in lists), default=0)):
        out.extend(l[k] for l in lists if k < len(l))
    return out
