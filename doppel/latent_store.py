"""
Content-addressed, sharded store of frozen-encoder outputs.

Layout under ``<root>/<spec_hash>/``::

    SPEC.json                              encoder + geometry spec (what the hash covers)
    clips/<ab>/<vkey>.safetensors          clean_latents (16,5,30,54), cond_latents (16,5,30,54),
                                           image_embeds (257,1280)            vkey = H(spec, sha(video))
    renders/<ab>/<rkey>.safetensors        render_latents (16,5,30,54)        rkey = H(spec, sha(render))
    prompts/<ab>/<pkey>.safetensors        embeds (L,4096), L = real tokens    pkey = H(spec, text)

Keys hash the content of the inputs (blake2b of the mp4 bytes, recorded in the
index by scripts/data/build_index.py), never paths or mtimes, so a re-rendered or re-exported clip gets
a new key and nothing can be served stale. Action statistics, sampling and
splits are not inputs: they are applied at load time (see :mod:`data_multi`).

Writes are atomic (tmp + ``os.replace``) and idempotent, so any number of
independent workers (``--shard_index/--shard_count``) can build the store and
an interrupted job resumes by skipping files that exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

STORE_VERSION = 1


def file_sha(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def make_spec(model_id: str, height: int = 240, width: int = 432, num_frames: int = 17,
              dtype: str = "bfloat16", encoder_impl: str = "wan_i2v_v1") -> Dict[str, Any]:
    return {
        "store_version": STORE_VERSION,
        "model_id": str(model_id),          # e.g. "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers@b184e23a"
        "height": int(height), "width": int(width), "num_frames": int(num_frames),
        "dtype": str(dtype),
        "encoder_impl": encoder_impl,       # bump when the encode recipe below changes
        "latent": "vae.latent_dist.mode(), Wan latents_mean/std normalised",
        "resize": "diffusers.VideoProcessor.preprocess_video (PIL, non-aspect-preserving)",
        "decode": "imageio-ffmpeg rgb24, all frames, exactly num_frames",
        "clip": "CLIPVisionModel hidden_states[-2] of frame 0",
        "cond": "vae.encode([frame0, zeros*(T-1)]).mode(), normalised (mask built at load)",
        "text": "UMT5 last_hidden_state, attention-masked, trimmed to real tokens (<=512)",
    }


def spec_hash(spec: Mapping[str, Any]) -> str:
    return hashlib.blake2b(json.dumps(spec, sort_keys=True).encode(), digest_size=8).hexdigest()


def _k(*parts: str) -> str:
    return hashlib.blake2b("\x1f".join(parts).encode(), digest_size=16).hexdigest()


class LatentStore:
    def __init__(self, root: str, spec: Mapping[str, Any], create: bool = False):
        self.spec = dict(spec)
        self.sh = spec_hash(self.spec)
        self.dir = os.path.join(root, self.sh)
        sp = os.path.join(self.dir, "SPEC.json")
        if create:
            os.makedirs(self.dir, exist_ok=True)
            if not os.path.exists(sp):
                tmp = f"{sp}.tmp{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump(self.spec, f, indent=1, sort_keys=True)
                os.replace(tmp, sp)
        if not os.path.isfile(sp):
            raise FileNotFoundError(f"no latent store for this spec at {self.dir} (build it first)")
        with open(sp) as f:
            on_disk = json.load(f)
        if on_disk != self.spec:
            raise ValueError(f"{sp} differs from the requested spec")
        self._prompt_cache = lru_cache(maxsize=4096)(self._load_prompt)

    # ---- keys / paths ---------------------------------------------------
    def clip_key(self, video_hash: str) -> str:
        return _k(self.sh, "clip", video_hash)

    def render_key(self, render_hash: str) -> str:
        return _k(self.sh, "render", render_hash)

    def prompt_key(self, text: str) -> str:
        return _k(self.sh, "prompt", text)

    def _path(self, kind: str, key: str) -> str:
        return os.path.join(self.dir, kind, key[:2], f"{key}.safetensors")

    # ---- write ----------------------------------------------------------
    def put(self, kind: str, key: str, tensors: Dict[str, torch.Tensor], meta: Mapping[str, Any]) -> str:
        p = self._path(kind, key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp.{socket.gethostname()}.{os.getpid()}"     # unique across nodes; same-key races are benign
        save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()}, tmp,
                  metadata={k: json.dumps(v) if not isinstance(v, str) else v for k, v in meta.items()})
        os.replace(tmp, p)
        return p

    def has(self, kind: str, key: str) -> bool:
        return os.path.isfile(self._path(kind, key))

    # ---- read -----------------------------------------------------------
    def _read(self, kind: str, key: str, what: str) -> Dict[str, torch.Tensor]:
        p = self._path(kind, key)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"latent store miss ({kind}) for {what}: {p} -- the input "
                                    f"changed or was never encoded; rebuild the store for this index")
        out = {}
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out

    def get(self, row: Mapping[str, Any], with_render: bool) -> Dict[str, torch.Tensor]:
        out = self._read("clips", self.clip_key(row["video_hash"]), str(row.get("video")))
        if with_render:
            out.update(self._read("renders", self.render_key(row["render_hash"]), str(row.get("render"))))
        return out

    def _load_prompt(self, text: str) -> torch.Tensor:
        return self._read("prompts", self.prompt_key(text), repr(text[:60]))["embeds"]

    def get_prompt(self, text: str) -> torch.Tensor:
        return self._prompt_cache(text)

    def missing(self, rows: Sequence[Mapping[str, Any]], with_render: bool,
                ignore_prompts: bool) -> Dict[str, List[int]]:
        miss: Dict[str, List[int]] = {"clips": [], "renders": [], "prompts": []}
        for i, r in enumerate(rows):
            if not self.has("clips", self.clip_key(r["video_hash"])):
                miss["clips"].append(i)
            if with_render and not self.has("renders", self.render_key(r["render_hash"])):
                miss["renders"].append(i)
            t = "" if ignore_prompts else str(r.get("prompt", "") or "")
            if not self.has("prompts", self.prompt_key(t)):
                miss["prompts"].append(i)
        return miss


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
class WanStoreEncoders:
    """The frozen Wan I2V encoders: VAE (video, render and first-frame
    condition latents), CLIP image encoder and UMT5 text encoder. Target latents
    are the VAE posterior mean (``.mode()``), so encoding is deterministic.
    Encoders are loaded lazily and one at a time."""

    def __init__(self, model_root: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 height: int = 240, width: int = 432, num_frames: int = 17, max_seq_len: int = 512):
        from diffusers.video_processor import VideoProcessor
        self.root, self.device, self.dtype = model_root, device, dtype
        self.h, self.w, self.T, self.L = height, width, num_frames, max_seq_len
        self.vp = VideoProcessor(vae_scale_factor=8)
        self._vae = self._clip = self._text = None

    def release(self, which: str) -> None:
        setattr(self, f"_{which}", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_vae(self):
        if self._vae is None:
            from doppel.model import WanVAEChunkedEncode
            vae = WanVAEChunkedEncode.from_pretrained(self.root, subfolder="vae", torch_dtype=self.dtype)
            self._vae = vae.to(self.device).eval()
            self._lm = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            self._ls = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        return self._vae

    def _norm(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self._lm.to(z)) / self._ls.to(z)

    @torch.no_grad()
    def video_latents(self, frames: List[Any]) -> torch.Tensor:
        vae = self._load_vae()
        vid = self.vp.preprocess_video(frames, height=self.h, width=self.w).to(self.device, self.dtype)
        return self._norm(vae.encode(vid).latent_dist.mode())[0].cpu()          # (16,T_lat,h,w)

    @torch.no_grad()
    def cond_latents(self, frame0: Any) -> torch.Tensor:
        vae = self._load_vae()
        first = self.vp.preprocess(frame0, height=self.h, width=self.w).unsqueeze(2).to(self.device, self.dtype)
        vc = torch.cat([first, first.new_zeros(1, first.shape[1], self.T - 1, self.h, self.w)], dim=2)
        return self._norm(vae.encode(vc).latent_dist.mode())[0].cpu()           # (16,T_lat,h,w)

    @torch.no_grad()
    def image_embeds(self, frame0: Any) -> torch.Tensor:
        if self._clip is None:
            from transformers import AutoImageProcessor, CLIPVisionModel
            self._proc = AutoImageProcessor.from_pretrained(self.root, subfolder="image_processor")
            self._clip = CLIPVisionModel.from_pretrained(self.root, subfolder="image_encoder",
                                                         torch_dtype=self.dtype).to(self.device).eval()
        pv = self._proc(images=frame0, return_tensors="pt").pixel_values.to(self.device, self.dtype)
        return self._clip(pv, output_hidden_states=True).hidden_states[-2][0].cpu()   # (257,1280)

    @torch.no_grad()
    def prompt_embeds(self, text: str) -> torch.Tensor:
        if self._text is None:
            from transformers import T5TokenizerFast, UMT5EncoderModel
            self._tok = T5TokenizerFast.from_pretrained(self.root, subfolder="tokenizer")
            self._text = UMT5EncoderModel.from_pretrained(self.root, subfolder="text_encoder",
                                                          torch_dtype=self.dtype).to(self.device).eval()
        tok = self._tok([text], padding="max_length", max_length=self.L, truncation=True,
                        add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        e = self._text(tok.input_ids.to(self.device), attention_mask=tok.attention_mask.to(self.device)).last_hidden_state
        n = int(tok.attention_mask.sum())
        return e[0, :n].cpu()                                                   # (L_real, 4096)


def encode_rows(store: LatentStore, rows: Sequence[Mapping[str, Any]], base_path: str, enc: Any, *,
                with_render: bool, ignore_prompts: bool, log=print) -> Dict[str, int]:
    """Encode whatever of ``rows`` the store lacks. Order: prompts (text encoder),
    then clips + renders (VAE + CLIP), so only one big encoder is needed at a time."""
    from doppel.data import _read_all_frames

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(base_path, p)

    n = {"prompts": 0, "clips": 0, "renders": 0, "skipped": 0}
    texts = sorted({"" if ignore_prompts else str(r.get("prompt", "") or "") for r in rows})
    for t in texts:
        k = store.prompt_key(t)
        if not store.has("prompts", k):
            store.put("prompts", k, {"embeds": enc.prompt_embeds(t)}, {"text": t})
            n["prompts"] += 1
    if hasattr(enc, "release"):
        enc.release("text")
    T = int(store.spec["num_frames"])
    for r in rows:
        vk = store.clip_key(r["video_hash"])
        need_clip = not store.has("clips", vk)
        rk = store.render_key(r["render_hash"]) if with_render else None
        need_render = bool(with_render and not store.has("renders", rk))
        if not (need_clip or need_render):
            n["skipped"] += 1
            continue
        if need_clip:
            frames = _read_all_frames(_abs(r["video"]))
            if len(frames) != T:
                raise ValueError(f"{r['video']}: decoded {len(frames)} frames != {T}")
            store.put("clips", vk, {
                "clean_latents": enc.video_latents(frames),
                "cond_latents": enc.cond_latents(frames[0]),
                "image_embeds": enc.image_embeds(frames[0]),
            }, {"video": str(r["video"]), "video_hash": r["video_hash"], "dataset": str(r.get("dataset", ""))})
            n["clips"] += 1
        if need_render:
            rf = _read_all_frames(_abs(r["render"]))
            if len(rf) != T:
                raise ValueError(f"{r['render']}: decoded {len(rf)} frames != {T}")
            store.put("renders", rk, {"render_latents": enc.video_latents(rf)},
                      {"render": str(r["render"]), "render_hash": r["render_hash"]})
            n["renders"] += 1
    return n
