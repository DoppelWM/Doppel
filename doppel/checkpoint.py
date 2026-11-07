"""
Loading trained Doppel weights and the model configuration that goes with them.

A training save (``epoch_N/``) holds:

* ``conditioner.pt``: the trainable subset of the DiT (adapter + the last
  ``unfreeze_last_n_blocks`` Wan blocks + ``proj_out`` + ``condition_embedder.time_proj``),
  EMA weights when ``ema_eval`` is on; ``conditioner_raw.pt`` holds the raw weights;
* ``meta.json``: model configuration, action statistics and training state.

Everything else (VAE, text and image encoders, frozen blocks) comes from the Wan 2.1
I2V 14B 480P Diffusers checkpoint.

A released model (e.g. ``doppel-emb8``) is instead a directory holding ``config.json``
and ``transformer_overlay/`` (sharded ``model-*.safetensors`` + ``model.safetensors.index.json``,
EMA weights, each tensor stored in the dtype it has at inference). :func:`load_conditioner`
accepts that directory, the overlay directory, its index JSON or a single ``.safetensors``
file, and :func:`model_config` accepts the release ``config.json``.

Checkpoints written by the research code base used different names; both are accepted:

=============================  ==========================
research code                  this package
=============================  ==========================
``action_adapter_v4.*``        ``render_action_adapter.*``
``legacy_render_variant: v4``  ``conditioning: redar``
``v4_adapter_kwargs``          ``redar_adapter_kwargs``
``render_dropout_prob``        ``cond_dropout_prob``
``render_conditioner.pt``      ``conditioner.pt``
=============================  ==========================
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Tuple

import torch

#: State-dict prefixes of research-code checkpoints and their names here.
LEGACY_PREFIXES = {"action_adapter_v4.": "render_action_adapter."}
#: Prefixes of modules that exist only in research-code checkpoints (auxiliary heads
#: of ablations, never used by the released models); dropped on load.
DROPPED_PREFIXES = ("tracks_head.",)
ADAPTER_PREFIXES = ("render_action_adapter.", "action_adapter_only.")


def remap_state_dict(sd: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Rename research-code keys to this package's names; drop unused auxiliary heads."""
    out: Dict[str, Any] = {}
    n_renamed = n_dropped = 0
    for k, v in sd.items():
        if k.startswith(DROPPED_PREFIXES):
            n_dropped += 1
            continue
        for old, new in LEGACY_PREFIXES.items():
            if k.startswith(old):
                k = new + k[len(old):]
                n_renamed += 1
                break
        out[k] = v
    return out, {"renamed": n_renamed, "dropped": n_dropped}


def _safetensors_files(path: str):
    """The shard files of a safetensors checkpoint given as a release directory (holding
    ``transformer_overlay/``), an overlay directory, its ``*.index.json`` or one
    ``.safetensors`` file; ``None`` if ``path`` is none of these."""
    p = str(path)
    if os.path.isdir(p) and os.path.isdir(os.path.join(p, "transformer_overlay")):
        p = os.path.join(p, "transformer_overlay")
    if os.path.isdir(p):
        idx = sorted(f for f in os.listdir(p) if f.endswith(".safetensors.index.json"))
        if idx:
            p = os.path.join(p, idx[0])
        else:
            files = sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".safetensors"))
            return files or None
    if p.endswith(".index.json"):
        with open(p, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        d = os.path.dirname(os.path.abspath(p))
        return [os.path.join(d, f) for f in sorted(set(weight_map.values()))]
    if p.endswith(".safetensors"):
        return [p]
    return None


def load_conditioner(model: torch.nn.Module, path: str) -> Dict[str, Any]:
    """Load trained weights into a conditioned transformer (in place).

    ``path`` is a training save's ``conditioner.pt`` (memory-mapped; the 40-block
    checkpoints are ~65 GB in fp32) or a released safetensors overlay (see
    :func:`_safetensors_files`; loaded one shard at a time). ``load_state_dict``
    casts each tensor to the parameter dtype during the copy. Raises if any
    checkpoint key is unknown to the model, any adapter parameter is missing from
    the checkpoint, or a block is only partially present."""
    files = _safetensors_files(path)
    if files is None:
        shards = [torch.load(path, map_location="cpu", weights_only=False, mmap=True)]
    else:
        from safetensors.torch import load_file
        shards = (load_file(f, device="cpu") for f in files)
    loaded, unexpected = set(), []
    info = {"renamed": 0, "dropped": 0}
    for raw in shards:
        sd, shard_info = remap_state_dict(raw)
        del raw
        for k in info:
            info[k] += shard_info[k]
        _missing, unexp = model.load_state_dict(sd, strict=False)
        unexpected += list(unexp)
        loaded.update(sd)
        del sd
    if unexpected:
        raise RuntimeError(f"{path}: {len(unexpected)} keys the model does not have, e.g. {unexpected[:5]}")
    missing = [k for k in model.state_dict() if k not in loaded]
    missing_adapter = [k for k in missing if k.startswith(ADAPTER_PREFIXES)]
    if missing_adapter:
        raise RuntimeError(f"{path}: adapter parameters missing from the checkpoint, e.g. {missing_adapter[:5]}")
    # A block the checkpoint trained must be complete.
    trained_blocks = {".".join(k.split(".")[:2]) + "." for k in loaded if k.startswith("blocks.")}
    partial = [k for k in missing if k.startswith(tuple(trained_blocks))]
    if partial:
        raise RuntimeError(f"{path}: partially saved blocks, e.g. {partial[:5]}")
    return {"path": path, "n_tensors": len(loaded), "n_trained_blocks": len(trained_blocks), **info}


_LEGACY_VARIANTS = {"v4": "redar", "action_only": "action_only"}


def model_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The inference-relevant model configuration from a training config JSON, a
    checkpoint's ``meta.json`` (this package's or the research code's key names) or a
    released model's ``config.json``."""
    cfg = dict(cfg)
    if "architecture" in cfg and "inputs" in cfg:      # release config.json (e.g. doppel-emb8)
        arch, inp = cfg["architecture"], cfg["inputs"]
        cfg = {"legacy_render_variant": arch.get("legacy_render_variant"),
               "v4_adapter_kwargs": arch.get("v4_adapter_kwargs"),
               "action_pad_dim": inp.get("action_pad_dim", 0),
               "action_field": inp.get("action_field", "state"),
               "action_stats_path": {rt: v.get("file") for rt, v in (cfg.get("action_stats") or {}).items()},
               "num_frames": inp.get("num_frames", 17), "height": inp.get("height", 240),
               "width": inp.get("width", 432)}
    cond = cfg.get("conditioning")
    if cond is None and "legacy_render_variant" in cfg:
        v = str(cfg["legacy_render_variant"])
        if v not in _LEGACY_VARIANTS:
            raise ValueError(f"unsupported legacy_render_variant {v!r} (released models are 'v4' and 'action_only')")
        cond = _LEGACY_VARIANTS[v]
    if cond not in ("redar", "action_only"):
        raise ValueError(f"config has no valid 'conditioning' (got {cond!r})")
    redar_kw = cfg.get("redar_adapter_kwargs") or cfg.get("v4_adapter_kwargs")
    if redar_kw is not None:
        redar_kw = dict(redar_kw)
        if redar_kw.get("use_actions") is False:
            raise ValueError("render-only checkpoints (use_actions=False) are not part of this release")
        redar_kw.pop("use_actions", None)
    ao_kw = cfg.get("action_only_adapter_kwargs")
    if ao_kw is not None:
        ao_kw = dict(ao_kw)
        if ao_kw.pop("mode", "egowm") != "egowm":     # research code: the released baseline is mode "egowm"
            raise ValueError("only the EgoWM-style action-only adapter is part of this release")
    pad = int(cfg.get("action_pad_dim") or 0)
    ad = (redar_kw or ao_kw or {}).get("action_dim")
    if pad and ad is not None and int(ad) != pad:
        raise ValueError(f"action_pad_dim {pad} != adapter action_dim {ad}")
    return {
        "conditioning": cond,
        "redar_adapter_kwargs": redar_kw if cond == "redar" else None,
        "action_only_adapter_kwargs": ao_kw if cond == "action_only" else None,
        "action_pad_dim": pad,
        "action_field": cfg.get("action_field", "state"),
        "robot_types": sorted((cfg.get("action_stats_path") or cfg.get("action_stats") or {}).keys()),
        "num_frames": int(cfg.get("num_frames", 17)),
        "height": int(cfg.get("height", 240)),
        "width": int(cfg.get("width", 432)),
    }


def load_model_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return model_config(json.load(f))
