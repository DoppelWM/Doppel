"""FSDP training of the conditioned Wan2.1 I2V DiT.

Two conditioning variants share this trainer (config key ``conditioning``):

* ``"redar"``: the render-action adapter
  (``RenderActionWanTransformer`` with ``RenderActionAdapter``),
  conditioned on the rendered robot video latents and the action stream.
* ``"action_only"``: the action-only baseline adapter (``ActionOnlyWanTransformer``).

The VAE, CLIP and text encoders are not loaded: their outputs are read from
the latent store (``scripts/data/build_latent_store.py``).
The objective is rectified flow (flow matching): the DiT predicts the velocity
``noise - clean_latents``; latent frame 0 is the clean I2V conditioning frame and
is excluded from the loss.

Launch on 8 GPUs::

  accelerate launch --num_processes 8 --config_file configs/accelerate_fsdp.yaml \
    -m doppel.train --config configs/droid_render_action.json

Resume after ``epoch_9`` (``resume_mode: continue`` restores the optimizer,
scheduler, EMA and RNG state exactly; it needs the same number of GPUs)::

  accelerate launch --num_processes 8 --config_file configs/accelerate_fsdp.yaml \
    -m doppel.train --config configs/droid_render_action.json \
    --resume_ckpt runs/droid_render_action/run_<id>/epoch_9/conditioner.pt \
    --epoch_offset 10 --num_epochs 980
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.utils import GradientAccumulationPlugin, set_seed
from tqdm.auto import tqdm

from doppel import resume_state as _rs
from doppel.checkpoint import _safetensors_files, remap_state_dict
from doppel.data_multi import (
    DATASET_IDS,
    EMBODIMENTS,
    EpochBatchSampler,
    IndexView,
    MultiEmbodimentCache,
    action_dim,
    epoch_batches,
    interleave_by_dataset,
    list_collate,
    load_stats_map,
    resolve_stats_path,
    row_weights,
    stale_rows,
)
from doppel.ema import (
    ShardEMA, ema_shard_path, free_unsharded_flat_params, fsdp_flat_param_handles,
)
from doppel.model_action_only import ActionOnlyWanTransformer
from doppel.model_redar import RenderActionWanTransformer
from doppel.train_utils import (
    _setup_run_dir_and_logging,
    build_lr_scheduler,
    build_param_lr_groups,
    fsdp_micro_steps_per_epoch,
    optimizer_step_count,
    resolve_wan_model_root,
)

# Conditioning variant -> DiT attribute holding its adapter, the newly introduced
# (always trainable) module.
CONDITIONER_ATTR = {"redar": "render_action_adapter", "action_only": "action_adapter_only"}
VAL_DATASETS = tuple(sorted(DATASET_IDS, key=DATASET_IDS.get))   # order of data_multi.DATASET_IDS


def _store_caches(args, accelerator):
    """Train/val datasets over the latent store.

    Returns ``(train_cache, val_cache, scheduler, row_w, store)``. Each item is a
    dict of stored encoder outputs (clean and render latents, I2V condition, CLIP
    image and text embeddings, normalised actions). Fails fast when the index is
    stale or the store lacks a row.
    """
    import pandas as _pd
    from diffusers import FlowMatchEulerDiscreteScheduler
    from doppel.latent_store import LatentStore, make_spec

    if int(args.max_sequence_length) != 512:
        raise SystemExit("[store] max_sequence_length must be 512: diffusers' Wan attention splits "
                         "the context at shape[1]-512 into text and CLIP tokens")
    if (int(args.num_frames), int(args.height), int(args.width)) != (17, 240, 432):
        raise SystemExit("[store] the latent store holds 17-frame clips on a 240x432 canvas")
    spec = make_spec(args.store_model_id or args.model_path, args.height, args.width, args.num_frames)
    store = LatentStore(args.latent_store, spec)
    stats = None
    if args.action_stats_path:
        if not isinstance(args.action_stats_path, dict):
            raise SystemExit("[store] action_stats_path must be a {robot_type: json} map")
        stats = load_stats_map(args.action_stats_path, args.dataset_base_path)

    def _rows(path, splits):
        df = _pd.read_parquet(path)
        want = [x for x in str(splits).split(",") if x]
        keep = df["split_name"].astype(str).isin(want)          # exact: train | val | test
        df = df[keep & (df["errors"].astype(str) == "")]
        rows = df.to_dict("records")
        if not rows:
            raise SystemExit(f"[store] no rows in {path} for splits {want}")
        return rows

    def _check(rows, tag):
        bad = stale_rows(rows, args.dataset_base_path,
                         missing_ok=bool(args.store_only_corpus))
        if bad:
            raise SystemExit(f"[store] {tag}: {len(bad)} inputs changed since the index was "
                             f"built (first: {bad[:3]}); rebuild the index and the latent store")
        miss = store.missing(rows, bool(args.store_with_render), bool(args.ignore_prompts))
        if any(miss.values()):
            raise SystemExit(f"[store] {tag}: store {store.dir} lacks "
                             + ", ".join(f"{len(v)} {k}" for k, v in miss.items())
                             + " -> run scripts/data/build_latent_store.py for this index")

    train_rows = _rows(args.train_index, args.index_train_splits)
    if accelerator.is_main_process:
        _check(train_rows, "train")
    common = dict(base_path=args.dataset_base_path, store=store, stats_map=stats,
                  with_render=bool(args.store_with_render),
                  ignore_prompts=bool(args.ignore_prompts),
                  action_field=args.action_field,
                  action_pad_dim=int(args.action_pad_dim or 0) or None)
    train_cache = MultiEmbodimentCache(train_rows, **common)
    akw = args.redar_adapter_kwargs if args.conditioning == "redar" else args.action_only_adapter_kwargs
    if stats is not None and isinstance(akw, dict) and akw.get("action_dim") is not None:
        if train_cache.action_pad_dim is not None:              # every sample is padded to action_pad_dim
            have, what = train_cache.action_pad_dim, "action_pad_dim"
        else:                                                   # one action width per index
            have = action_dim(train_rows[0]["robot_type"])
            what = f"the index's action width ({train_rows[0]['robot_type']})"
        if int(akw["action_dim"]) != have:
            raise SystemExit(f"[store] adapter action_dim {akw['action_dim']} != {what} {have}; "
                             "the adapter cannot consume these actions")
    val_cache = None
    if args.val_index:
        val_rows = _rows(args.val_index, args.index_val_splits)
        if accelerator.is_main_process:
            _check(val_rows, "val")
        val_base = MultiEmbodimentCache(val_rows, **common)
        # Dataset-interleaved order, so any val_n_samples prefix is balanced.
        val_cache = IndexView(val_base, interleave_by_dataset(val_base.rows))
    accelerator.wait_for_everyone()
    # Row sampling probabilities: per-dataset weights (multi-dataset runs) or
    # uniform. Each epoch draws epoch_len rows with replacement.
    row_w = None
    if args.sampler_dataset_weights:
        row_w = row_weights(train_rows, args.sampler_dataset_weights)
    elif int(args.epoch_len or 0) > 0:
        row_w = np.full(len(train_rows), 1.0 / len(train_rows))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        resolve_wan_model_root(args.model_path), subfolder="scheduler")
    if accelerator.is_main_process:
        by = _pd.Series([r["dataset"] for r in train_rows]).value_counts().to_dict()
        accelerator.print(f"[store] {store.dir}: train rows {len(train_rows)} {by}; "
                          f"val {len(val_cache) if val_cache is not None else 0}; "
                          f"stats sha {({k: v['sha256'][:12] for k, v in (stats or {}).items()})}; "
                          f"weighted sampling {row_w is not None}")
    return train_cache, val_cache, scheduler, row_w, store


def _cast_module_for_fsdp(module: torch.nn.Module, dtype: torch.dtype) -> Dict[str, int]:
    """FSDP flattening requires a uniform floating dtype within each wrapped module,
    but the DiT keeps a few modules in fp32. Cast all floating params/buffers to
    ``dtype`` before wrapping."""
    n_params = 0
    n_buffers = 0
    for p in module.parameters():
        if p.is_floating_point() and p.dtype != dtype:
            p.data = p.data.to(dtype=dtype)
            n_params += p.numel()
    for b in module.buffers():
        if b.is_floating_point() and b.dtype != dtype:
            b.data = b.data.to(dtype=dtype)
            n_buffers += b.numel()
    return {"params": n_params, "buffers": n_buffers}


def _materialize_meta_submodules(model: torch.nn.Module) -> int:
    """Replace meta tensors (modules absent from the pretrained checkpoint) with
    real CPU tensors before any ``.cpu()`` / FSDP wrapping."""
    meta_modules = [
        m
        for _name, m in model.named_modules()
        if any(p.is_meta for p in m.parameters(recurse=False))
        or any(b.is_meta for b in m.buffers(recurse=False))
    ]
    if not meta_modules:
        return 0

    for m in meta_modules:
        m.to_empty(device="cpu", recurse=False)
        # nn.MultiheadAttention exposes only the private ``_reset_parameters``;
        # without the elif its in_proj_weight would fall through to the zero branch.
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()
        elif hasattr(m, "_reset_parameters") and callable(m._reset_parameters):
            m._reset_parameters()
        else:
            for p in m.parameters(recurse=False):
                with torch.no_grad():
                    p.zero_()
            for b in m.buffers(recurse=False):
                with torch.no_grad():
                    b.zero_()
    return len(meta_modules)


def _trainable_prefixes(args, n_blocks_total: int):
    """State-dict key prefixes of the trainable subset (what ``conditioner.pt`` holds)."""
    prefixes = [CONDITIONER_ATTR[args.conditioning] + "."]
    n_blocks = int(args.unfreeze_last_n_blocks)
    if n_blocks > 0:
        prefixes += [f"blocks.{i}." for i in range(n_blocks_total - n_blocks, n_blocks_total)]
        prefixes += ["norm_out.", "proj_out.", "condition_embedder.time_proj."]
    return tuple(prefixes)


def _conditioner_writer(args, dit, accelerator: Accelerator):
    """A ``write(full_state, ckpt_path)`` closure that strips a gathered full state
    dict down to the trainable subset and writes it atomically (tmp + fsync +
    rename). Main process only."""
    keep_prefixes = _trainable_prefixes(args, len(accelerator.unwrap_model(dit).blocks))

    def write(full_state, ckpt_path: str):
        small_state = {
            k: v.detach().to("cpu")
            for k, v in full_state.items()
            if k.startswith(keep_prefixes)
        }
        tmp_ckpt = ckpt_path + ".tmp"
        with open(tmp_ckpt, "wb") as f:
            torch.save(small_state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_ckpt, ckpt_path)
        n_saved = sum(v.numel() for v in small_state.values())
        accelerator.print(
            f"Saved trainable weights ({n_saved / 1e6:.2f}M params, "
            f"{len(small_state)} tensors) to {ckpt_path}"
        )
        del small_state

    return write


def _load_resume_ckpt(
    dit: torch.nn.Module, ckpt_path: str, accelerator: Accelerator
) -> None:
    """Load a saved ``conditioner.pt`` (trainable subset) or a released safetensors overlay
    (``--resume_mode warm`` only) before FSDP wrapping.

    ``strict=False`` so only the saved keys move. ``mmap=True`` lets all ranks on a
    node share the file through the page cache instead of each holding a private
    copy; ``load_state_dict`` casts each tensor in place.
    """
    path = ckpt_path.strip()
    files = _safetensors_files(path)      # a released safetensors overlay (bf16 blocks, fp32 adapter)
    if files is None:
        sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    else:
        from safetensors.torch import load_file
        sd = {}
        for f in files:
            sd.update(load_file(f, device="cpu"))
    sd, _ = remap_state_dict(sd)          # research-code key names -> this package's
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    if accelerator.is_main_process:
        n_ok = len(sd) - len(unexpected)
        accelerator.print(f"[train] resume: loaded {n_ok}/{len(sd)} tensors from {path}")
        if unexpected:
            accelerator.print(f"[train] resume WARN unexpected keys (first 8): {unexpected[:8]}")


def _bool_flag(v) -> bool:
    """argparse ``type`` for a flag that is also a JSON config key: accepts a JSON
    bool (via ``set_defaults``) or the strings 1/0/true/false/yes/no on the CLI."""
    if isinstance(v, bool):
        return v
    sv = str(v).strip().lower()
    if sv in ("1", "true", "yes", "y", "on"):
        return True
    if sv in ("0", "false", "no", "n", "off", ""):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def _json_arg(v):
    """argparse ``type`` for a JSON-valued option (a JSON string on the CLI)."""
    return json.loads(v) if isinstance(v, str) else v


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args()
    defaults: Dict[str, Any] = {}
    if pre_args.config:
        with open(pre_args.config, encoding="utf-8") as f:
            defaults = json.load(f)

    p = argparse.ArgumentParser(
        description="FSDP training of the conditioned Wan2.1 I2V DiT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="JSON file whose keys are the option names below.")
    # ---- model ----
    p.add_argument("--model_path", type=str, default=None,
                   help="Wan2.1 I2V diffusers checkpoint (HF repo id or local dir).")
    p.add_argument("--conditioning", type=str, default="redar", choices=tuple(CONDITIONER_ATTR),
                   help="redar: render latents + actions through the render-action adapter. "
                        "action_only: actions only, through the action-only adapter.")
    p.add_argument("--redar_adapter_kwargs", type=_json_arg, default=None,
                   help="JSON dict for RenderActionAdapter (conditioning=redar); "
                        "action_dim is required.")
    p.add_argument("--action_only_adapter_kwargs", type=_json_arg, default=None,
                   help="JSON dict for the action-only adapter (conditioning=action_only); "
                        "action_dim is required.")
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=0,
                   help="Number of final DiT blocks trained together with the adapter "
                        "(plus norm_out, proj_out and the timestep projection).")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=("no", "fp16", "bf16"),
                   help="DiT storage/compute dtype, passed to accelerate. With bf16/fp16 the "
                        "trainable FSDP parameters keep fp32 master weights.")
    # ---- data (latent store) ----
    p.add_argument("--dataset_base_path", type=str, default=None,
                   help="Root that relative paths in the index and in action_stats_path resolve against.")
    p.add_argument("--latent_store", type=str, default=None,
                   help="Root of the latent store (scripts/data/build_latent_store.py).")
    p.add_argument("--store_model_id", type=str, default=None,
                   help="Model id recorded in the store spec (default: --model_path).")
    p.add_argument("--train_index", type=str, default=None, help="index.parquet (scripts/data/build_index.py)")
    p.add_argument("--val_index", type=str, default=None, help="index.parquet holding the val rows")
    p.add_argument("--index_train_splits", type=str, default="train",
                   help="Comma list of index 'split_name' values (train | val | test).")
    p.add_argument("--index_val_splits", type=str, default="val")
    p.add_argument("--store_with_render", action="store_true",
                   help="Read the render latents too (conditioning=redar).")
    p.add_argument(
        "--store_only_corpus", action="store_true",
        help="The corpus carries the latent store but not the source videos: the freshness "
             "check skips rows whose video/render file is absent. A present file that changed "
             "since indexing is still fatal, and the store-completeness check always applies.")
    p.add_argument("--action_stats_path", type=_json_arg, default=None,
                   help="JSON {robot_type: stats.json} map (relative to --dataset_base_path) with "
                        "per-dimension mean/std for z-scoring the action stream.")
    p.add_argument("--action_field", type=str, default="state",
                   help="Field of the per-clip action file to condition on.")
    p.add_argument("--action_pad_dim", type=int, default=0,
                   help="Mixed-width embodiments: zero-extend every sample's z-scored (17, D) action to "
                        "this many slots so all embodiments share one action encoder and one batch "
                        "(data_multi.pad_actions). Must equal the adapter's action_dim. 0 = off: the "
                        "index must hold a single action width.")
    p.add_argument("--num_frames", type=int, default=17)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=432)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--ignore_prompts", action="store_true",
                   help="Use the empty-prompt text embedding for every clip.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epoch_len", type=int, default=0,
                   help="Samples drawn per epoch (with replacement); 0 = one permutation of the rows.")
    p.add_argument("--sampler_dataset_weights", type=_json_arg, default=None,
                   help='Per-dataset sampling probabilities, e.g. {"droid": 0.34, "robomind": 0.33, '
                        '"bridge": 0.33}; clips are sampled uniformly within each dataset.')
    # ---- optimisation ----
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-GPU micro-batch size. Global batch = batch_size * num_gpus * "
                        "gradient_accumulation_steps.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="LR of the pretrained Wan backbone.")
    p.add_argument("--conditioner_lr_multiplier", type=float, default=1.0,
                   help="LR multiplier for the newly introduced conditioning adapter, relative "
                        "to --learning_rate.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    p.add_argument("--lr_scheduler", type=str, default="cosine_warmup",
                   choices=("constant", "cosine", "cosine_warmup"),
                   help="cosine_warmup: linear warmup, cosine decay to lr * lr_min_ratio, then "
                        "constant; cosine: the same without warmup.")
    p.add_argument("--lr_warmup_steps", type=int, default=300)
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--cond_dropout_prob", type=float, default=0.0,
                   help="Conditioning dropout: with probability p a training step runs without "
                        "render latents and actions (the adapter output is zero), regularizing "
                        "the adapter and providing an action-unconditioned branch.")
    p.add_argument("--seed", type=int, default=42)
    # ---- EMA of the trainable weights (doppel.ema) ----
    p.add_argument(
        "--ema_decay", type=float, default=0.0,
        help="EMA of the trainable weights, ema = d*ema + (1-d)*w per update; 0 = off. With "
             "--ema_update_every N the averaging horizon is N/(1-d) optimizer steps. Kept per "
             "rank over the rank's own FSDP shard, in fp32.")
    p.add_argument("--ema_update_every", type=int, default=1,
                   help="Update the EMA every N-th optimizer step.")
    p.add_argument("--ema_device", type=str, default="gpu", choices=("gpu", "cpu"),
                   help="Where the EMA shadow lives ('cpu' uses host RAM instead of GPU memory).")
    p.add_argument(
        "--ema_eval", type=_bool_flag, default=None,
        help="Validate and export conditioner.pt from the EMA weights (the raw weights "
             "go to conditioner_raw.pt and val_raw/*). Default: true when --ema_decay > 0.")
    p.add_argument("--ema_warmup_steps", type=int, default=0,
                   help="Through this many optimizer steps the EMA is a copy of the weights.")
    # ---- schedule length, validation, checkpoints ----
    p.add_argument("--num_epochs", type=int, default=1, help="Epochs to run in this launch.")
    p.add_argument("--total_epochs", type=int, default=None,
                   help="Epochs the LR schedule spans (default epoch_offset + num_epochs).")
    p.add_argument("--epoch_offset", type=int, default=0,
                   help="Added to the epoch index for data order, logs and checkpoint names "
                        "(epoch_{offset + i}); set to the next epoch when resuming.")
    p.add_argument("--val_every_n_epochs", type=int, default=1)
    p.add_argument("--val_n_samples", type=int, default=-1,
                   help="Val clips per pass (-1 = all).")
    p.add_argument("--val_timesteps", type=int, default=4,
                   help="Fixed, evenly spaced timesteps per val clip.")
    p.add_argument("--save_every_n_epochs", type=int, default=10)
    p.add_argument("--keep_last_n_epochs", type=int, default=0,
                   help="Keep only the newest N epoch_* checkpoint dirs (0 = keep all).")
    p.add_argument("--save_optimizer_state", type=int, default=None, choices=(0, 1),
                   help="Write the resume state (sharded fp32 master weights + AdamW moments, "
                        "RNG, resume_state.json) at every save. Default 1 unless resume_mode is warm.")
    p.add_argument("--keep_optim_state_last_n", type=int, default=1,
                   help="Keep the optimizer and EMA state for only the newest N epoch dirs (0 = all).")
    # ---- resume ----
    p.add_argument(
        "--resume_mode", type=str, default="warm", choices=("warm", "schedule", "continue"),
        help="warm: weights only, fresh schedule over this launch. schedule: + LR position and "
             "global_step, fresh AdamW moments. continue: + AdamW moments, scheduler, EMA and "
             "per-rank RNG.")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="conditioner.pt of a saved epoch (requires --epoch_offset > 0). With "
                        "ema_eval it holds the EMA weights; warm/schedule resumes that should continue "
                        "from the raw weights use conditioner_raw.pt.")
    p.add_argument("--resume_state_dir", type=str, default=None,
                   help="epoch_N dir holding resume_state.json + optim/ + rng_rank*.pt "
                        "(default: dirname of --resume_ckpt).")
    p.add_argument("--allow_schedule_mismatch", action="store_true",
                   help="schedule/continue: accept changed schedule hyperparameters and rebuild the "
                        "LR curve from the current args at the same relative position.")
    # ---- logging ----
    p.add_argument("--output_dir", type=str, default="runs/doppel")
    p.add_argument("--log_every_n_steps", type=int, default=25)
    p.add_argument("--wandb_project", type=str, default="doppel")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Default: the run directory name.")
    p.add_argument("--wandb_mode", type=str, default="disabled",
                   choices=("online", "offline", "disabled"))

    # set_defaults() accepts any keyword, so an unknown config key would be
    # silently ignored; reject it instead.
    known = {a.dest for a in p._actions}
    unknown = sorted(k for k in defaults if k != "config" and k not in known)
    if unknown:
        p.error("unknown config key(s): " + ", ".join(unknown))
    p.set_defaults(**{k: v for k, v in defaults.items() if k != "config"})
    args = p.parse_args(remaining)
    if not (args.model_path and args.dataset_base_path and args.latent_store and args.train_index):
        p.error("Provide --model_path, --dataset_base_path, --latent_store and --train_index "
                "(via CLI or --config JSON).")
    args.ema_decay = float(args.ema_decay or 0.0)
    if args.ema_decay and not (0.0 < args.ema_decay < 1.0):
        p.error(f"--ema_decay must be 0 (off) or in (0, 1), got {args.ema_decay}")
    if int(args.ema_update_every) < 1:
        p.error(f"--ema_update_every must be >= 1, got {args.ema_update_every}")
    if args.ema_decay and int(args.ema_warmup_steps) < 0:
        p.error(f"--ema_warmup_steps must be >= 0, got {args.ema_warmup_steps}")
    if args.ema_eval is None:
        args.ema_eval = bool(args.ema_decay > 0)
    else:
        args.ema_eval = _bool_flag(args.ema_eval) and bool(args.ema_decay > 0)
    return args


def _load_dit_only(
    model_path: str,
    dtype: torch.dtype,
    conditioning: str,
    redar_adapter_kwargs: Optional[Dict[str, Any]] = None,
    action_only_adapter_kwargs: Optional[Dict[str, Any]] = None,
):
    """Load only the conditioned DiT; the encoders' outputs come from the latent store."""
    root = resolve_wan_model_root(model_path)
    if conditioning == "redar":
        model_cls = RenderActionWanTransformer
        tkw = dict(torch_dtype=dtype, redar_adapter_kwargs=redar_adapter_kwargs)
    elif conditioning == "action_only":
        model_cls = ActionOnlyWanTransformer
        tkw = dict(torch_dtype=dtype, action_only_adapter_kwargs=action_only_adapter_kwargs)
    else:
        raise ValueError(f"unknown conditioning {conditioning!r}")
    if os.path.isdir(os.path.join(root, "transformer")):
        return model_cls.from_pretrained(os.path.join(root, "transformer"), **tkw)
    return model_cls.from_pretrained(root, subfolder="transformer", **tkw)


@torch.no_grad()
def _validation_pass(
    dit,
    embed_cache_val,
    scheduler,
    accelerator,
    device,
    dtype: torch.dtype,
    n_timesteps_per_sample: int,
    seed: int,
    max_samples: int,
) -> Dict[str, float]:
    """Held-out flow loss over a fixed grid of timesteps per clip, with noise seeded
    per (clip, timestep), so the result is a deterministic function of the weights.
    Returns global means overall, per timestep quartile and per dataset.
    Collective under FSDP: every rank must call this.
    """
    import torch.distributed as dist

    was_training = dit.training
    dit.eval()

    rank = accelerator.process_index
    world = accelerator.num_processes
    n = len(embed_cache_val)
    if max_samples is not None and max_samples > 0:
        n = min(n, int(max_samples))
    K = max(1, int(n_timesteps_per_sample))

    n_t = len(scheduler.timesteps)
    t_indices = [int((k + 0.5) * n_t / K) for k in range(K)]
    bucket_edges = [n_t * k // 4 for k in range(5)]

    work = [(s, t) for s in range(n) for t in t_indices]
    pad = (-len(work)) % max(1, world)
    if pad:
        work = work + work[:pad]
    my_work = work[rank::world] if world > 1 else work

    total = 0.0
    count = 0
    bucket_sum = [0.0, 0.0, 0.0, 0.0]
    bucket_cnt = [0, 0, 0, 0]
    ds_sum = [0.0] * len(VAL_DATASETS)
    ds_cnt = [0] * len(VAL_DATASETS)
    last_idx, cached = None, None

    for s_idx, t_idx in my_work:
        if s_idx != last_idx:    # one read per clip, not per timestep
            cached, last_idx = embed_cache_val[s_idx], s_idx
        clean_latents = cached["clean_latents"].to(device=device, dtype=dtype, non_blocking=True)
        prompt_embeds = cached["prompt_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        image_embeds = cached["image_embeds"].to(device=device, dtype=dtype, non_blocking=True)
        condition = cached["condition"].to(device=device, dtype=dtype, non_blocking=True)

        gen = torch.Generator(device=clean_latents.device).manual_seed(
            int(seed) * 1_000_003 + int(s_idx) * 1_009 + int(t_idx)
        )
        noise = torch.randn(
            clean_latents.shape, generator=gen,
            dtype=clean_latents.dtype, device=clean_latents.device,
        )
        timestep_scalar = scheduler.timesteps[t_idx].to(device=device).float()
        timestep_batch = timestep_scalar.expand(clean_latents.shape[0])
        noisy_latents = scheduler.scale_noise(clean_latents, timestep_batch, noise)
        noisy_latents[:, :, 0:1] = clean_latents[:, :, 0:1]
        latent_in = torch.cat([noisy_latents, condition], dim=1)
        target = noise - clean_latents

        forward_kwargs = dict(
            hidden_states=latent_in,
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=image_embeds,
            return_dict=False,
        )
        if "render_latents" in cached:
            forward_kwargs["render_latents"] = cached["render_latents"].to(
                device=device, dtype=dtype, non_blocking=True)
        if "actions" in cached:
            forward_kwargs["actions"] = cached["actions"].to(
                device=device, dtype=dtype, non_blocking=True).unsqueeze(0)
        pred = dit(**forward_kwargs)[0]
        loss = F.mse_loss(pred[:, :, 1:].float(), target[:, :, 1:].float())
        v = float(loss.detach())
        total += v
        count += 1
        b = min(3, max(0, sum(1 for e in bucket_edges[1:] if t_idx >= e)))
        bucket_sum[b] += v
        bucket_cnt[b] += 1
        if "dataset_id" in cached:
            ds_sum[int(cached["dataset_id"])] += v
            ds_cnt[int(cached["dataset_id"])] += 1

    # All-reduce sums + counts so every rank gets the global means.
    nd = len(VAL_DATASETS)
    arr = torch.tensor(
        [total, count, *bucket_sum, *bucket_cnt, *ds_sum, *ds_cnt],
        dtype=torch.float64, device=device,
    )
    if dist.is_available() and dist.is_initialized() and world > 1:
        dist.all_reduce(arr, op=dist.ReduceOp.SUM)
    a = arr.tolist()
    g_total, g_count = a[0], a[1]
    g_b_sum = a[2:6]
    g_b_cnt = a[6:10]

    if was_training:
        dit.train()

    def _safe_div(num, den):
        return float(num / den) if den > 0 else float("nan")

    return {
        "loss_flow_mean": _safe_div(g_total, g_count),
        **{f"loss_flow_q{k}": _safe_div(g_b_sum[k], g_b_cnt[k]) for k in range(4)},
        **{f"loss_flow_{name}": _safe_div(a[10 + k], a[10 + nd + k])
           for k, name in enumerate(VAL_DATASETS) if a[10 + nd + k] > 0},
        "n_evaluations": int(g_count),
        "n_samples_evaluated": int(g_count // max(1, K)),
    }


def _grad_accum_plugin(num_steps: int) -> GradientAccumulationPlugin:
    """Gradient accumulation that does not grow FSDP memory.

    accelerate's default ``accumulate()`` wraps every non-sync micro-step in
    ``model.no_sync()``, which under FSDP keeps the unsharded gradient of every
    trainable parameter on every rank until the sync step. ``sync_each_batch``
    reduce-scatters every micro-step instead and accumulates into the sharded
    gradient; ``accelerator.sync_gradients`` still fires once per ``num_steps``
    micro-steps, so optimizer / scheduler / EMA gating and the 1/num_steps loss
    scaling are unchanged.
    """
    return GradientAccumulationPlugin(num_steps=max(1, int(num_steps)), sync_each_batch=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.resume_ckpt is not None and not str(args.resume_ckpt).strip():
        args.resume_ckpt = None
    if args.resume_ckpt and int(args.epoch_offset) <= 0:
        raise SystemExit(
            "[train] --resume_ckpt requires --epoch_offset > 0 (e.g. 10 after "
            "epoch_9) so new epoch_* directories do not overwrite the earlier ones."
        )

    # Long collective timeout: rank 0 alone validates the index against the store
    # before training, and checkpoint writes take minutes.
    accelerator = Accelerator(
        gradient_accumulation_plugin=_grad_accum_plugin(int(args.gradient_accumulation_steps)),
        mixed_precision=args.mixed_precision,
        log_with=None,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=4))],
    )
    # Resolved before the run dir is set up so args.json records it.
    resume_state = _rs.resolve_resume_mode(args, accelerator, log=accelerator.print)

    # set_seed() is not rank-aware; offset the global RNG per rank so the ranks draw
    # different timesteps and noise. The conditioning-dropout coin and the val noise
    # use their own generators and stay identical across ranks.
    _rank_seed = int(args.seed) + 100_003 * int(accelerator.process_index)
    torch.manual_seed(_rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_rank_seed)
    np.random.seed(_rank_seed % (2**32 - 1))
    random.seed(_rank_seed)

    args = _setup_run_dir_and_logging(args, accelerator)

    # DiT storage dtype. With mixed_precision bf16, accelerate keeps fp32 master
    # copies of the trainable FSDP parameters (fp32 gradients and AdamW moments).
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }[args.mixed_precision]

    if accelerator.is_main_process:
        accelerator.print(
            f"[train] world_size={accelerator.num_processes}  device={accelerator.device}  "
            f"conditioning={args.conditioning}  mixed_precision={args.mixed_precision}"
        )

    # ---- data ----
    embed_cache, embed_cache_val, scheduler, row_w, latent_store = _store_caches(args, accelerator)

    # ---- model ----
    dit = _load_dit_only(
        args.model_path, dtype, args.conditioning,
        redar_adapter_kwargs=args.redar_adapter_kwargs,
        action_only_adapter_kwargs=args.action_only_adapter_kwargs,
    )
    n_meta = _materialize_meta_submodules(dit)
    if accelerator.is_main_process and n_meta > 0:
        accelerator.print(f"[train] materialized {n_meta} meta submodules on CPU")
    # Keep the DiT on CPU until FSDP wraps it (avoids per-rank device mismatch).
    dit = dit.cpu()
    casted = _cast_module_for_fsdp(dit, dtype)
    if accelerator.is_main_process and (casted["params"] > 0 or casted["buffers"] > 0):
        accelerator.print(
            f"[train] cast DiT floating tensors to {dtype}: "
            f"{casted['params'] / 1e6:.2f}M params, {casted['buffers'] / 1e6:.2f}M buffers"
        )
    dit.train()

    # Trainable set: the conditioning adapter, the last N DiT blocks, the output
    # head and the timestep projection. Everything else stays frozen.
    conditioner = getattr(dit, CONDITIONER_ATTR[args.conditioning])
    for p_ in dit.parameters():
        p_.requires_grad = False
    for p_ in conditioner.parameters():
        p_.requires_grad = True
    if args.unfreeze_last_n_blocks > 0:
        for block in dit.blocks[-args.unfreeze_last_n_blocks:]:
            for p_ in block.parameters():
                p_.requires_grad = True
        for p_ in dit.norm_out.parameters():
            p_.requires_grad = True
        for p_ in dit.proj_out.parameters():
            p_.requires_grad = True
        for p_ in dit.condition_embedder.time_proj.parameters():
            p_.requires_grad = True

    if args.resume_ckpt:
        _load_resume_ckpt(dit, str(args.resume_ckpt), accelerator)

    trainable_params = [p for p in dit.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in dit.parameters())
    if accelerator.is_main_process:
        n_cond = sum(p.numel() for p in conditioner.parameters())
        accelerator.print(
            f"[train] DiT params: {n_total / 1e9:.2f}B total, "
            f"{n_trainable / 1e6:.1f}M trainable ({100 * n_trainable / n_total:.2f}%); "
            f"{CONDITIONER_ATTR[args.conditioning]}={n_cond / 1e6:.2f}M"
        )

    # Per-module learning rates: the pretrained backbone at learning_rate, the
    # newly introduced adapter at learning_rate * conditioner_lr_multiplier.
    COND_LR_MULT = float(args.conditioner_lr_multiplier)
    param_groups, _lr_info = build_param_lr_groups(
        [(n, p_) for n, p_ in dit.named_parameters() if p_.requires_grad],
        float(args.learning_rate),
        conditioner_mult=COND_LR_MULT,
    )
    if accelerator.is_main_process:
        accelerator.print(
            f"[train] LR groups: backbone={_lr_info['base_numel'] / 1e6:.1f}M params "
            f"(lr={args.learning_rate:.1e}), conditioner={_lr_info['conditioner_numel'] / 1e6:.2f}M "
            f"params in {_lr_info['conditioner_tensors']} tensors "
            f"(lr={args.learning_rate * COND_LR_MULT:.1e}): " + ", ".join(_lr_info["conditioner_roots"])
        )

    optimizer = torch.optim.AdamW(
        # LambdaLR scales every group off its own base LR, so the ratio between
        # the groups holds through warmup and the whole cosine.
        param_groups,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )

    n_samples = int(args.epoch_len or 0) or len(embed_cache)
    world_size = int(accelerator.num_processes)
    micro_pe = fsdp_micro_steps_per_epoch(n_samples, world_size, int(args.batch_size))
    # The LR curve spans the whole planned run (total_epochs), except for a warm
    # restart, which schedules only this launch.
    sched_epochs = int(args.total_epochs) if args.resume_mode != "warm" else int(args.num_epochs)
    opt_total = optimizer_step_count(micro_pe * sched_epochs, int(args.gradient_accumulation_steps))
    lr_sched = build_lr_scheduler(
        optimizer,
        str(args.lr_scheduler),
        int(args.lr_warmup_steps),
        float(args.lr_min_ratio),
        opt_total,
    )

    dit, optimizer = accelerator.prepare(dit, optimizer)

    # reshard around every forward / state-dict under ema.swap_in() (see doppel/ema.py)
    _ema_handles = fsdp_flat_param_handles(dit)

    def _ema_reshard() -> None:
        free_unsharded_flat_params(_ema_handles)

    # Activation checkpointing is enabled on the wrapped model; enabling it before
    # FSDP wrapping breaks the backward under FULL_SHARD.
    _dit_inner = accelerator.unwrap_model(dit)
    if hasattr(_dit_inner, "enable_gradient_checkpointing"):
        _dit_inner.enable_gradient_checkpointing()

    # ---- resume: optimizer + scheduler position (after prepare, before any backward) ----
    # dcp_load restores the model shards first, which refreshes FSDP's orig-param
    # views to fp32; only then is the optimizer state touched (resume_state.py).
    opt_step = 0
    if resume_state is not None:
        _inner_opt = _rs.inner_optimizer(optimizer)
        if args.resume_mode == "continue":
            _rs.quiet_fsdp_load_warnings(accelerator.print)
            _t_load = time.time()
            _rs.dcp_load(dit, _inner_opt, resume_state["_dir"],
                         expect_step=int(resume_state.get("opt_step", 0)),
                         dcp_dir=resume_state.get("dcp_dir", _rs.RESUME_DCP_DIR),
                         log=accelerator.print, rank=int(accelerator.process_index))
            accelerator.print(
                f"[resume-state] loaded {resume_state['_dir']} in {time.time() - _t_load:.0f}s"
            )
            _rs.compare_checksums(
                _rs.state_checksums(accelerator, dit, _inner_opt),
                resume_state.get("checksum") or {}, accelerator.print,
            )
            _rs.reassert_optimizer_hparams(_inner_opt, args, accelerator.print)
            # Return the load's transient buffers to the allocator before training.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        if args.resume_mode != "warm":
            opt_step = _rs.fast_forward_scheduler(
                lr_sched, _inner_opt, resume_state,
                now=_rs.schedule_now(args, opt_total=opt_total, micro_pe=int(micro_pe),
                                     lr_sched=lr_sched),
                allow_mismatch=bool(args.allow_schedule_mismatch), log=accelerator.print,
            )

    # ---- EMA of the trainable weights (per-rank shard shadow, no collectives) ----
    # Built after the resume block so opt_step is final, and initialised lazily at
    # the first successful optimizer step, when the orig-param views are fp32.
    ema: Optional[ShardEMA] = None
    EMA_EVAL = False

    def _ema_all_ranks(value):
        """all_gather_object of one small Python value (rank order); collective."""
        import torch.distributed as dist
        world = int(accelerator.num_processes)
        if dist.is_available() and dist.is_initialized() and world > 1:
            out = [None] * world
            dist.all_gather_object(out, value)
            return out
        return [value]

    def _ema_check_ranks_agree(opt_step_now: int) -> None:
        """Collective, called by every rank right after the step that materialised
        its shadow. Loading the saved shards is per rank (a rank may fail its
        shape / world-size check and fall back to the weights), so compare the
        sources: under continue a disagreement is fatal, under schedule/warm every
        rank falls back to the weights so the EMA is never a rank-wise mixture."""
        srcs = _ema_all_ranks(str(ema.initialised_from or "?"))
        short = ["shards" if s.startswith("shards:") else s for s in srcs]
        per_rank = ", ".join(f"r{r}={s}" for r, s in enumerate(short))
        accelerator.print(f"[ema] initialised on {len(srcs)} rank(s): {per_rank}")
        if len(set(short)) > 1:
            if args.resume_mode == "continue":
                raise SystemExit(
                    f"[ema] ranks disagree on the EMA source under --resume_mode continue "
                    f"({per_rank}). Restore the missing/mismatched optim/ema/local_rank*.pt "
                    "shards or use --resume_mode schedule (EMA restarts from the weights)."
                )
            ema.reinitialise_from_weights(
                opt_steps=int(opt_step_now),
                reason=f"ranks disagree on the EMA source ({per_rank}); with --resume_mode "
                       f"{args.resume_mode} every rank falls back to the weights",
            )

    if float(args.ema_decay) > 0.0:
        ema = ShardEMA(
            list(dit.named_parameters()), float(args.ema_decay),
            update_every=int(args.ema_update_every), device=str(args.ema_device),
            warmup_steps=int(args.ema_warmup_steps), log=accelerator.print,
        )
        ema.opt_steps = int(opt_step)
        EMA_EVAL = bool(args.ema_eval)
        _ema_src = "the current weights at the first optimizer step"
        if resume_state is not None and resume_state.get("_dir"):
            _ema_path = ema_shard_path(
                str(resume_state["_dir"]), int(accelerator.process_index),
                resume_state.get("dcp_dir", _rs.RESUME_DCP_DIR),
            )
            _ema_dir = os.path.dirname(_ema_path)
            _ema_saved = resume_state.get("ema") if isinstance(resume_state.get("ema"), dict) else {}
            _ema_expected = bool(_ema_saved.get("initialised"))
            # The shard file is per rank: gather who has it so every rank takes the
            # same branch.
            _ema_have = _ema_all_ranks(bool(os.path.isfile(_ema_path)))
            _ema_missing = [r for r, h in enumerate(_ema_have) if not h]
            if _ema_missing and args.resume_mode == "continue" and _ema_expected:
                raise SystemExit(
                    f"[ema] --resume_mode continue: {resume_state['_dir']}/{_rs.RESUME_STATE_FILE} "
                    f"says the EMA was initialised, but the EMA shard is missing for rank(s) "
                    f"{_ema_missing} under {_ema_dir}. Restore optim/ema/local_rank*.pt, or use "
                    "--resume_mode schedule to restart the EMA from the weights."
                )
            if not _ema_missing:
                # continue: same world size, so a mismatch is an error. schedule/warm
                # may run on another world size and fall back to the weights (made
                # collective by _ema_check_ranks_agree after the first step). The
                # shards carry the EMA's own step count; under warm the trainer's
                # opt_step restarts at 0, so a difference is expected there.
                ema.schedule_load(
                    _ema_path, strict=(args.resume_mode == "continue"),
                    warn_on_step_mismatch=(args.resume_mode != "warm"),
                    world_size=int(accelerator.num_processes),
                )
                _ema_src = f"saved shards {_ema_dir}"
            elif len(_ema_missing) == len(_ema_have):
                accelerator.print(
                    f"[ema] WARN no EMA shards under {_ema_dir}; the EMA restarts from the "
                    "current weights at the first optimizer step"
                )
                _ema_src += " (no saved shards)"
            else:
                accelerator.print(
                    f"[ema] WARN EMA shards under {_ema_dir} are missing for rank(s) {_ema_missing}; "
                    f"all {len(_ema_have)} ranks restart the EMA from the current weights"
                )
                _ema_src += " (saved shards incomplete)"
        accelerator.print(
            f"[ema] decay={args.ema_decay} update_every={args.ema_update_every} "
            f"device={args.ema_device} warmup_steps={args.ema_warmup_steps} "
            f"eval={EMA_EVAL}  {ema.n_params} trainable tensors, "
            f"~{ema.numel() * 4 / 2**30:.1f} GiB fp32 per rank; initialised from {_ema_src}"
        )

    if accelerator.is_main_process and lr_sched is not None:
        accelerator.print(
            f"[train] lr_scheduler={args.lr_scheduler}  warmup_steps={args.lr_warmup_steps}  "
            f"lr_min_ratio={args.lr_min_ratio}  total_opt_steps={opt_total}  "
            f"micro_steps/epoch/rank={micro_pe}"
        )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- wandb (optional) ----
    wandb_run = None
    if accelerator.is_main_process and args.wandb_mode != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=vars(args),
                dir=args.output_dir,
            )
        except Exception as e:
            accelerator.print(f"[wandb] disabled (init failed: {e})")
            wandb_run = None

    rank = accelerator.process_index
    device = accelerator.device
    global_step = (
        int(resume_state["global_step"])
        if (resume_state is not None and args.resume_mode != "warm")
        else 0
    )
    t0 = time.time()
    # Smoothed training-loss signals. The per-step flow loss depends strongly on the
    # sampled noise level, so the loop keeps an EMA over steps, running means per
    # timestep quartile, and a per-epoch mean.
    flow_ema: Optional[float] = None
    flow_ema_decay = 0.95
    last_valid_grad_norm: float = float("nan")
    n_t = len(scheduler.timesteps)
    bucket_edges = [n_t * k // 4 for k in range(5)]   # [0, n/4, n/2, 3n/4, n]
    bucket_sum = [0.0, 0.0, 0.0, 0.0]
    bucket_cnt = [0, 0, 0, 0]

    if resume_state is not None and args.resume_mode == "continue":
        _st = resume_state.get("stats") or {}
        flow_ema = _st.get("flow_ema", flow_ema)
        bucket_sum = list(_st.get("bucket_sum", bucket_sum))
        bucket_cnt = list(_st.get("bucket_cnt", bucket_cnt))
        _rng_path = os.path.join(resume_state["_dir"], f"rng_rank{accelerator.process_index}.pt")
        if os.path.isfile(_rng_path):
            # Last RNG-consuming step before the loop.
            _rs.restore_rng(torch.load(_rng_path, weights_only=False), accelerator.device)
            accelerator.print("[resume-state] restored per-rank RNG and logging accumulators")
        else:
            accelerator.print(
                f"[resume-state] WARN {_rng_path} missing: the noise/timestep stream "
                "restarts from the rank-offset seed"
            )

    # One persistent DataLoader for the whole run: the workers fork once, and each
    # epoch's batch plan reaches them through the stateful batch sampler. The
    # loader's own generator keeps its worker-seed draw off the global CPU RNG.
    _nw = int(args.num_workers)
    store_sampler = EpochBatchSampler()
    store_loader = torch.utils.data.DataLoader(
        embed_cache, batch_sampler=store_sampler, collate_fn=list_collate,
        num_workers=_nw, pin_memory=True, prefetch_factor=(4 if _nw > 0 else None),
        persistent_workers=(_nw > 0),
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    bs = max(1, int(args.batch_size))

    for epoch in range(args.num_epochs):
        abs_e = int(args.epoch_offset) + int(epoch)
        epoch_flow_sum = 0.0
        epoch_flow_cnt = 0
        # This rank's micro-batches, identical on every rank without communication:
        # n_samples rows drawn from row_w (seeded by seed + epoch), padded to a
        # multiple of world_size * batch_size and strided by rank, so all ranks run
        # the same number of equally sized steps.
        my_batches = epoch_batches(len(embed_cache), abs_e, args.seed, world_size, rank, bs,
                                   weights=row_w, epoch_len=n_samples)
        # Consumes one int64 from the global CPU RNG per epoch so the noise/timestep
        # stream depends only on (seed, epoch, step) and a resumed run reproduces it.
        torch.empty((), dtype=torch.int64).random_()
        store_sampler.set_epoch(abs_e, my_batches)
        batch_iter = iter(store_loader)

        bar = tqdm(my_batches, disable=not accelerator.is_main_process, desc=f"epoch {abs_e}")
        for _ in bar:
            with accelerator.accumulate(dit):
                cached_list = next(batch_iter)

                # Latents / embeddings / condition carry a leading batch dim of 1:
                # concatenate along dim 0 -> (B, ...).
                def _cat(key):
                    return torch.cat(
                        [c[key] for c in cached_list], dim=0
                    ).to(device=device, dtype=dtype, non_blocking=True)
                clean_latents = _cat("clean_latents")
                render_latents = (
                    _cat("render_latents")
                    if "render_latents" in cached_list[0] else None
                )
                prompt_embeds = _cat("prompt_embeds")
                image_embeds = _cat("image_embeds")
                condition = _cat("condition")
                # Per-frame action stream, (T, D) per clip -> (B, T, D).
                actions: Optional[torch.Tensor] = None
                if "actions" in cached_list[0]:
                    actions = torch.stack(
                        [c["actions"] for c in cached_list], dim=0
                    ).to(device=device, dtype=dtype, non_blocking=True)

                with torch.no_grad():
                    noise = torch.randn_like(clean_latents)
                    idx = torch.randint(0, len(scheduler.timesteps), (1,))
                    # Timesteps stay float32: Wan's flow-matching scheduler emits
                    # non-integer values, and casting them to bf16 or long would
                    # corrupt the sinusoidal time embedding and the noise scaling.
                    timestep_scalar = scheduler.timesteps[idx].to(device=device).float()
                    timestep_batch = timestep_scalar.expand(clean_latents.shape[0])
                    noisy_latents = scheduler.scale_noise(clean_latents, timestep_batch, noise)
                    # Latent frame 0 is the clean I2V conditioning frame.
                    noisy_latents[:, :, 0:1] = clean_latents[:, :, 0:1]

                latent_model_input = torch.cat([noisy_latents, condition], dim=1)
                target = noise - clean_latents     # rectified-flow velocity

                # Conditioning dropout: with probability cond_dropout_prob this step runs
                # without render latents and actions (the adapter output is zero), which
                # regularizes the adapter and provides an action-unconditioned branch.
                # The coin comes from a generator seeded by (seed, epoch, step), so every
                # rank takes the same branch.
                drop_cond_step = False
                if args.cond_dropout_prob > 0:
                    drop_g = torch.Generator()
                    drop_g.manual_seed(int(args.seed) * 1_000_003 + abs_e * 65521 + global_step)
                    drop_cond_step = bool(
                        torch.rand((1,), generator=drop_g).item() < args.cond_dropout_prob
                    )

                forward_kwargs = dict(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                )
                if not drop_cond_step:
                    if render_latents is not None:
                        forward_kwargs["render_latents"] = render_latents
                    if actions is not None:
                        forward_kwargs["actions"] = actions
                pred = dit(**forward_kwargs)[0]
                # Flow-matching MSE over the generated latent frames.
                loss_flow = F.mse_loss(pred[:, :, 1:].float(), target[:, :, 1:].float())

                # Non-finite guard, reduced across ranks so every rank skips (or runs)
                # the backward together; a partial backward would deadlock FSDP.
                loss_finite_all = accelerator.gather_for_metrics(
                    torch.isfinite(loss_flow.detach()).to(torch.float32).reshape(1)
                )
                loss_is_finite = bool(loss_finite_all.min().item() > 0.5)

                grad_norm = torch.tensor(float("nan"), device=device)
                if not loss_is_finite:
                    if accelerator.is_main_process:
                        accelerator.print(
                            f"[nan-guard] step {global_step}: non-finite loss "
                            f"({float(loss_flow):.4g}); skipping the step."
                        )
                    optimizer.zero_grad(set_to_none=True)
                else:
                    accelerator.backward(loss_flow)
                    # Global pre-clip grad norm. Under FSDP this must be the FSDP
                    # module's clip_grad_norm_: accelerator.clip_grad_norm_ delegates
                    # to FSDP only when given exactly model.parameters(), and on a
                    # parameter subset it would see only the local shard.
                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.FSDP:
                            gn = dit.clip_grad_norm_(args.max_grad_norm)
                        else:
                            gn = accelerator.clip_grad_norm_(
                                trainable_params, max_norm=args.max_grad_norm
                            )
                        if gn is not None:
                            grad_norm = gn.detach().to(device)
                            if torch.isfinite(grad_norm).item():
                                last_valid_grad_norm = float(grad_norm)
                    # Rank-consistent grad-finite decision on sync steps.
                    if accelerator.sync_gradients:
                        grad_finite_all = accelerator.gather_for_metrics(
                            torch.isfinite(grad_norm).reshape(1).to(torch.float32)
                        )
                        grad_finite = bool(grad_finite_all.min().item() > 0.5)
                    else:
                        grad_finite = True
                    if not grad_finite:
                        if accelerator.is_main_process:
                            accelerator.print(
                                f"[nan-guard] step {global_step}: non-finite grad_norm "
                                f"({float(grad_norm):.4g}); skipping optimizer.step."
                            )
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.step()
                        # accelerate turns optimizer.step() into a no-op on
                        # accumulation micro-steps but does not gate the standalone
                        # scheduler, so the scheduler and the EMA advance on sync
                        # steps only.
                        if accelerator.sync_gradients:
                            # opt_step == lr_sched.last_epoch == AdamW state['step'];
                            # skipped steps advance none of them.
                            opt_step += 1
                            if ema is not None:
                                _ema_first_step = not ema.initialised
                                ema.step()
                                if _ema_first_step:
                                    # Same step on every rank (sync_gradients is
                                    # collective), so this gather is safe here.
                                    _ema_check_ranks_agree(int(opt_step))
                        if lr_sched is not None and accelerator.sync_gradients:
                            lr_sched.step()
                        optimizer.zero_grad(set_to_none=True)

            global_step += 1

            flow_val = float(accelerator.gather_for_metrics(loss_flow.detach().reshape(1)).mean())
            if flow_ema is None:
                flow_ema = flow_val
            else:
                flow_ema = flow_ema_decay * flow_ema + (1.0 - flow_ema_decay) * flow_val
            t_idx = int(idx.item())
            b = min(3, max(0, sum(1 for e in bucket_edges[1:] if t_idx >= e)))
            bucket_sum[b] += flow_val
            bucket_cnt[b] += 1
            epoch_flow_sum += flow_val
            epoch_flow_cnt += 1

            if global_step % args.log_every_n_steps == 0 and accelerator.is_main_process:
                elapsed = time.time() - t0
                # On accumulation micro-steps grad_norm is NaN; show the last real one.
                gn_disp = (
                    float(grad_norm) if torch.isfinite(grad_norm).item() else last_valid_grad_norm
                )
                # scheduler.timesteps is descending: index 0 is the highest noise level.
                sigma_frac = 1.0 - float(t_idx) / max(1, n_t - 1)
                bar.set_postfix(
                    flow=f"{flow_val:.3f}",
                    ema=f"{flow_ema:.3f}",
                    sig=f"{sigma_frac:.2f}",
                    gn=f"{gn_disp:.3f}",
                    ok="Y" if loss_is_finite else "N",
                    step=global_step, t=f"{elapsed:.0f}s",
                )
                if wandb_run is not None:
                    log_payload = {
                        "train/loss_flow": flow_val,
                        "train/loss_flow_ema": float(flow_ema),
                        "train/grad_norm": gn_disp,
                        "train/loss_is_finite": float(loss_is_finite),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/timestep": float(timestep_scalar.item()),
                        "train/timestep_sigma_frac": sigma_frac,
                        "train/epoch": abs_e,
                        "train/elapsed_s": elapsed,
                        "train/drop_cond_step": float(drop_cond_step),
                    }
                    if ema is not None:
                        # decay applied at the last update: 0.0 while copying
                        # (initialisation / warmup), else ema_decay.
                        log_payload["train/ema_decay"] = float(ema.last_decay)
                        log_payload["train/ema_updates"] = float(ema.n_updates)
                    for k in range(4):
                        if bucket_cnt[k] > 0:
                            log_payload[f"train/flow_by_t_q{k}"] = bucket_sum[k] / bucket_cnt[k]
                    wandb_run.log(log_payload, step=global_step)

        epoch_mean = epoch_flow_sum / max(1, epoch_flow_cnt)
        if accelerator.is_main_process:
            accelerator.print(
                f"[epoch {abs_e}] mean flow_loss = {epoch_mean:.4f} "
                f"over {epoch_flow_cnt} steps  |  ema = {flow_ema:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log({"train/loss_flow_epoch_mean": epoch_mean,
                               "train/epoch_done": abs_e}, step=global_step)

        # ---- held-out validation (collective) ----
        if (
            embed_cache_val is not None
            and (
                (epoch + 1) % max(1, int(args.val_every_n_epochs)) == 0
                or epoch == args.num_epochs - 1
            )
        ):
            def _run_validation_pass():
                return _validation_pass(
                    dit=dit,
                    embed_cache_val=embed_cache_val,
                    scheduler=scheduler,
                    accelerator=accelerator,
                    device=device,
                    dtype=dtype,
                    n_timesteps_per_sample=int(args.val_timesteps),
                    seed=int(args.seed),
                    max_samples=int(args.val_n_samples),
                )

            # With ema_eval, val/* scores the EMA weights and val_raw/* the raw ones on
            # the same clips and noise. Each pass is bracketed by a reshard (see above).
            val_stats_raw = None
            if ema is not None and EMA_EVAL:
                _ema_reshard()
                val_stats_raw = _run_validation_pass()
                with ema.swap_in(reshard=_ema_reshard) as _swapped:
                    val_stats = _run_validation_pass() if _swapped else val_stats_raw
            else:
                val_stats = _run_validation_pass()
            if accelerator.is_main_process:
                if val_stats_raw is not None:
                    accelerator.print(
                        f"[val   {abs_e}] raw weights: loss_flow_mean = "
                        f"{val_stats_raw['loss_flow_mean']:.4f}  (EMA below; delta EMA-raw = "
                        f"{val_stats['loss_flow_mean'] - val_stats_raw['loss_flow_mean']:+.4f})"
                    )
                accelerator.print(
                    f"[val   {abs_e}] {'EMA ' if val_stats_raw is not None else ''}"
                    f"loss_flow_mean = {val_stats['loss_flow_mean']:.4f}  "
                    f"q0={val_stats['loss_flow_q0']:.4f}  q1={val_stats['loss_flow_q1']:.4f}  "
                    f"q2={val_stats['loss_flow_q2']:.4f}  q3={val_stats['loss_flow_q3']:.4f}  "
                    f"({val_stats['n_samples_evaluated']} samples x {args.val_timesteps} timesteps)"
                    + "".join(
                        f"  {name}={val_stats[f'loss_flow_{name}']:.4f}"
                        for name in VAL_DATASETS
                        if f"loss_flow_{name}" in val_stats
                    )
                )
                if wandb_run is not None:
                    payload = {f"val/{k}": v for k, v in val_stats.items() if k.startswith("loss_flow_")}
                    payload["val/epoch"] = abs_e
                    if val_stats_raw is not None:
                        payload.update({f"val_raw/{k}": v for k, v in val_stats_raw.items()
                                        if k.startswith("loss_flow_")})
                    wandb_run.log(payload, step=global_step)

        # ---- checkpoint ----
        if (epoch + 1) % args.save_every_n_epochs == 0 or epoch == args.num_epochs - 1:
            accelerator.wait_for_everyone()
            save_dir = os.path.join(args.output_dir, f"epoch_{abs_e}")
            if accelerator.is_main_process:
                os.makedirs(save_dir, exist_ok=True)

            # With ema_eval: conditioner.pt = EMA weights, conditioner_raw.pt = raw weights.
            write = _conditioner_writer(args, dit, accelerator)
            weights_kind = "raw"
            if ema is not None and EMA_EVAL:
                _ema_reshard()
                raw_state = accelerator.get_state_dict(dit)
                if accelerator.is_main_process:
                    write(raw_state, os.path.join(save_dir, "conditioner_raw.pt"))
                del raw_state
                with ema.swap_in(reshard=_ema_reshard) as _ema_swapped:
                    full_state = accelerator.get_state_dict(dit)
                weights_kind = "ema" if _ema_swapped else "raw"
            else:
                full_state = accelerator.get_state_dict(dit)
            if accelerator.is_main_process:
                write(full_state, os.path.join(save_dir, "conditioner.pt"))
                # meta.json: what inference needs to rebuild the model and the
                # preprocessing, with the action normalisation stats embedded.
                meta_blob: Dict[str, Any] = {
                    "conditioning": args.conditioning,
                    "redar_adapter_kwargs": args.redar_adapter_kwargs,
                    "action_only_adapter_kwargs": args.action_only_adapter_kwargs,
                    "action_field": args.action_field,
                    "action_stats_path": args.action_stats_path,
                    # > 0: actions were zero-extended to this width (data_multi.pad_actions);
                    # inference must pad the same way.
                    "action_pad_dim": int(args.action_pad_dim or 0),
                    # data_multi.load_actions unwraps the raw root axis-angle of these
                    # embodiments per clip before the z-score; inference must do the same.
                    "action_root_unwrap": True,
                    "action_root_rotvec": {rt: list(e["root_rotvec"]) for rt, e in EMBODIMENTS.items()
                                           if e.get("root_rotvec") is not None},
                    "num_frames": int(args.num_frames),
                    "height": int(args.height),
                    "width": int(args.width),
                    "ignore_prompts": bool(args.ignore_prompts),
                    "unfreeze_last_n_blocks": int(args.unfreeze_last_n_blocks),
                    "learning_rate": float(args.learning_rate),
                    "epoch": int(abs_e),
                    "global_step": int(global_step),
                    "opt_step": int(opt_step),
                    "resume_mode": str(args.resume_mode),
                    "total_epochs": int(args.total_epochs or 0),
                    "world_size": int(accelerator.num_processes),
                    "micro_pe": int(micro_pe),
                    # "ema": conditioner.pt holds the EMA weights; "raw" otherwise.
                    "weights": weights_kind,
                    "ema": ema.settings() if ema is not None else None,
                    "ema_eval": bool(EMA_EVAL),
                    "latent_store": {"dir": latent_store.dir, "spec": latent_store.spec},
                    "index": {"train": args.train_index, "val": args.val_index},
                    "sampler": {"dataset_weights": args.sampler_dataset_weights,
                                "epoch_len": int(args.epoch_len or 0)},
                }
                _emb: Dict[str, Any] = {}
                _sha: Dict[str, str] = {}
                for _rt, _p in (args.action_stats_path or {}).items():
                    _p = resolve_stats_path(_p, args.dataset_base_path)
                    if os.path.isfile(_p):
                        with open(_p, "rb") as f:
                            _b = f.read()
                        _emb[str(_rt)] = json.loads(_b)
                        _sha[str(_rt)] = hashlib.sha256(_b).hexdigest()
                meta_blob["action_stats"] = _emb
                meta_blob["action_stats_sha256"] = _sha
                with open(os.path.join(save_dir, "meta.json"), "w") as f:
                    json.dump(meta_blob, f, indent=2, default=str)
                if EMA_EVAL:
                    accelerator.print(f"[ema] conditioner.pt = {weights_kind.upper()} weights")
            del full_state

            # ---- resume state ----
            # Written after the weights and before any pruning, so an interrupted
            # write never leaves zero resumable epochs on disk.
            if int(args.save_optimizer_state or 0):
                accelerator.wait_for_everyone()
                _rs.save_resume_state(
                    accelerator, dit, optimizer, lr_sched, save_dir, args=args,
                    global_step=global_step, opt_step=opt_step, abs_e=abs_e,
                    opt_total=opt_total, micro_pe=int(micro_pe),
                    stats={"flow_ema": flow_ema, "bucket_sum": bucket_sum,
                           "bucket_cnt": bucket_cnt},
                    ema=ema,
                )
            elif ema is not None and ema.initialised:
                # No optimizer state requested, but the EMA is still resume state:
                # write this rank's shards (per rank, atomic, no collective).
                accelerator.wait_for_everyone()
                _rs.save_ema_shards(accelerator, ema, save_dir)
                accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                keep_n = int(args.keep_last_n_epochs or 0)
                if keep_n > 0:
                    eds = []
                    for name in os.listdir(args.output_dir):
                        m = re.match(r"^epoch_(\d+)$", name)
                        if m and os.path.isfile(os.path.join(
                                args.output_dir, name, "conditioner.pt")):
                            eds.append((int(m.group(1)), name))
                    eds.sort()
                    for _, name in eds[:-keep_n]:
                        victim = os.path.join(args.output_dir, name)
                        shutil.rmtree(victim, ignore_errors=True)
                        accelerator.print(f"[rotate] removed {victim} (keep_last_n_epochs={keep_n})")
                _rs.prune_optim_state(
                    args.output_dir, int(args.keep_optim_state_last_n or 0), accelerator.print,
                )
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print(f"Training complete. {global_step} steps in {time.time() - t0:.0f}s")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
