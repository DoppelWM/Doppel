"""Checkpoint resume state for the FSDP trainer.

``--resume_mode``:

* ``warm``: weights only; fresh optimizer and LR schedule over this launch.
* ``schedule``: weights + LR schedule position + ``global_step``; AdamW moments restart.
* ``continue``: weights, fp32 masters, AdamW moments, scheduler, ``global_step``,
  per-rank RNG and logging accumulators; needs the same world size.

The model state is loaded before the optimizer state: until the first forward or
state-dict call the FSDP parameter views are still bf16, and touching the
optimizer first would create bf16 moments (checked by :func:`assert_fp32_views`).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import glob
import shutil
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

RESUME_STATE_FILE = "resume_state.json"
RESUME_STATE_SCHEMA = 1
RESUME_DCP_DIR = "optim"
RESUME_DCP_THREADS = 8

_CK_KEYS = ("params", "frozen", "exp_avg", "exp_avg_sq", "step", "n_state")
# Hyperparameters that define the LR curve. A resume that changes any of them
# aborts unless --allow_schedule_mismatch is given.
_SCHED_KEYS = (
    "opt_total", "micro_pe", "grad_accum", "lr_scheduler", "lr_warmup_steps",
    "lr_min_ratio", "learning_rate", "conditioner_lr_multiplier", "weight_decay",
    "scheduler_kind",
)


# --------------------------------------------------------------------------- small helpers
def inner_optimizer(opt):
    """``AcceleratedOptimizer`` -> the wrapped ``torch.optim.AdamW``.

    The public DSD entry points isinstance-check ``torch.optim.Optimizer``, so the
    accelerate wrapper must be unwrapped before they are called.
    """
    return getattr(opt, "optimizer", opt)


def dcp_paths(state_dir: str, dcp_dir: str = RESUME_DCP_DIR):
    base = os.path.join(state_dir, dcp_dir)
    return os.path.join(base, "model"), os.path.join(base, "optim")


def has_state(state_dir: str) -> bool:
    """True when *state_dir* holds a complete resume state (marker + both shard sets)."""
    if not os.path.isfile(os.path.join(state_dir, RESUME_STATE_FILE)):
        return False
    mdir, _ = dcp_paths(state_dir)
    if not os.path.isfile(os.path.join(mdir, ".metadata")):
        return False
    return bool(glob.glob(_local_optim_path(state_dir, 0).replace("rank0", "rank*")))


def read_resume_state(state_dir: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(state_dir, RESUME_STATE_FILE)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    if int(st.get("schema", -1)) != RESUME_STATE_SCHEMA:
        raise SystemExit(
            f"[resume-state] {p}: schema {st.get('schema')} != {RESUME_STATE_SCHEMA}"
        )
    st["_dir"] = state_dir
    st["_has_dcp"] = has_state(state_dir)
    return st


# --------------------------------------------------------------------------- RNG
def rng_blob(device) -> Dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(blob: Mapping[str, Any], device) -> None:
    torch.set_rng_state(blob["torch_cpu"])
    if blob.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(blob["torch_cuda"], device)
    np.random.set_state(blob["numpy"])
    random.setstate(blob["python"])


# --------------------------------------------------------------------------- invariants
def assert_fp32_views(dit, where: str) -> None:
    """Check that every trainable view is fp32 (see the module docstring)."""
    bad = [(n, str(p.dtype)) for n, p in dit.named_parameters()
           if p.requires_grad and p.numel() and p.dtype is not torch.float32]
    if bad:
        raise SystemExit(
            f"[resume-state] {where}: {len(bad)} trainable params are not fp32 "
            f"(e.g. {bad[:3]}); call get_model_state_dict (or run one forward) "
            "before touching the optimizer state."
        )


def state_checksums(accelerator, dit, inner_opt) -> Dict[str, float]:
    """Canonical fp64 per-key totals over all ranks.

    Order-independent on purpose: a load reorders the optimizer state dict
    (FSDP's ``_flatten_optim_state_dict`` iterates ``named_parameters()`` order
    while the saved dict is in param-group insertion order), so a plain ``sum()``
    over ``state.values()`` would not be stable across a round trip.
    """
    dev = accelerator.device
    tot = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in _CK_KEYS}
    for p in dit.parameters():
        if p.numel():
            tot["params" if p.requires_grad else "frozen"] += p.detach().double().sum().to(dev)
    for g in inner_opt.param_groups:
        for p in g["params"]:
            st = inner_opt.state.get(p)
            if not st:
                continue
            tot["n_state"] += 1.0
            for k in sorted(st):
                v = st[k]
                if not torch.is_tensor(v) or not v.numel():
                    continue
                if k not in tot:
                    raise SystemExit(
                        f"[resume-state] unexpected AdamW state key {k!r} "
                        "(expected step/exp_avg/exp_avg_sq)"
                    )
                tot[k] += v.detach().double().sum().to(dev)
    stacked = torch.stack([tot[k] for k in _CK_KEYS]).reshape(1, len(_CK_KEYS))
    g = accelerator.gather(stacked).sum(dim=0)
    return {k: float(g[i]) for i, k in enumerate(_CK_KEYS)}


def compare_checksums(now: Mapping[str, float], saved: Mapping[str, float], log) -> None:
    bad = []
    for k in _CK_KEYS:
        if k not in saved:
            continue
        a, b = float(now[k]), float(saved[k])
        ok = (a == b) if k == "n_state" else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
        log(f"[resume-state]   checksum {k:>10}: {a!r} (saved {b!r}) {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(k)
    if bad:
        raise SystemExit(f"[resume-state] checksum mismatch after load: {bad}")


# --------------------------------------------------------------------------- DCP save / load
def _local_optim_path(state_dir: str, rank: int, dcp_dir: str = RESUME_DCP_DIR) -> str:
    return os.path.join(state_dir, dcp_dir, "optim", f"local_rank{rank}.pt")


def _save_local_optim(inner_opt, state_dir: str, rank: int, dcp_dir: str = RESUME_DCP_DIR) -> None:
    """Save this rank's optimizer shard with no collective.

    ``get_optimizer_state_dict`` routes through ``FSDP.optim_state_dict`` ->
    ``_allgather_orig_param_states``, which materialises each parameter's state
    unsharded on the GPU before honouring ``cpu_offload``; with fp32 masters and
    AdamW moments resident that does not fit on an 80 GiB card. Under
    ``use_orig_params=True`` the entries in ``inner_opt.state`` already are this
    rank's shards, so writing them directly is correct and needs no extra device
    memory. The state cannot be resharded onto a different world size, which
    ``continue`` refuses anyway (``resolve_resume_mode``); ``schedule`` covers
    that case.
    """
    p = _local_optim_path(state_dir, rank, dcp_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    sd = inner_opt.state_dict()
    # Move to CPU one tensor at a time: peak extra device memory is one tensor.
    cpu_state = {}
    for k, st in sd.get("state", {}).items():
        cpu_state[k] = {
            kk: (vv.detach().to("cpu", copy=True) if torch.is_tensor(vv) else vv)
            for kk, vv in st.items()
        }
    blob = {"state": cpu_state,
            "param_groups": sd.get("param_groups"),
            "rank": int(rank)}
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(blob, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    del cpu_state, blob, sd


def _load_local_optim(inner_opt, state_dir: str, rank: int, dcp_dir: str = RESUME_DCP_DIR) -> None:
    p = _local_optim_path(state_dir, rank, dcp_dir)
    if not os.path.isfile(p):
        raise SystemExit(f"[resume-state] missing optimizer shard for rank {rank}: {p}")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    inner_opt.load_state_dict({"state": blob["state"], "param_groups": blob["param_groups"]})
    del blob


# --------------------------------------------------------------------------- #
# torch 2.5: ShardedTensor.is_meta raises on a rank that owns no shard (parameters
# with size(0) < world_size, e.g. the per-block scale_shift_table), which breaks
# get_model_state_dict. Register an is_meta that returns False there; dropping the
# key instead would skip a collective in the sharded load and hang.
IS_META_GETTER = torch.Tensor.is_meta.__get__  # the key torch itself registers


def _st_is_meta_shardless_ok(types, args=(), kwargs=None, pg=None):
    """``ShardedTensor.is_meta`` that tolerates a rank owning no local shard."""
    shards = args[0].local_shards()
    return bool(shards[0].tensor.is_meta) if shards else False


@contextlib.contextmanager
def shardless_is_meta_ok():
    """Scope the ``ShardedTensor.is_meta`` override to one checkpoint phase.

    Restores the previous registration on the way out, so nothing leaks into the
    training step even if the body raises.
    """
    from torch.distributed._shard.sharded_tensor.api import _CUSTOM_SHARDED_OPS
    sentinel = object()
    prev = _CUSTOM_SHARDED_OPS.get(IS_META_GETTER, sentinel)
    _CUSTOM_SHARDED_OPS[IS_META_GETTER] = _st_is_meta_shardless_ok
    try:
        yield
    finally:
        if prev is sentinel:
            _CUSTOM_SHARDED_OPS.pop(IS_META_GETTER, None)
        else:
            _CUSTOM_SHARDED_OPS[IS_META_GETTER] = prev


def dcp_save(dit, inner_opt, state_dir: str, dcp_dir: str = RESUME_DCP_DIR, rank: int = 0) -> None:
    """Collective: every rank must call this.

    Phase 1 (model, via DCP) runs first: ``get_model_state_dict`` refreshes the
    orig-param views to fp32, which everything below relies on. Phase 2
    (optimizer) is a per-rank local write, see ``_save_local_optim``.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict,
    )
    mdir, _ = dcp_paths(state_dir, dcp_dir)
    with shardless_is_meta_ok():
        msd = get_model_state_dict(
            dit,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True, ignore_frozen_params=True),
        )
        dcp.save({"model": msd}, storage_writer=dcp.FileSystemWriter(mdir, thread_count=RESUME_DCP_THREADS))
    del msd
    assert_fp32_views(dit, "save")
    _save_local_optim(inner_opt, state_dir, rank, dcp_dir)


def dcp_load(dit, inner_opt, state_dir: str, *, expect_step=None,
             dcp_dir: str = RESUME_DCP_DIR, log=print, rank: int = 0) -> None:
    """Collective. Call after ``accelerator.prepare`` and before the first backward."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, set_model_state_dict,
    )
    if any(p.grad is not None for p in dit.parameters()):
        raise SystemExit(
            "[resume-state] dcp_load: some params already have .grad; _init_optim_state "
            "would return early and the optimizer template would be empty"
        )
    if inner_opt.state:
        raise SystemExit(
            "[resume-state] dcp_load: optimizer state is not empty (load must run before the first step)"
        )
    mdir, _ = dcp_paths(state_dir, dcp_dir)
    with shardless_is_meta_ok():
        msd = get_model_state_dict(
            dit,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True, ignore_frozen_params=True),
        )
        dcp.load({"model": msd}, storage_reader=dcp.FileSystemReader(mdir))
        # strict=False: the frozen FQNs of the root FSDP unit are not saved.
        set_model_state_dict(
            dit, msd,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True, strict=False),
        )
    del msd
    assert_fp32_views(dit, "load/after model")
    # Build the optimizer state template on this rank (zero grads + an lr=0 step),
    # then overwrite it with this rank's saved shard. The views are fp32 by now, so
    # the moments are created fp32 and load_state_dict does not round them.
    if not inner_opt.state:
        from torch.distributed.checkpoint.state_dict import _init_optim_state as _ios
        _ios(inner_opt)
    _load_local_optim(inner_opt, state_dir, rank, dcp_dir)

    n_tr = sum(1 for g in inner_opt.param_groups for p in g["params"]
               if p.requires_grad and p.numel())
    if len(inner_opt.state) != n_tr:
        raise SystemExit(
            f"[resume-state] optimizer state has {len(inner_opt.state)} entries, "
            f"expected {n_tr} local trainable params"
        )
    dts = sorted({str(v.dtype) for st in inner_opt.state.values()
                  for k, v in st.items() if k != "step" and torch.is_tensor(v)})
    if dts != ["torch.float32"]:
        raise SystemExit(f"[resume-state] AdamW moments are {dts}, expected fp32")
    steps = sorted({float(st["step"]) for st in inner_opt.state.values() if "step" in st})
    if expect_step is not None and steps != [float(expect_step)]:
        raise SystemExit(
            f"[resume-state] AdamW step values {steps[:4]} != saved opt_step {expect_step}"
        )
    log(f"[resume-state] optimizer restored: {len(inner_opt.state)} tensors, "
        f"moments fp32, step={steps[:1]}")


def quiet_fsdp_load_warnings(log) -> None:
    """Frozen root-unit FQNs are not part of a ``continue`` checkpoint."""
    logging.getLogger("torch.distributed.fsdp._state_dict_utils").setLevel(logging.ERROR)
    log("[resume-state] frozen params are not saved in the resume state; FSDP's "
        "'Did not find param with FQN' warnings are suppressed (the frozen checksum "
        "verifies them).")


# --------------------------------------------------------------------------- scheduler
def scheduler_blob(lr_sched):
    """``LambdaLR`` state dicts are JSON-safe once lambdas are stringified."""
    if lr_sched is None:
        return None
    return json.loads(json.dumps(lr_sched.state_dict(), default=str))


def schedule_now(args, *, opt_total, micro_pe, lr_sched) -> Dict[str, Any]:
    return {
        "opt_total": int(opt_total), "micro_pe": int(micro_pe),
        "grad_accum": int(args.gradient_accumulation_steps),
        "lr_scheduler": str(args.lr_scheduler),
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "lr_min_ratio": float(args.lr_min_ratio),
        "learning_rate": float(args.learning_rate),
        "conditioner_lr_multiplier": float(getattr(args, "conditioner_lr_multiplier", 1.0)),
        "weight_decay": float(args.weight_decay),
        "scheduler_kind": type(lr_sched).__name__ if lr_sched is not None else None,
    }


def reassert_optimizer_hparams(inner_opt, args, log) -> None:
    """``load_state_dict`` copies hyperparameters from the checkpoint's param
    groups. ``lr`` is owned by the scheduler; re-assert the rest from current args."""
    betas = tuple(inner_opt.defaults.get("betas", (0.9, 0.999)))
    for gi, g in enumerate(inner_opt.param_groups):
        for k, want in (("weight_decay", float(args.weight_decay)), ("betas", betas)):
            if k in g and g[k] != want:
                log(f"[resume-state] param_group[{gi}].{k}: checkpoint {g[k]} -> current args {want}")
                g[k] = want


def fast_forward_scheduler(lr_sched, inner_opt, st, *, now, allow_mismatch, log) -> int:
    """Move a freshly built scheduler to the saved position. Returns ``opt_step``."""
    saved_sched = (st or {}).get("schedule") or {}
    opt_step = int((st or {}).get("opt_step", 0))
    if saved_sched:
        diff = {k: (saved_sched.get(k), now.get(k)) for k in _SCHED_KEYS
                if k in saved_sched and saved_sched.get(k) != now.get(k)}
        if diff:
            msg = "[resume-state] schedule hyperparameters differ from the checkpoint: " + ", ".join(
                f"{k}: {a} -> {b}" for k, (a, b) in sorted(diff.items()))
            if not allow_mismatch:
                raise SystemExit(
                    msg + "\n  Pass --allow_schedule_mismatch to rebuild the LR closure from the "
                    "current args at the same relative position, or fix the config."
                )
            log(msg + "  (--allow_schedule_mismatch: rebuilding from current args)")
            old_total = int(saved_sched.get("opt_total") or 0)
            new_total = int(now.get("opt_total") or 0)
            if old_total and new_total and old_total != new_total:
                opt_step = int(round(opt_step * new_total / old_total))
                log(f"[resume-state] rescaled position to opt_step={opt_step} "
                    f"(opt_total {old_total} -> {new_total})")
    if lr_sched is None:
        return opt_step
    blob = (st or {}).get("scheduler_state")
    restored = False
    if blob and not allow_mismatch:
        try:
            blob = dict(blob)
            # lr_lambdas were saved as strings; keep the closures built from args
            if "lr_lambdas" in blob:
                blob["lr_lambdas"] = [None] * len(blob["lr_lambdas"])
            lr_sched.load_state_dict(blob)
            restored = True
        except Exception as exc:  # noqa: BLE001 - fall back to stepping
            log(f"[resume-state] WARN could not load scheduler state ({exc}); stepping instead")
    if not restored:
        for _ in range(max(0, opt_step)):
            lr_sched.step()
    # LambdaLR.load_state_dict is a __dict__.update and does not write param_group['lr'].
    for g, lr in zip(inner_opt.param_groups, lr_sched.get_last_lr()):
        g["lr"] = lr
    log(f"[resume-state] scheduler at opt_step={opt_step} "
        f"(last_epoch={getattr(lr_sched, 'last_epoch', None)}) lr={lr_sched.get_last_lr()}")
    return opt_step


# --------------------------------------------------------------------------- mode resolution
def resolve_resume_mode(args, accelerator=None, log=print):
    """Validate the flags and locate the state dir. Returns the resume state or None."""
    mode = str(getattr(args, "resume_mode", "warm") or "warm")
    if mode not in ("warm", "schedule", "continue"):
        raise SystemExit(f"[resume-state] --resume_mode {mode!r} must be warm|schedule|continue")
    ckpt = getattr(args, "resume_ckpt", None)
    if getattr(args, "save_optimizer_state", None) is None:
        args.save_optimizer_state = 0 if mode == "warm" else 1
    if getattr(args, "total_epochs", None) in (None, 0):
        args.total_epochs = int(args.epoch_offset) + int(args.num_epochs)
    if not ckpt:
        if mode != "warm":
            log(f"[resume-state] --resume_mode {mode} with no --resume_ckpt: fresh start "
                f"(schedule horizon total_epochs={args.total_epochs})")
        return None
    state_dir = str(getattr(args, "resume_state_dir", None) or "").strip() or os.path.dirname(str(ckpt))
    st = read_resume_state(state_dir)
    if st is None:
        if mode != "warm":
            raise SystemExit(
                f"[resume-state] --resume_mode {mode} but {state_dir} has no {RESUME_STATE_FILE}.\n"
                "  That checkpoint was written without optimizer state; resume it with\n"
                "  --resume_mode warm (weights only)."
            )
        # warm: weights only; the dir is still searched for saved EMA shards.
        return {"_dir": state_dir, "global_step": 0, "opt_step": 0,
                "scheduler_state": None, "_has_dcp": False, "world_size": None}
    if mode == "continue":
        if not st.get("_has_dcp"):
            raise SystemExit(
                f"[resume-state] {state_dir}/{RESUME_STATE_FILE} exists but the sharded state is "
                "missing or incomplete (wanted optim/model/.metadata and at least one\n"
                "  optim/optim/local_rank*.pt -- see has_state()). Use --resume_mode schedule."
            )
        ws = st.get("world_size")
        cur = int(accelerator.num_processes) if accelerator is not None else None
        if ws is not None and cur is not None and int(ws) != cur:
            raise SystemExit(
                f"[resume-state] checkpoint was saved with world_size={ws} but this run has {cur}.\n"
                "  Sharded optimizer state is not resharded here. Relaunch with the same GPU count,\n"
                "  or use --resume_mode schedule (weights + exact LR position, fresh moments)."
            )
    return st


# --------------------------------------------------------------------------- EMA shards
def save_ema_shards(accelerator, ema, state_dir: str, dcp_dir: str = RESUME_DCP_DIR) -> str:
    """Write this rank's EMA shadow shard next to its optimizer shard
    (``<state_dir>/optim/ema/local_rank<r>.pt``). Per rank, atomic, no collective;
    the caller places the barriers. Same world-size caveat as ``_save_local_optim``."""
    from doppel.ema import ema_shard_path
    p = ema_shard_path(state_dir, int(accelerator.process_index), dcp_dir)
    t0 = time.time()
    ema.save_shards(p, rank=int(accelerator.process_index),
                    world_size=int(accelerator.num_processes))
    ws = int(accelerator.num_processes)
    rank0_gib = ema.numel() * 4 / 2**30
    accelerator.print(
        f"[ema] saved {os.path.dirname(p)}/local_rank*.pt "
        f"({rank0_gib:.1f} GiB fp32 on rank 0, ~{rank0_gib * ws:.1f} GiB on disk across "
        f"{ws} ranks; rotated by --keep_optim_state_last_n; {time.time() - t0:.0f}s)"
    )
    return p


# --------------------------------------------------------------------------- save
def save_resume_state(accelerator, dit, optimizer, lr_sched, save_dir, *, args,
                      global_step, opt_step, abs_e, opt_total, micro_pe,
                      stats=None, ema=None) -> None:
    """Write sharded model/optimizer state + per-rank RNG + the ``resume_state.json``
    completion marker.

    Collective. ``resume_state.json`` is written last, after a barrier, so an
    interrupted write never leaves a state dir that looks resumable. *ema* (a
    ``ShardEMA`` or None) has its per-rank shards written under ``optim/ema/``
    and its settings recorded in the marker.
    """
    t0 = time.time()
    inner = inner_optimizer(optimizer)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    ck = state_checksums(accelerator, dit, inner)
    dcp_save(dit, inner, save_dir, rank=int(accelerator.process_index))
    if ema is not None and ema.initialised:
        save_ema_shards(accelerator, ema, save_dir, RESUME_DCP_DIR)
    torch.save(rng_blob(accelerator.device),
               os.path.join(save_dir, f"rng_rank{accelerator.process_index}.pt"))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        blob = {
            "schema": RESUME_STATE_SCHEMA,
            "dcp_dir": RESUME_DCP_DIR,
            "global_step": int(global_step),
            "opt_step": int(opt_step),
            "abs_epoch": int(abs_e),
            "world_size": int(accelerator.num_processes),
            "resume_mode_written_by": str(getattr(args, "resume_mode", "warm")),
            "total_epochs": int(getattr(args, "total_epochs", 0) or 0),
            "schedule": schedule_now(args, opt_total=opt_total, micro_pe=micro_pe, lr_sched=lr_sched),
            "scheduler_state": scheduler_blob(lr_sched),
            "checksum": ck,
            "stats": stats or {},
            # EMA settings + counters (None when --ema_decay is 0). The shards
            # themselves are optim/ema/local_rank*.pt, pruned with the rest.
            "ema": ema.settings() if ema is not None else None,
        }
        tmp = os.path.join(save_dir, RESUME_STATE_FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(save_dir, RESUME_STATE_FILE))
        accelerator.print(
            f"[resume-state] saved {save_dir} ({time.time() - t0:.0f}s, "
            f"opt_step={opt_step}, global_step={global_step})"
        )
    accelerator.wait_for_everyone()


# --------------------------------------------------------------------------- pruning
def _epoch_dirs(output_dir: str):
    out = []
    if not os.path.isdir(output_dir):
        return out
    for name in os.listdir(output_dir):
        if not name.startswith("epoch_"):
            continue
        try:
            n = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((n, os.path.join(output_dir, name)))
    return sorted(out)


def _tree_size(p: str) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def prune_optim_state(output_dir: str, keep_n: int, log=print) -> None:
    """Drop sharded state + RNG + marker from all but the newest *keep_n* epoch dirs.

    ``conditioner.pt`` and ``meta.json`` are never touched: weights are kept
    by ``keep_last_n_epochs``, while the (far larger) optimizer state is usually
    only needed for the newest epochs.

    An epoch dir without a marker is left alone except for ``optim/ema/``: with
    ``save_optimizer_state 0`` the trainer writes only the EMA shards (no marker),
    and those rotate with the same ``keep_n``.
    """
    if keep_n <= 0:
        return
    dirs = _epoch_dirs(output_dir)
    for _, d in dirs[:-keep_n] if len(dirs) > keep_n else []:
        if not os.path.isfile(os.path.join(d, RESUME_STATE_FILE)):
            ema_dir = os.path.join(d, RESUME_DCP_DIR, "ema")
            if os.path.isdir(ema_dir):
                freed = _tree_size(ema_dir)
                shutil.rmtree(ema_dir, ignore_errors=True)
                try:                                    # optim/ is empty now -> drop it
                    os.rmdir(os.path.join(d, RESUME_DCP_DIR))
                except OSError:
                    pass
                log(f"[resume-state] pruned EMA shards from {os.path.basename(d)} "
                    f"({freed / 2**30:.1f} GiB freed; keep_optim_state_last_n={keep_n})")
            continue
        freed = 0
        p = os.path.join(d, RESUME_DCP_DIR)
        if os.path.isdir(p):
            freed += _tree_size(p)
            shutil.rmtree(p, ignore_errors=True)
        for fn in os.listdir(d):
            if fn.startswith("rng_rank") and fn.endswith(".pt"):
                try:
                    os.remove(os.path.join(d, fn))
                except OSError:
                    pass
        try:
            os.remove(os.path.join(d, RESUME_STATE_FILE))
        except OSError:
            pass
        log(f"[resume-state] pruned optimizer state from {os.path.basename(d)} "
            f"({freed / 2**30:.1f} GiB freed; keep_optim_state_last_n={keep_n})")
