"""Helpers for the FSDP trainer (:mod:`doppel.train`): step counting, per-module
learning-rate groups, the warmup + cosine schedule, run-directory setup and
Wan checkpoint resolution."""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def fsdp_micro_steps_per_epoch(n_samples: int, world_size: int, batch_size: int = 1) -> int:
    """Micro-steps (forward/backward passes) each rank runs per epoch.

    The epoch's index list is padded to a multiple of ``world_size * batch_size``
    and strided per rank, so each rank runs ``Lp / (world_size * batch_size)``
    micro-batches.
    """
    L = int(n_samples)
    ws = max(1, int(world_size))
    bs = max(1, int(batch_size))
    Lp = L + ((-L) % (ws * bs))
    return Lp // (ws * bs)


def optimizer_step_count(micro_steps_total: int, gradient_accumulation_steps: int) -> int:
    """Number of ``optimizer.step()`` calls implied by micro-steps and grad accumulation."""
    ga = max(1, int(gradient_accumulation_steps))
    mt = max(0, int(micro_steps_total))
    return max(1, math.ceil(mt / ga))


def _mul_cosine_1_to_min_then_plateau(step: int, n_cosine: int, min_r: float) -> float:
    """
    Multiplicative LR factor in [1.0, min_r]: full LR at the first step of the segment,
    ``min_r`` at the end of a cosine, then constant ``min_r`` (no periodic restart).
    ``n_cosine`` = number of scheduler steps the cosine part spans (>=1).
    """
    s, n = int(step), int(n_cosine)
    if n <= 1 or min_r >= 0.999999:
        return float(min_r) if s >= 0 else 1.0
    if s >= n:
        return float(min_r)
    u = float(s) / float(max(1, n - 1))
    c = (1.0 + math.cos(math.pi * u)) / 2.0
    return min_r + (1.0 - min_r) * c


def _mul_warmup_then_cosine_plateau(step: int, warmup: int, total: int, min_r: float) -> float:
    """
    Linear warmup from 1e-6 to 1.0 over the first ``warmup`` steps (none when
    ``warmup`` is 0), then cosine from 1.0 to ``min_r`` over the remaining
    ``total - warmup`` steps, then constant ``min_r`` (avoids ``CosineAnnealingLR``'s
    post-``T_max`` restart).
    """
    s, w, tot = int(step), int(warmup), int(total)
    if w > 0 and s < w:
        return 1e-6 + (1.0 - 1e-6) * (float(s) / float(max(1, w - 1)))
    t = s - w
    cos_n = max(0, tot - w)
    if cos_n <= 0:
        return 1.0
    return _mul_cosine_1_to_min_then_plateau(t, cos_n, min_r)


# Submodules of the DiT that come from the pretrained Wan2.1 checkpoint.
# Every other trainable parameter belongs to a newly introduced conditioning
# module (the render-action adapter or the action-only adapter).
WAN_PRETRAINED_PREFIXES = (
    "blocks.",
    "patch_embedding.",
    "condition_embedder.",
    "norm_out.",
    "proj_out.",
    "rope.",
    "scale_shift_table",
)


def is_pretrained_param(name: str) -> bool:
    """True if ``name`` is a parameter of the pretrained Wan backbone."""
    return name.startswith(WAN_PRETRAINED_PREFIXES)


def param_lr_for(name: str, base_lr: float, conditioner_mult: float = 1.0) -> float:
    """LR for one parameter: ``base_lr`` for the pretrained backbone and
    ``base_lr * conditioner_mult`` for the newly introduced conditioning modules."""
    lr = float(base_lr)
    if not is_pretrained_param(name):
        lr *= float(conditioner_mult)
    return lr


def build_param_lr_groups(named_trainable_params, base_lr: float, conditioner_mult: float = 1.0):
    """Group trainable ``(name, param)`` pairs into AdamW param groups by LR.

    Returns ``(param_groups, info)``. ``param_groups`` is ordered by ascending
    LR and is safe to hand to ``build_lr_scheduler``: LambdaLR scales every
    group off its own ``base_lr``, so the ratio between groups is preserved
    through warmup and the whole cosine. ``info`` carries the numbers worth
    logging.
    """
    by_lr: Dict[float, list] = {}
    cond_names: list = []
    cond_numel = 0
    base_numel = 0
    for name, prm in named_trainable_params:
        lr = param_lr_for(name, base_lr, conditioner_mult)
        by_lr.setdefault(lr, []).append(prm)
        if is_pretrained_param(name):
            base_numel += prm.numel()
        else:
            cond_names.append(name)
            cond_numel += prm.numel()
    param_groups = [{"params": ps, "lr": lr} for lr, ps in sorted(by_lr.items())]
    info = {
        "base_numel": base_numel,
        "conditioner_numel": cond_numel,
        "conditioner_tensors": len(cond_names),
        "conditioner_roots": sorted({n.split(".")[0] for n in cond_names}),
        "group_lrs": [g["lr"] for g in param_groups],
        "group_sizes": [len(g["params"]) for g in param_groups],
    }
    return param_groups, info


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: str,
    warmup_steps: int,
    min_lr_ratio: float,
    total_optimizer_steps: int,
) -> Optional[LRScheduler]:
    """LR schedule over ``total_optimizer_steps`` optimizer steps.

    ``cosine_warmup``: linear warmup over ``warmup_steps``, then cosine from the
    base LR to ``min_lr_ratio * base_lr``, then constant at that floor.

    ``cosine``: the same without warmup.

    ``constant``: returns ``None`` (the caller keeps the fixed learning rate).
    """
    mode = (lr_scheduler or "constant").strip().lower()
    if mode in ("constant", "none", ""):
        return None
    if mode in ("cosine", "cosine_warmup"):
        total = max(1, int(total_optimizer_steps))
        w = 0
        if mode == "cosine_warmup":
            w = max(0, min(int(warmup_steps), max(0, total - 1)))
        min_r = float(min_lr_ratio)
        if not (0.0 <= min_r < 1.0):
            raise ValueError(f"{mode}: lr_min_ratio should be in [0, 1), got {min_r}")
        return LambdaLR(
            optimizer,
            lambda s: _mul_warmup_then_cosine_plateau(int(s), w, total, min_r),
        )
    raise ValueError(
        f"Unknown lr_scheduler={lr_scheduler!r}; expected one of: constant, cosine, cosine_warmup."
    )


def _setup_run_dir_and_logging(args, accelerator):
    """Per-run output layout::

        args.output_dir / run_<YYYYMMDD_HHMMSS> /
            train.log     # rank-0 stdout+stderr, line-buffered
            args.json     # resolved arguments
            epoch_<N>/    # checkpoints
            wandb/        # wandb local run dir (if enabled)

    Mutates ``args.output_dir`` in place so checkpoints and the wandb dir land
    in the run-specific subdir, and sets ``args.wandb_run_name`` to the run id
    if unset. The log tee runs on rank 0 only.
    """
    import datetime
    import json as _json
    import sys

    run_dir = None
    if accelerator.is_main_process:
        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "train.log")

        class _Tee:
            """Duplicate writes to multiple streams; delegate isatty/fileno/etc.
            to the first (live terminal) stream so tqdm still animates."""
            def __init__(self, *streams):
                self.streams = streams
            def write(self, s):
                for st in self.streams:
                    try:
                        st.write(s)
                        st.flush()
                    except Exception:
                        pass
                return len(s) if isinstance(s, str) else 0
            def flush(self):
                for st in self.streams:
                    try:
                        st.flush()
                    except Exception:
                        pass
            def __getattr__(self, name):
                return getattr(self.streams[0], name)

        log_f = open(log_path, "a", buffering=1, encoding="utf-8")
        log_f.write(f"\n===== run start: {run_id} =====\n")
        sys.stdout = _Tee(sys.__stdout__, log_f)
        sys.stderr = _Tee(sys.__stderr__, log_f)

        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
            _json.dump(vars(args), f, indent=2, default=str)

        if not getattr(args, "wandb_run_name", None):
            args.wandb_run_name = run_id

    # Broadcast the run dir + wandb run name from rank 0 so every rank writes
    # checkpoints to the same place (skipped in a single-process run).
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and accelerator.num_processes > 1:
            obj = [run_dir, getattr(args, "wandb_run_name", None)]
            dist.broadcast_object_list(obj, src=0, device=accelerator.device)
            run_dir = obj[0]
            args.wandb_run_name = obj[1]
    except Exception:
        pass

    if run_dir:
        args.output_dir = run_dir

    if accelerator.is_main_process:
        accelerator.print(f"[run] output_dir = {args.output_dir}")
        accelerator.print(f"[run] log_file   = {os.path.join(args.output_dir, 'train.log')}")
    return args


def resolve_wan_model_root(model_path: str, revision: Optional[str] = None) -> str:
    """Resolve ``model_path`` to a local directory.

    A local dir is returned as is. A HuggingFace repo id (e.g.
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers") is resolved to its snapshot in the
    local HF cache, so ``from_pretrained`` never needs network access. Falls
    back to the original string when the snapshot is not cached. ``revision`` pins
    the snapshot (inference passes the revision the released models were trained
    from) and is downloaded if it is not cached.
    """
    root = str(model_path).rstrip("/")
    if os.path.isdir(root):
        return root
    try:
        from huggingface_hub import snapshot_download
        try:
            local = snapshot_download(root, revision=revision, local_files_only=True)
        except Exception:
            if revision is None:
                raise
            local = snapshot_download(root, revision=revision)
        print(f"[resolve] HF repo id -> local snapshot: {local}")
        return local
    except Exception as exc:  # noqa: BLE001
        print(f"[resolve] WARN: could not resolve {root!r} to a local snapshot "
              f"({exc}); falling back to the repo id")
        return root
