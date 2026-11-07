"""EMA of the trainable weights, kept per rank over the rank's own FSDP shard.

With FSDP1 and ``use_orig_params=True`` each trainable parameter is a view of this
rank's slice of the flat parameter, so a per-rank EMA needs no collectives and
swapping it in is a local copy; the next all-gather assembles the EMA weights.
The shadow is created at the first optimizer step (before the first forward the
views are still bf16). Shards are saved per rank and cannot be resharded.

FSDP1 keeps the root unit's unsharded flat parameter after a forward and skips the
next all-gather, so :func:`free_unsharded_flat_params` must be called around every
forward / state-dict done with the EMA swapped in.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

EMA_SHARD_DIR = "ema"
EMA_SHARD_SCHEMA = 1
# Elements per foreach chunk when the shadow lives on another device than the
# params (host staging buffer: 2**28 fp32 = 1 GiB).
DEFAULT_CHUNK_NUMEL = 2 ** 28


def ema_shard_path(state_dir: str, rank: int, dcp_dir: str = "optim") -> str:
    """``<state_dir>/<dcp_dir>/ema/local_rank<rank>.pt`` -- next to the optimizer shards."""
    return os.path.join(state_dir, dcp_dir, EMA_SHARD_DIR, f"local_rank{int(rank)}.pt")


def fsdp_flat_param_handles(model: Any) -> List[Any]:
    """The ``FlatParamHandle`` of every FSDP1 unit under *model* (``[]`` when
    *model* is not FSDP-wrapped, e.g. a single-process CPU run)."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:  # pragma: no cover
        return []
    if not isinstance(model, FSDP):
        return []
    return [m._handle for m in FSDP.fsdp_modules(model) if getattr(m, "_handle", None) is not None]


def free_unsharded_flat_params(handles: Sequence[Any]) -> int:
    """Reshard every handle that still holds its unsharded flat parameter, so the
    next ``unshard()`` all-gathers again. Call outside forward/backward.
    Returns the number of handles freed.
    """
    freed = 0
    for h in handles:
        if h is None or not getattr(h, "uses_sharded_strategy", False):
            continue
        if h.needs_unshard():
            continue  # storage already freed: the next unshard gathers anyway
        h.reshard(True)
        h.post_reshard()
        h._prefetched = False
        freed += 1
    return freed


def _atomic_torch_save(blob: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(blob, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ShardEMA:
    """EMA of the trainable parameters, one shadow tensor per local shard.

    Parameters
    ----------
    named_params:
        ``(name, Parameter)`` pairs, typically ``dit.named_parameters()`` after
        ``accelerator.prepare``. Entries with ``requires_grad=False`` are
        dropped here, so frozen weights never enter the EMA.
    decay:
        ``ema = decay * ema + (1 - decay) * p`` per update.
    update_every:
        Only every N-th successful optimizer step updates the EMA (the others
        are counted but skipped).
    device:
        ``"gpu"`` keeps each shadow on its parameter's device; ``"cpu"`` keeps
        it in host memory and stages the update through chunks of
        ``chunk_numel`` elements.
    warmup_steps:
        While the number of successful optimizer steps is ``<= warmup_steps``
        the shadow is a copy of the weights instead of an average.
    """

    def __init__(
        self,
        named_params: Sequence[Tuple[str, torch.nn.Parameter]],
        decay: float,
        *,
        update_every: int = 1,
        device: str = "gpu",
        warmup_steps: int = 0,
        chunk_numel: int = DEFAULT_CHUNK_NUMEL,
        log: Callable[[str], None] = print,
    ) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"ema decay must be in (0, 1), got {decay!r}")
        if int(update_every) < 1:
            raise ValueError(f"ema update_every must be >= 1, got {update_every!r}")
        if str(device) not in ("gpu", "cpu"):
            raise ValueError(f"ema device must be 'gpu' or 'cpu', got {device!r}")
        pairs = [(str(n), p) for n, p in named_params if p.requires_grad]
        self.names: List[str] = [n for n, _ in pairs]
        self.params: List[torch.nn.Parameter] = [p for _, p in pairs]
        self.decay = float(decay)
        self.update_every = int(update_every)
        self.device_mode = str(device)
        self.warmup_steps = int(warmup_steps)
        self.chunk_numel = int(chunk_numel)
        self.log = log
        self.shadow: Optional[List[torch.Tensor]] = None
        # successful optimizer steps; restored on resume
        self.opt_steps: int = 0
        self.n_updates: int = 0
        self.last_decay: float = 0.0
        self.initialised_from: Optional[str] = None
        self._pending_load: Optional[Tuple[str, bool, bool, Optional[int]]] = None
        self._swapped = False

    # ------------------------------------------------------------------ props
    @property
    def initialised(self) -> bool:
        return self.shadow is not None

    @property
    def n_params(self) -> int:
        return len(self.params)

    def numel(self) -> int:
        """Elements held in the shadow (this rank's shard), or the would-be size."""
        if self.shadow is not None:
            return sum(t.numel() for t in self.shadow)
        return sum(p.data.numel() for p in self.params)

    def settings(self) -> Dict[str, Any]:
        """JSON-safe summary for ``resume_state.json`` / ``meta.json``."""
        return {
            "decay": self.decay,
            "update_every": self.update_every,
            "device": self.device_mode,
            "warmup_steps": self.warmup_steps,
            "opt_steps": int(self.opt_steps),
            "n_updates": int(self.n_updates),
            "n_tensors": int(self.n_params),
            "initialised": bool(self.initialised),
            "initialised_from": self.initialised_from,
        }

    # ------------------------------------------------------------------ views
    def _target_device(self, p: torch.Tensor) -> torch.device:
        return torch.device("cpu") if self.device_mode == "cpu" else p.device

    def _views(self, where: str) -> List[torch.Tensor]:
        """Fresh ``p.data`` views (FSDP rebuilds them after every unshard)."""
        views = []
        for n, p in zip(self.names, self.params):
            v = p.data
            if v.numel() and v.dtype is not torch.float32:
                raise SystemExit(
                    f"[ema] {where}: trainable param {n} is {v.dtype}, expected fp32. "
                    "The orig-param views are refreshed to fp32 by the first forward / "
                    "state-dict call; master weights kept in low precision are not supported."
                )
            views.append(v)
        return views

    def _check_shapes(self, views: List[torch.Tensor], where: str) -> None:
        assert self.shadow is not None
        if len(views) != len(self.shadow):
            raise SystemExit(
                f"[ema] {where}: {len(views)} params vs {len(self.shadow)} shadow tensors"
            )
        for n, v, s in zip(self.names, views, self.shadow):
            if v.shape != s.shape or s.dtype is not torch.float32:
                raise SystemExit(
                    f"[ema] {where}: shape/dtype mismatch for {n}: param {tuple(v.shape)} "
                    f"{v.dtype} vs shadow {tuple(s.shape)} {s.dtype}"
                )

    def _chunks(self, views: List[torch.Tensor]):
        """Index ranges whose total numel stays under ``chunk_numel`` (>= 1 each)."""
        if self.device_mode != "cpu":
            yield 0, len(views)
            return
        start, acc = 0, 0
        for i, v in enumerate(views):
            if acc and acc + v.numel() > self.chunk_numel:
                yield start, i
                start, acc = i, 0
            acc += v.numel()
        if start < len(views):
            yield start, len(views)

    # ------------------------------------------------------------------ init / load
    def schedule_load(self, path: str, *, strict: bool, warn_on_step_mismatch: bool = True,
                      world_size: Optional[int] = None) -> None:
        """Initialise the shadow from *path* (this rank's saved shards) instead of
        the weights, when the shadow is first materialised. ``strict=False``
        falls back to copying the weights on a mismatch (shapes, or the blob's
        recorded ``world_size`` vs *world_size* -- e.g. a ``schedule``-mode
        resume onto another GPU count); ``strict=True`` raises ``SystemExit``.
        The fallback is per rank: the trainer makes it collective by comparing
        :attr:`initialised_from` across ranks after the first step.

        The shards carry the EMA's own step count; on load it replaces
        :attr:`opt_steps`, so the warmup / ``update_every`` phase continues from
        the EMA's history rather than from whatever the trainer's counter says
        (a ``warm`` resume restarts that counter at 0, which must not turn the
        restored average back into a copy of the weights). A difference between
        the two counts is logged as a WARN unless ``warn_on_step_mismatch`` is
        False (pass False for ``warm`` resumes, where it is expected).
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self._pending_load = (str(path), bool(strict), bool(warn_on_step_mismatch),
                              None if world_size is None else int(world_size))

    def _initialise(self, views: List[torch.Tensor], *, resumed_at: int) -> bool:
        """Materialise the shadow. Returns True when it was loaded from saved
        shards (the caller then applies this step's update on top of it) and
        False when it was copied from the weights (that copy is the update).
        *resumed_at* is the step count before the step that triggered this."""
        if self._pending_load is not None:
            path, strict, warn_mismatch, world_size = self._pending_load
            self._pending_load = None
            try:
                self._load_into(path, views, resumed_at=resumed_at, warn_mismatch=warn_mismatch,
                                world_size=world_size)
                self.initialised_from = f"shards:{path}"
                self.log(f"[ema] restored {len(self.shadow)} shadow tensors "
                         f"({self.numel() / 1e6:.1f}M elements) from {path} "
                         f"(n_updates={self.n_updates}, opt_steps={self.opt_steps - 1})")
                return True
            except SystemExit as e:
                if strict:
                    raise
                self.log(f"[ema] WARN {path} does not match this rank's shards ({e}); "
                         "initialising the EMA from the current weights instead")
                self.shadow = None
        self._copy_from_weights(views)
        self.log(f"[ema] initialised {len(self.shadow)} shadow tensors "
                 f"({self.numel() / 1e6:.1f}M elements, {self.device_mode}) from the "
                 f"current weights at opt_step={self.opt_steps}")
        return False

    def _copy_from_weights(self, views: List[torch.Tensor]) -> None:
        self.shadow = [v.detach().to(self._target_device(v), copy=True) for v in views]
        self.initialised_from = "weights"
        self.last_decay = 0.0
        self.n_updates += 1

    @torch.no_grad()
    def reinitialise_from_weights(self, *, opt_steps: int, reason: str) -> None:
        """Throw the shadow away and restart it as a copy of the current weights,
        with the given step count. Used by the trainer's collective fallback:
        when the ranks disagree on where their EMA came from (some restored
        saved shards, some could not), every rank must end up in the state the
        non-restored ranks are already in -- a copy taken at this step -- or the
        gathered EMA would be a rank-wise mixture of two different averages."""
        if self._swapped:
            raise RuntimeError("[ema] reinitialise_from_weights while the EMA weights are swapped in")
        views = self._views("reinitialise")
        self.shadow = None
        self.n_updates = 0
        self._copy_from_weights(views)
        self.opt_steps = int(opt_steps)
        self.log(f"[ema] WARN re-initialised {len(self.shadow)} shadow tensors from the current "
                 f"weights at opt_step={self.opt_steps}: {reason}")

    def _load_into(self, path: str, views: List[torch.Tensor], *, resumed_at: int,
                   warn_mismatch: bool, world_size: Optional[int] = None) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if int(blob.get("schema", -1)) != EMA_SHARD_SCHEMA:
            raise SystemExit(f"[ema] {path}: schema {blob.get('schema')} != {EMA_SHARD_SCHEMA}")
        saved_ws = blob.get("world_size")
        if world_size is not None and saved_ws is not None and int(saved_ws) != int(world_size):
            raise SystemExit(
                f"[ema] {path}: shards were saved with world_size={int(saved_ws)} but this run "
                f"has {int(world_size)}; EMA shards are not resharded"
            )
        names, shards = list(blob["names"]), list(blob["shards"])
        if names != self.names:
            raise SystemExit(
                f"[ema] {path}: parameter names differ from the model "
                f"({len(names)} saved vs {len(self.names)} trainable)"
            )
        if len(shards) != len(views):
            raise SystemExit(f"[ema] {path}: {len(shards)} shards vs {len(views)} params")
        for n, s, v in zip(names, shards, views):
            if tuple(s.shape) != tuple(v.shape):
                raise SystemExit(
                    f"[ema] {path}: shard shape {tuple(s.shape)} != param shard "
                    f"{tuple(v.shape)} for {n} (saved with another world size?)"
                )
        self.shadow = [s.to(torch.float32).to(self._target_device(v)) for s, v in zip(shards, views)]
        self.n_updates = int(blob.get("n_updates", 0))
        saved_steps = blob.get("opt_steps")
        if saved_steps is not None:
            saved_steps = int(saved_steps)
            if saved_steps != int(resumed_at):
                self.log(f"[ema] {'WARN ' if warn_mismatch else ''}{path} was saved at "
                         f"opt_step={saved_steps}, this run resumes at {resumed_at}; the EMA "
                         f"continues its own count from {saved_steps} (warmup / update_every)")
            # The current step is already being counted by the caller.
            self.opt_steps = saved_steps + 1
        del blob, shards

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def step(self) -> bool:
        """Call once after every successful ``optimizer.step`` (never after a
        NaN-skipped one). Returns True when the shadow was touched."""
        if self._swapped:
            raise RuntimeError("[ema] step() called while the EMA weights are swapped in")
        resumed_at = self.opt_steps
        self.opt_steps += 1
        views = self._views("step")
        if self.shadow is None and not self._initialise(views, resumed_at=resumed_at):
            return True
        if self.opt_steps % self.update_every != 0:
            return False
        self._check_shapes(views, "step")
        if self.opt_steps <= self.warmup_steps:
            d = 0.0
        else:
            d = self.decay
        for a, b in self._chunks(views):
            src = [v.to(s.device, non_blocking=False) for v, s in zip(views[a:b], self.shadow[a:b])]
            dst = self.shadow[a:b]
            if d == 0.0:
                torch._foreach_copy_(dst, src)
            else:
                torch._foreach_mul_(dst, d)
                torch._foreach_add_(dst, src, alpha=1.0 - d)
            del src
        self.last_decay = d
        self.n_updates += 1
        return True

    # ------------------------------------------------------------------ swap
    def _exchange(self, where: str) -> None:
        """Swap the contents of every shadow tensor and its param shard in place.

        One tensor of scratch at a time, on the shadow's device; the copies are
        same-dtype so a swap followed by a swap is bit-exact.
        """
        assert self.shadow is not None
        views = self._views(where)
        self._check_shapes(views, where)
        for v, s in zip(views, self.shadow):
            if not v.numel():
                continue
            tmp = v.detach().to(s.device, copy=True)
            v.copy_(s)
            s.copy_(tmp)
            del tmp

    @contextlib.contextmanager
    def swap_in(self, *, reshard: Optional[Callable[[], Any]] = None):
        """Temporarily put the EMA weights into the model's (local) params.

        No collective: each rank swaps its own shard, and FSDP's next
        all-gather assembles the EMA weights. The raw weights live in the
        shadow tensors meanwhile and are swapped back on exit, even on error.
        When the shadow is not initialised yet (no successful step so far) the
        model is left untouched and a warning is logged.

        *reshard* (the trainer passes ``free_unsharded_flat_params`` over the
        model's handles) is called right after the swap-in and again after the
        swap-out: FSDP1 keeps the root unit's unsharded buffer alive across a
        forward, so without it the next forward / state-dict would reuse the
        stale raw gather instead of gathering the swapped shards (module
        docstring). The caller must also reshard before the raw pass that
        precedes a swap when a forward may have run since the last backward.
        """
        if self.shadow is None:
            self.log("[ema] WARN swap_in before the first optimizer step: the EMA is not "
                     "initialised, evaluating the raw weights instead")
            yield False
            return
        if self._swapped:
            raise RuntimeError("[ema] swap_in is not reentrant")
        with torch.no_grad():
            self._exchange("swap_in")
        self._swapped = True
        try:
            if reshard is not None:
                reshard()
            yield True
        finally:
            with torch.no_grad():
                self._exchange("swap_out")
            self._swapped = False
            if reshard is not None:
                reshard()

    # ------------------------------------------------------------------ persistence
    def save_shards(self, path: str, *, rank: int, world_size: int) -> None:
        """Write this rank's shadow tensors (CPU, fp32) atomically to *path*."""
        if self.shadow is None:
            raise RuntimeError("[ema] save_shards before the EMA was initialised")
        if self._swapped:
            raise RuntimeError("[ema] save_shards while the EMA weights are swapped in")
        blob = {
            "schema": EMA_SHARD_SCHEMA,
            "names": list(self.names),
            "shards": [t.detach().to("cpu", copy=True) for t in self.shadow],
            "opt_steps": int(self.opt_steps),
            "n_updates": int(self.n_updates),
            "rank": int(rank),
            "world_size": int(world_size),
            "settings": self.settings(),
        }
        _atomic_torch_save(blob, path)
        del blob
