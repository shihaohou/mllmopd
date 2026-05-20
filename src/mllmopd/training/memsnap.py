"""GPU memory snapshot helper for T1-2 OOM v11 diagnosis.

T1-2 OOM #1..#10 converged to ~118 GiB trainer alloc but the per-rank
budget can only account for ~35 GiB. To find the missing ~83 GiB we need
ground-truth allocation snapshots inside the train loop, taken right
before the OOM-ing fused_vocab_parallel_cross_entropy / logprob call in
policy_loss_function. See docs/gpt-reply-2026-05-20-v10-update.md.

Gated by env vars so it's a true no-op when MLLMOPD_MEMSNAP is unset:

  MLLMOPD_MEMSNAP              "1" → enabled; default unset → disabled.
  MLLMOPD_MEMSNAP_DIR          Output dir (default /tmp/mllmopd_memsnap).
                               Launcher sets it to
                               ${EXPERIMENT_DIR}/diagnostics/memsnap.
  MLLMOPD_MEMSNAP_MAX_ENTRIES  Allocation events buffered by PyTorch
                               (default 2_000_000).
  MLLMOPD_MEMSNAP_MAX_DUMPS    Per-process dump cap (default 50). Each
                               dump = .json + .summary.txt + .pickle and
                               a single pickle can reach 50–200 MiB.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import torch


_STARTED = False
_DUMPED = 0


def _enabled() -> bool:
    return os.environ.get("MLLMOPD_MEMSNAP", "0") == "1"


def _rank_tag() -> str:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "unknown"
    pid = os.getpid()
    host = socket.gethostname()
    try:
        dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    return f"host={host}_rank={rank}_pid={pid}_cuda={dev}"


def _out_dir() -> Path:
    base = os.environ.get("MLLMOPD_MEMSNAP_DIR", "/tmp/mllmopd_memsnap")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def start_memory_history_once() -> None:
    """Enable PyTorch allocator event recording (idempotent, env-gated)."""
    global _STARTED
    if not _enabled() or _STARTED or not torch.cuda.is_available():
        return

    max_entries = int(os.environ.get("MLLMOPD_MEMSNAP_MAX_ENTRIES", "2000000"))

    # _record_memory_history signature varies across PyTorch versions; try
    # the richest call first, fall back progressively.
    attempts = (
        dict(enabled="all", stacks="all", max_entries=max_entries),
        dict(enabled=True, max_entries=max_entries),
        dict(max_entries=max_entries),
    )
    for kwargs in attempts:
        try:
            torch.cuda.memory._record_memory_history(**kwargs)
            _STARTED = True
            print(
                f"[memsnap] recording started kwargs={list(kwargs.keys())} "
                f"max_entries={max_entries} out_dir={_out_dir()}",
                flush=True,
            )
            return
        except TypeError:
            continue
        except Exception as e:
            print(
                f"[memsnap] _record_memory_history failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return

    print("[memsnap] _record_memory_history: no compatible signature found",
          flush=True)


def dump_memory_snapshot(tag: str, **meta) -> None:
    """Write .json + .summary.txt + .pickle for `tag` (env-gated, capped)."""
    global _DUMPED
    if not _enabled() or not torch.cuda.is_available():
        return

    max_dumps = int(os.environ.get("MLLMOPD_MEMSNAP_MAX_DUMPS", "50"))
    if _DUMPED >= max_dumps:
        return

    start_memory_history_once()

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    out_dir = _out_dir()
    ts = int(time.time())
    safe_tag = tag.replace("/", "_").replace(" ", "_")
    prefix = out_dir / f"{ts}_{_DUMPED:03d}_{safe_tag}_{_rank_tag()}"

    try:
        free, total = torch.cuda.mem_get_info()
        free_gib, total_gib = free / 2**30, total / 2**30
    except Exception:
        free_gib, total_gib = None, None

    summary = {
        "tag": tag,
        "time": ts,
        "rank_tag": _rank_tag(),
        "dump_idx": _DUMPED,
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "free_gib": free_gib,
        "total_gib": total_gib,
    }
    for k, v in meta.items():
        if isinstance(v, (int, float, str, bool, list, tuple, type(None))):
            summary[k] = v
        else:
            summary[k] = str(v)

    try:
        with open(f"{prefix}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as e:
        print(f"[memsnap] json write failed at {tag}: "
              f"{type(e).__name__}: {e}", flush=True)

    try:
        with open(f"{prefix}.summary.txt", "w") as f:
            f.write(torch.cuda.memory_summary(abbreviated=False))
    except Exception as e:
        print(f"[memsnap] memory_summary failed at {tag}: "
              f"{type(e).__name__}: {e}", flush=True)

    try:
        torch.cuda.memory._dump_snapshot(f"{prefix}.pickle")
        print(
            f"[memsnap] dumped {prefix.name} "
            f"alloc={summary['allocated_gib']:.2f}GiB "
            f"reserved={summary['reserved_gib']:.2f}GiB "
            f"free={'NA' if free_gib is None else f'{free_gib:.2f}GiB'}",
            flush=True,
        )
    except Exception as e:
        print(f"[memsnap] _dump_snapshot failed at {tag}: "
              f"{type(e).__name__}: {e}", flush=True)

    _DUMPED += 1


def before_train_step_hook(
    args, rollout_id, step_id, model, optimizer, opt_param_scheduler,
):
    """`--custom-megatron-before-train-step-hook-path` entrypoint.

    Fires inside Uni-OPD's train_one_step at the top of the function
    (miles/backends/megatron_utils/model.py:~349), before forward/backward.
    Used for per-step baseline snapshots + one-time DDP buffer audit.
    """
    start_memory_history_once()
    try:
        rid = int(rollout_id) if rollout_id is not None else None
    except Exception:
        rid = None
    try:
        sid = int(step_id) if step_id is not None else None
    except Exception:
        sid = None
    dump_memory_snapshot("before_train_step", rollout_id=rid, step_id=sid)

    # GPT diag Q1: confirm whether Megatron's distributed optimizer still
    # holds a full-size fp32 grad buffer after we dropped
    # --accumulate-allreduce-grads-in-fp32 in v10. One-shot at step_id==0.
    if sid == 0:
        try:
            buffers = getattr(model[0], "buffers", None)
            if buffers is None:
                print("[memsnap] DDP buffer audit: model[0].buffers missing",
                      flush=True)
            else:
                for i, buf in enumerate(buffers):
                    pd = getattr(buf, "param_data", None)
                    gd = getattr(buf, "grad_data", None)
                    pd_info = None if pd is None else (pd.numel(), str(pd.dtype))
                    gd_info = None if gd is None else (gd.numel(), str(gd.dtype))
                    print(
                        f"[memsnap DDP buf {i}] "
                        f"param_data={pd_info} grad_data={gd_info}",
                        flush=True,
                    )
        except Exception as e:
            print(f"[memsnap] DDP buffer audit failed: "
                  f"{type(e).__name__}: {e}", flush=True)
