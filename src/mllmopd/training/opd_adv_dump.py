"""T2-1 A0c: dump per-sample old_log_probs at compute_advantages_and_returns time.

Sidecar to opd_diagnostics_hook.post_process_rewards_with_diagnostics.

The existing diag hook writes lp_full / lp_blank / vd / vd_weights /
sample_index *before* the trainer forward pass — at which point
sample.old_log_probs is still None. So the energy audit cannot
reconstruct adv_t = lp_full(t) - old_log_probs(t).

This module fills the gap: a SECOND dump fired from inside
compute_advantages_and_returns (loss.py P19) AFTER the trainer forward
has produced student log_probs but BEFORE PG loss is computed. It
writes only what the existing diag hook lacks: per-sample
old_log_probs, keyed by sample_index so the audit can join with the
existing diag file.

Guarded by env ``MLLMOPD_DUMP_OPD_ADV=1`` so all non-T2-1-A0 runs are
byte-identical (T1-2 / T1-3 baselines, normal T2-1, future T2-* with
the env unset).

Filename: ``<MLLMOPD_RUNS>/<OPD_RUN_NAME>/diagnostics/step_NNNNNN.adv_dp{R}.jsonl.gz``

dp-rank suffix is required because compute_advantages_and_returns
runs on every data-parallel rank with disjoint sample shards. The
audit merges across ranks for one step via sample_index.

Row schema::
    {
      "sample_index": int,        # matches Sample.index / rollout_data["sample_indices"]
      "step": int,                # local step counter (this module)
      "dp_rank": int,
      "old_log_probs": list[float],  # length = response_length
    }
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_STEP_COUNTER = {"i": 0}


def _out_dir() -> Path:
    base = Path(os.environ.get("MLLMOPD_RUNS", "runs"))
    run_name = os.environ.get("OPD_RUN_NAME", "t1_default")
    out = base / run_name / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_float_list(x) -> list[float]:
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach().cpu().float()
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def dump_opd_adv(rollout_data, student_log_probs, parallel_state) -> None:
    """Dump (sample_index, old_log_probs) per-sample for one trainer step.

    Args:
        rollout_data: Uni-OPD RolloutBatch dict. Must contain
            ``sample_indices`` (list[int]).
        student_log_probs: list[torch.Tensor], each of length =
            response_length. Semantically this is "old_log_probs": the
            student logp on the rolled-out tokens, computed by a forward
            pass at trainer-step start, BEFORE any optimization update.
        parallel_state: Uni-OPD ParallelState. Used for dp_rank in
            filename. Other attrs ignored.

    No return; writes a per-step .jsonl.gz sidecar.
    No-op when env MLLMOPD_DUMP_OPD_ADV != "1".
    """
    if os.environ.get("MLLMOPD_DUMP_OPD_ADV", "0") != "1":
        return

    try:
        dp_rank = int(getattr(parallel_state, "dp_rank", 0))
    except Exception:
        dp_rank = 0

    step_i = _STEP_COUNTER["i"]
    _STEP_COUNTER["i"] += 1

    out_path = _out_dir() / f"step_{step_i:06d}.adv_dp{dp_rank}.jsonl.gz"
    t0 = time.time()

    sample_indices = rollout_data.get("sample_indices") or []
    n_written = 0
    n_no_index = 0

    with gzip.open(out_path, "wt") as fout:
        for i, lp in enumerate(student_log_probs):
            sample_index = sample_indices[i] if i < len(sample_indices) else None
            if sample_index is None:
                n_no_index += 1
            row = {
                "sample_index": sample_index,
                "step": step_i,
                "dp_rank": dp_rank,
                "old_log_probs": _to_float_list(lp),
            }
            fout.write(json.dumps(row) + "\n")
            n_written += 1

    logger.info(
        f"[opd_adv_dump] step {step_i} dp{dp_rank}: wrote {n_written} rows "
        f"({n_no_index} missing sample_index) to {out_path} "
        f"({time.time() - t0:.2f}s)"
    )
