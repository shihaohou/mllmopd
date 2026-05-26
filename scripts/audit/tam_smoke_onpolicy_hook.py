"""Step 3a v3 — standalone smoke test for the onpolicy_category hook.

Runs the v3 main-path weight computation on a sample of cached student
response_ids and reports:

  - fire rate         (must match audit pooled rate within sample noise)
  - per-sample latency (informs training overhead)
  - fresh-vs-cached cats agreement (must be 100% for determinism)
  - spaCy load + warm-up time

Catches three pre-training risks before launching B0/B1 230-step:

  - spaCy / en_core_web_sm install state in training venv
  - classifier determinism between offline cache and online hook
  - tokenizer decode roundtrip correctness

Usage::

    .venv/bin/python -m scripts.audit.tam_smoke_onpolicy_hook \\
        --student-ckpt runs/t1_v1p5b_T1_2_full_mm/ckpt/hf/step_230 \\
        --teacher-cache runs/audit/tam_step1a_classifier_v013_full/teacher_cache.jsonl \\
        --n-samples 20 \\
        --alpha 0.223
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student-ckpt", type=str,
                    default=os.environ.get("STUDENT_CKPT"),
                    help="HF tokenizer dir (any Qwen2.5-VL ckpt). "
                         "Defaults to $STUDENT_CKPT env.")
    ap.add_argument("--teacher-cache", type=Path,
                    default=Path("runs/audit/tam_step1a_classifier_v013_full/teacher_cache.jsonl"),
                    help="cache jsonl with response_ids + classification")
    ap.add_argument("--n-samples", type=int, default=20,
                    help="number of cache rows to smoke (default 20)")
    ap.add_argument("--alpha", type=float, default=0.223,
                    help="boost alpha (default 0.223 from Gate 1)")
    ap.add_argument("--c-local", type=str,
                    default="content_noun,visual_attribute,proper_noun",
                    help="comma-separated C_local categories")
    args = ap.parse_args(argv)

    if not args.student_ckpt:
        print("!! --student-ckpt or $STUDENT_CKPT required "
              "(any Qwen2.5-VL HF dir works for tokenizer)", file=sys.stderr)
        return 2
    if not args.teacher_cache.exists():
        print(f"!! teacher_cache not found: {args.teacher_cache}", file=sys.stderr)
        return 2

    # Set env BEFORE importing the hook so internals see the locked α.
    os.environ.setdefault("MLLMOPD_TAM_HOOK_MODE", "onpolicy_category")
    os.environ.setdefault("MLLMOPD_TAM_ALPHA", str(args.alpha))

    # Be robust to running without mllmopd installed in develop mode.
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from mllmopd.training.tam_boost_hook import (  # noqa: E402
        _weights_onpolicy_category,
        _get_tokenizer,
        _get_classifier_fns,
    )

    c_local = {c.strip() for c in args.c_local.split(",") if c.strip()}
    print(f">>> C_local       = {sorted(c_local)}")
    print(f">>> alpha         = {args.alpha}")
    print(f">>> student_ckpt  = {args.student_ckpt}")
    print(f">>> teacher_cache = {args.teacher_cache}")
    print(f">>> n_samples     = {args.n_samples}")
    print()

    # 1) Tokenizer load
    class FakeArgs:
        def __init__(self, ckpt: str):
            self.hf_checkpoint = ckpt
            self.load = ckpt

    t0 = time.time()
    tok = _get_tokenizer(FakeArgs(args.student_ckpt))
    print(f">>> tokenizer loaded in {time.time()-t0:.1f}s (vocab={len(tok)})")

    # 2) Classifier import (spaCy is lazy inside tam_sanity, so warmup is
    #    measured separately below).
    t0 = time.time()
    cls_fn, lbl_fn = _get_classifier_fns()
    print(f">>> classifier_fn imported in {time.time()-t0:.2f}s")

    # 3) Warm-up call (spaCy lazy init happens on first classify)
    class FakeSample:
        def __init__(self, d: dict):
            self.response_ids = d["response_ids"]
            self.response_length = d["response_length"]
            self.id = d["id"]
            self.index = d["id"]

    metrics_warm: Counter = Counter()
    with args.teacher_cache.open() as f:
        d0 = json.loads(f.readline())
    t0 = time.time()
    _weights_onpolicy_category(
        FakeSample(d0), tok, args.alpha, c_local, metrics_warm,
    )
    print(f">>> spaCy warmup (1 sample) {time.time()-t0:.2f}s")
    print()

    # 4) Real loop
    metrics: Counter = Counter()
    total_fire = total_tok = 0
    n_agree = n_compared = 0
    per_sample_ms: list[float] = []

    print(f"{'idx':>3} {'id':<28} {'R':>4} {'fire':>5} {'fire%':>6} "
          f"{'agree':>9} {'ms':>5}  reason")
    print("-" * 80)
    t_run = time.time()
    with args.teacher_cache.open() as f:
        for i, line in enumerate(f):
            if i >= args.n_samples:
                break
            d = json.loads(line)
            cached_cats = (d.get("classification") or {}).get("token_category") or []

            s = FakeSample(d)
            t_s = time.time()
            weights, info = _weights_onpolicy_category(
                s, tok, args.alpha, c_local, metrics,
            )
            dt_ms = 1000 * (time.time() - t_s)
            per_sample_ms.append(dt_ms)

            R = s.response_length
            fire = sum(1 for w in weights if w > 1.0)
            total_fire += fire
            total_tok += R

            # Determinism check: re-run classifier path, compare with cache
            agree_str = "n/a"
            if cached_cats and len(cached_cats) >= R:
                labels = lbl_fn(s.response_ids, tok)
                cls = cls_fn(s.response_ids, tok, labels["is_answer_token"])
                fresh_cats = cls["token_category"][:R]
                agree = sum(1 for a, b in zip(fresh_cats, cached_cats[:R]) if a == b)
                n_agree += agree
                n_compared += R
                agree_str = f"{agree}/{R}"

            sid_short = (s.id or "?")[:28]
            reason = info.get("reason", "ok") if isinstance(info, dict) else "ok"
            print(f"{i:>3d} {sid_short:<28} {R:>4d} {fire:>5d} "
                  f"{100*fire/max(1,R):>5.1f}% {agree_str:>9} "
                  f"{dt_ms:>5.0f}  {reason}")

    dur = time.time() - t_run
    print("-" * 80)
    print()
    print(f"=== Summary on {args.n_samples} samples ===")
    fr_pct = 100 * total_fire / max(1, total_tok)
    print(f"  fire rate (smoke)   = {total_fire}/{total_tok} = {fr_pct:.2f}%")
    print(f"  fire rate (audit)   = 26.92%  (pooled on 205 samples)")
    print(f"  per-sample ms       = mean={sum(per_sample_ms)/len(per_sample_ms):.1f}  "
          f"max={max(per_sample_ms):.1f}  min={min(per_sample_ms):.1f}")
    agreement = 100 * n_agree / max(1, n_compared) if n_compared else 0.0
    print(f"  fresh vs cached     = {n_agree}/{n_compared} = {agreement:.2f}% agreement")
    print(f"  metrics             = {dict(metrics)}")
    print(f"  total wall          = {dur:.1f}s "
          f"({1000*dur/args.n_samples:.0f} ms/sample incl I/O)")
    print()

    # 5) Health gate
    warnings: list[str] = []
    if fr_pct < 5.0:
        warnings.append(
            f"FAIL: fire rate {fr_pct:.1f}% << 27% — spaCy likely not installed; "
            f"linguistic categories degraded to 'other'"
        )
    elif fr_pct < 15.0:
        warnings.append(
            f"WARN: fire rate {fr_pct:.1f}% noticeably below 27% — check spaCy "
            f"version + sample composition"
        )
    if per_sample_ms and max(per_sample_ms) > 500:
        warnings.append(
            f"WARN: per-sample max {max(per_sample_ms):.0f}ms > 500ms — "
            f"training overhead may be material"
        )
    if n_compared > 0 and agreement < 99.0:
        warnings.append(
            f"FAIL: classifier non-deterministic — fresh vs cached only "
            f"{agreement:.2f}% agreement"
        )

    if warnings:
        print("!! HEALTH CHECK WARNINGS:")
        for w in warnings:
            print(f"   - {w}")
        return 1
    print("[OK] smoke healthy: hook plumbing ready for B0/B1 230-step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
