"""H2 audit: forced-decoding logprob extraction for "visual dependency".

For each completion in a source JSONL (output of `run_audit_pass*`), compute
per-token logprobs of the completion under the same model in two image
conditions: full_image and blank_image. The per-token difference

    vd(t) = logp(t | prefix, full_image) - logp(t | prefix, blank_image)

is the visual dependency at that token (positive = the image actively
raises the probability of this token). This unlocks H2 audit: does the
OPD reward / KL signal concentrate on tokens with high VD (visually
grounded) or low VD (language-prior, "would have said anyway")?

Backend: sglang. Forced decoding with `return_logprob=True` and
`max_new_tokens=1` (the generated token is discarded — we only want the
prompt's per-token logprobs). HF transformers backend was dropped because
the 500 × 2 forward-pass workload is decode-bound under HF batch=1 and
takes >1h; sglang submits all 1000 forced-decode calls and finishes in
minutes.

Must run in the train venv (sglang installed).

Usage:
    python -m mllmopd.diagnostics.score_completion \\
        --subset data/audit/level1_subset_v0.jsonl \\
        --source runs/audit/level1_v4_sysprompt_fixed/T_RL_full.jsonl \\
        --model /path/to/MMR1-7B-RL \\
        --system-prompt-text "$MMR1_SYSTEM_PROMPT" \\
        --id-filter runs/audit/level1_v4_sysprompt_fixed/opd_target_ids.json \\
        --out runs/audit/level1_v4_sysprompt_fixed/T_RL_score_opd_target.jsonl

For cross-model scoring (e.g., student logp under teacher's completion),
point `--source` at the teacher's jsonl but `--model` at the student.

IMPORTANT: For MMR1-trained models the `--system-prompt-text` flag is REQUIRED.
Pass the verbatim MMR1 training-time system prompt (defined in
`scripts/audit/run_smoke.sh` as $MMR1_SYSTEM_PROMPT). Skipping it leaves
the model in base-model mode and the resulting VD distribution will not
match the canonical audit prefix. See `run_audit_pass.py::_build_messages`
for the exact content-list ordering ([text:sysprompt, image, text:question]).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mllmopd.data import mllm_corruptions
from mllmopd.diagnostics.run_audit_pass import _build_messages, _load_subset


def _build_prefix_and_full(
    tokenizer, rec, image, response_text: str, system_prompt: str = "",
) -> tuple[str, str]:
    """Chat-template prefix + assistant response = scoring prompt.

    Reuses the canonical `_build_messages` from `run_audit_pass` so the prefix
    is byte-identical to what the audit pipeline feeds the same model.
    Critical for MMR1 / MMR1-derived checkpoints: the training-time system
    prompt must be emitted as a text block at the start of the user turn,
    BEFORE the image. Without it the teacher stays in base-model mode and
    the resulting logp_full/logp_blank distribution is not a measurement
    of MMR1's vision-conditioned behavior. See `_build_messages` docstring.
    """
    messages = _build_messages(rec, image, prefix=None, system_prompt=system_prompt)
    prefix_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    return prefix_text, prefix_text + response_text


def _load_source_index(path: Path) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx[r["id"]] = r
    return idx


def _load_subset_image(rec):
    image_field = rec.get("image")
    if isinstance(image_field, list) and image_field:
        image_field = image_field[0]
    if isinstance(image_field, (str, Path)):
        try:
            return mllm_corruptions.load(image_field)
        except Exception as e:
            print(f"!! could not load image {image_field}: {e}", file=sys.stderr)
            return None
    if hasattr(image_field, "convert"):
        return image_field
    return None


def _extract_input_logprobs(out) -> list[float] | None:
    """Pull per-input-token logprobs from one sglang output. Handles API drift:
    sglang has named this field a few different ways across versions
    (`input_token_logprobs`, `input_logprobs`, `prompt_token_logprobs`).
    Entries can be either floats or `(logprob, token_id, token_text)` tuples."""
    if isinstance(out, dict):
        meta = out.get("meta_info") or {}
    else:
        meta = getattr(out, "meta_info", {}) or {}
    for key in ("input_token_logprobs", "input_logprobs", "prompt_token_logprobs"):
        v = meta.get(key)
        if v is None:
            continue
        if not v:
            return []
        if isinstance(v[0], (list, tuple)):
            # Tuple/list per token: take first element (the logprob).
            # Some entries (e.g. the BOS position) may have None — coerce to 0.0.
            return [float(item[0]) if item[0] is not None else 0.0 for item in v]
        return [float(x) if x is not None else 0.0 for x in v]
    return None


def _engine_generate(engine, prompts, image_data_list, sampling_params, **kwargs):
    """Wrap engine.generate to keep image_data optional + handle batched logprob calls."""
    call_kwargs = {
        "prompt": prompts,
        "sampling_params": sampling_params,
        **kwargs,
    }
    if any(img is not None for img in image_data_list):
        call_kwargs["image_data"] = image_data_list
    return engine.generate(**call_kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", required=True, type=Path,
                    help="audit subset jsonl (provides image + question by id)")
    ap.add_argument("--source", required=True, type=Path,
                    help="existing run_audit_pass jsonl whose `prediction`s are scored")
    ap.add_argument("--model", required=True,
                    help="model to score under (typically the source-completion's own model)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="K requests per engine call. Throughput is dominated by "
                         "continuous batching inside sglang, so chunk size only "
                         "affects how often progress is printed.")
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--max-running-requests", type=int, default=64)
    ap.add_argument("--debug", action="store_true",
                    help="dump per-row visual-dependency summary to stderr")
    ap.add_argument("--id-filter", type=Path, default=None,
                    help="Optional path to a JSON file mapping benchmark→[ids] "
                         "(or a flat list of ids). Only scores prompts whose id "
                         "is in this set. Use opd_target_ids.json from "
                         "paired_vision_critical to focus the H2 audit on the "
                         "vision-conditioned subset.")
    ap.add_argument("--system-prompt-text", default="",
                    help="Text prepended to the user turn as a separate text "
                         "block BEFORE the image. For MMR1 / MMR1-RL teacher "
                         "checkpoints this MUST be the verbatim training-time "
                         "system prompt (see scripts/audit/run_smoke.sh "
                         "MMR1_SYSTEM_PROMPT). Without it MMR1 stays in base-"
                         "model mode and the VD distribution will not match "
                         "the canonical audit run.")
    args = ap.parse_args()

    if args.system_prompt_text:
        first_60 = args.system_prompt_text.strip()[:60].replace("\n", " ")
        print(f">>> system_prompt_text active (first 60 chars): {first_60!r}...",
              file=sys.stderr)
    else:
        print(">>> WARNING: no --system-prompt-text. For MMR1-trained models "
              "this leaves the model in base-mode and the VD distribution "
              "will NOT be comparable to the canonical audit. Use the "
              "verbatim MMR1 training-time sysprompt for MMR1 teachers.",
              file=sys.stderr)

    # Same cuDNN init race precaution as run_audit_pass_sglang (see common-pitfalls E1).
    import torch  # noqa: F401
    torch.backends.cudnn.enabled = False

    from transformers import AutoTokenizer  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    src_by_id = _load_source_index(args.source)
    print(f">>> loaded {len(src_by_id)} source completions from {args.source}", file=sys.stderr)

    id_filter: set[str] | None = None
    if args.id_filter:
        raw = json.loads(args.id_filter.read_text())
        if isinstance(raw, dict):
            id_filter = set()
            for ids in raw.values():
                id_filter.update(ids)
        elif isinstance(raw, list):
            id_filter = set(raw)
        else:
            sys.exit(f"!! --id-filter expects JSON dict[bench→[ids]] or [ids], got {type(raw)}")
        print(f">>> id_filter active: {len(id_filter)} ids loaded from {args.id_filter}",
              file=sys.stderr)

    # Build per-record request payload (prefix + response text, full / blank images,
    # and a count of response tokens so we can slice the right tail from logprobs).
    requests: list[dict] = []
    skipped = 0
    for rec in _load_subset(args.subset):
        if args.limit and len(requests) >= args.limit:
            break
        rid = rec["id"]
        if id_filter is not None and rid not in id_filter:
            continue
        src = src_by_id.get(rid)
        if src is None:
            continue
        response = (src.get("prediction") or "").strip()
        if not response:
            continue
        pil = _load_subset_image(rec)
        if pil is None:
            skipped += 1
            continue

        full_img, _ = mllm_corruptions.apply_mode(pil, "full_image")
        blank_img, _ = mllm_corruptions.apply_mode(pil, "blank_image")

        prefix_text, full_text = _build_prefix_and_full(
            tokenizer, rec, full_img, response,
            system_prompt=args.system_prompt_text,
        )
        # Number of response tokens via standalone tokenization. Sglang's internal
        # tokenization expands image patches in the prefix region, but those live
        # BEFORE the response in the token stream, so the *last* n_resp logprobs
        # are still the response's regardless of how many image-patch tokens
        # sglang inserts.
        response_token_ids = tokenizer.encode(response, add_special_tokens=False)
        n_resp = len(response_token_ids)
        if n_resp == 0:
            skipped += 1
            continue

        requests.append({
            "id": rid,
            "benchmark": rec.get("benchmark"),
            "full_text": full_text,
            "image_full": full_img,
            "image_blank": blank_img,
            "response_token_ids": response_token_ids,
            "n_resp": n_resp,
        })

    print(f">>> {len(requests)} requests built, {skipped} skipped", file=sys.stderr)
    if not requests:
        args.out.touch()
        return

    print(f">>> launching sglang engine for {args.model}", file=sys.stderr)
    from sglang import Engine  # type: ignore
    engine = Engine(
        model_path=args.model,
        dtype="bfloat16",
        mem_fraction_static=args.mem_fraction,
        max_running_requests=args.max_running_requests,
        log_level="warning",
    )

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    t0 = time.time()

    with args.out.open("w") as fout:
        for chunk_start in range(0, len(requests), args.chunk_size):
            chunk = requests[chunk_start:chunk_start + args.chunk_size]
            prompts = [r["full_text"] for r in chunk]
            full_imgs = [r["image_full"] for r in chunk]
            blank_imgs = [r["image_blank"] for r in chunk]

            full_outs = _engine_generate(
                engine, prompts, full_imgs, sampling_params,
                return_logprob=True, logprob_start_len=0,
            )
            blank_outs = _engine_generate(
                engine, prompts, blank_imgs, sampling_params,
                return_logprob=True, logprob_start_len=0,
            )

            for r, fo, bo in zip(chunk, full_outs, blank_outs):
                full_logp_all = _extract_input_logprobs(fo)
                blank_logp_all = _extract_input_logprobs(bo)
                if full_logp_all is None or blank_logp_all is None:
                    n_written += 1
                    fout.write(json.dumps({
                        "id": r["id"], "error": "no_logprobs_in_output",
                    }, ensure_ascii=False) + "\n")
                    continue

                n_resp = r["n_resp"]
                if len(full_logp_all) < n_resp or len(blank_logp_all) < n_resp:
                    n_written += 1
                    fout.write(json.dumps({
                        "id": r["id"], "error": "logprobs_shorter_than_response",
                        "n_resp": n_resp,
                        "n_full": len(full_logp_all),
                        "n_blank": len(blank_logp_all),
                    }, ensure_ascii=False) + "\n")
                    continue

                logp_full = full_logp_all[-n_resp:]
                logp_blank = blank_logp_all[-n_resp:]
                vd = [a - b for a, b in zip(logp_full, logp_blank)]
                token_strs = [tokenizer.decode([t]) for t in r["response_token_ids"]]

                row = {
                    "id": r["id"],
                    "benchmark": r["benchmark"],
                    "model": args.model,
                    "source": str(args.source),
                    "n_tokens": n_resp,
                    "completion_tokens": r["response_token_ids"],
                    "token_strs": token_strs,
                    "logp_full": logp_full,
                    "logp_blank": logp_blank,
                    "visual_dependency": vd,
                    "logp_full_sum": sum(logp_full),
                    "logp_blank_sum": sum(logp_blank),
                    "vd_sum": sum(vd),
                    "vd_mean": sum(vd) / n_resp,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1

                if args.debug:
                    top_high = sorted(range(n_resp), key=lambda i: -vd[i])[:5]
                    top_low = sorted(range(n_resp), key=lambda i: vd[i])[:5]
                    print(f"=== {r['id']} (n={n_resp}, vd_sum={sum(vd):+.2f}, "
                          f"vd_mean={sum(vd)/n_resp:+.3f}) ===", file=sys.stderr)
                    print(f"  high VD: " + ", ".join(
                        f"{token_strs[i]!r}({vd[i]:+.2f})" for i in top_high
                    ), file=sys.stderr)
                    print(f"  low  VD: " + ", ".join(
                        f"{token_strs[i]!r}({vd[i]:+.2f})" for i in top_low
                    ), file=sys.stderr)

            fout.flush()
            os.fsync(fout.fileno())

            elapsed = time.time() - t0
            rate = n_written / max(1.0, elapsed)
            print(f"... {n_written}/{len(requests)} ({rate:.2f}/s)", file=sys.stderr)

    elapsed = time.time() - t0
    rate = n_written / max(1.0, elapsed)
    print(f">>> wrote {n_written} -> {args.out}  ({rate:.2f} rows/s, {elapsed:.0f}s)",
          file=sys.stderr)

    for name in ("shutdown", "stop", "close"):
        fn = getattr(engine, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


if __name__ == "__main__":
    main()
