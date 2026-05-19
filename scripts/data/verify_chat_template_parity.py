"""T1 punch list #3 / Risk #2: byte-level verify chat-template parity.

For a sample row from data/opd_train/v0_2k/train.jsonl, render the
prompt two ways:

  AUDIT path     : run_audit_pass._build_messages(question, image,
                   system_prompt=MMR1_SYSTEM_PROMPT)
                   → tokenizer.apply_chat_template(..., add_generation_prompt=True)
  UNI-OPD path   : miles.utils.data._build_messages(problem, images,
                   prompt_key="problem", multimodal_keys={"image": "images"})
                   → tokenizer.apply_chat_template(...)

The prep pipeline prepends MMR1_SYSTEM_PROMPT to `problem` so that
Uni-OPD's regex split on `<image>` should yield the same content list
as the audit's manual `[text, image, text]` assembly. This script
asserts byte-identical rendering. If the strings diverge, T1 is in
"base model mode" and the negative control is invalid.

Pass criteria: byte-identical rendered text + byte-identical token ids
on 5 sampled prep rows.

Usage (devbox or local, audit venv — needs transformers + PIL,
no GPU / sglang):
    python scripts/data/verify_chat_template_parity.py \\
        --train-jsonl data/opd_train/v0_2k/train.jsonl \\
        --model /home/web_server/antispam/project/houshihao/models/MMR1-7B-RL \\
        --n-samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _render_audit(messages_fn, tokenizer, raw_problem: str, image, sysprompt: str) -> str:
    """Reproduce the audit path: pass the *unprefixed* question text +
    image + explicit system_prompt to audit's _build_messages, then
    apply_chat_template."""
    # MMR1-RL's raw_problem is "<image>\n<question>". Strip the leading
    # placeholder + optional newline; what remains is the question text
    # the audit pipeline would store under rec["question"].
    if raw_problem.startswith("<image>"):
        question_text = raw_problem[len("<image>"):]
    else:
        # Fall back: just remove first occurrence.
        question_text = raw_problem.replace("<image>", "", 1)
    rec = {"question": question_text}
    messages = messages_fn(rec, image, prefix=None, system_prompt=sysprompt)
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


def _render_uniopd(messages_fn, tokenizer, problem: str, image) -> str:
    """Reproduce Uni-OPD's path: pass the sysprompt-prefixed problem
    (which still contains `<image>`) + multimodal_keys mapping.
    Uni-OPD's _build_messages will regex-split on <image>."""
    data = {"problem": problem, "images": [image]}
    messages = messages_fn(
        data, prompt_key="problem", as_conversation=True,
        multimodal_keys={"image": "images"},
    )
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


def _first_diff(a: str, b: str) -> tuple[int, str, str]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            start = max(0, i - 40)
            end_a = min(len(a), i + 40)
            end_b = min(len(b), i + 40)
            return i, a[start:end_a], b[start:end_b]
    if len(a) != len(b):
        return n, a[n:n+80], b[n:n+80]
    return -1, "", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", type=Path, required=True,
                    help="Output of prep_opd_train_data.py")
    ap.add_argument("--model", required=True,
                    help="Tokenizer source (MMR1-7B-RL or any Qwen2.5-VL "
                         "checkpoint with the same chat template)")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--uni-opd-root", type=Path,
                    default=Path("third_party/Uni-OPD/miles"),
                    help="Path to the Uni-OPD `miles` package root (the dir "
                         "containing the `miles/` subpackage). Added to "
                         "sys.path so we can import miles.utils.data.")
    args = ap.parse_args()

    # Import-side setup. Audit's _build_messages comes from this repo;
    # Uni-OPD's from the submodule.
    sys.path.insert(0, str(args.uni_opd_root.resolve()))
    try:
        from miles.utils.data import _build_messages as uniopd_build  # type: ignore
    except Exception as e:
        sys.exit(f"ERROR: could not import miles.utils.data._build_messages: {e}\n"
                 f"  expected uni-opd-root: {args.uni_opd_root.resolve()}\n"
                 f"  ensure the Uni-OPD submodule is initialized.")

    from mllmopd.diagnostics.run_audit_pass import _build_messages as audit_build

    # MMR1 sysprompt — must match the one prep_opd_train_data.py prepended.
    from scripts.data.prep_opd_train_data import MMR1_SYSTEM_PROMPT  # type: ignore

    from transformers import AutoTokenizer  # type: ignore
    print(f">>> loading tokenizer from {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    from PIL import Image  # type: ignore

    # Load N training rows.
    rows: list[dict] = []
    with args.train_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= args.n_samples:
                break
    print(f">>> loaded {len(rows)} sample rows from {args.train_jsonl}",
          file=sys.stderr)

    all_pass = True
    for i, row in enumerate(rows):
        problem = row["problem"]
        raw_problem = row["raw_problem"]
        img_path = row["images"][0]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            audit_text = _render_audit(
                audit_build, tokenizer, raw_problem, img, MMR1_SYSTEM_PROMPT,
            )
            uniopd_text = _render_uniopd(
                uniopd_build, tokenizer, problem, img,
            )

        # Byte-level equality.
        match_text = audit_text == uniopd_text
        audit_ids = tokenizer.encode(audit_text, add_special_tokens=False)
        uniopd_ids = tokenizer.encode(uniopd_text, add_special_tokens=False)
        match_ids = audit_ids == uniopd_ids

        status = "PASS" if (match_text and match_ids) else "FAIL"
        print(f"--- row {i} (id={row['id']}) — {status} ---")
        print(f"  audit_len_chars   = {len(audit_text)}")
        print(f"  uniopd_len_chars  = {len(uniopd_text)}")
        print(f"  audit_len_tokens  = {len(audit_ids)}")
        print(f"  uniopd_len_tokens = {len(uniopd_ids)}")
        print(f"  text_byte_equal   = {match_text}")
        print(f"  token_ids_equal   = {match_ids}")

        if not match_text:
            diff_idx, audit_ctx, uniopd_ctx = _first_diff(audit_text, uniopd_text)
            print(f"  first text diff at char {diff_idx}:")
            print(f"    audit  : ...{audit_ctx!r}...")
            print(f"    uni-opd: ...{uniopd_ctx!r}...")
            all_pass = False
        if not match_ids:
            # Find first divergent token id.
            n = min(len(audit_ids), len(uniopd_ids))
            div_at = next((j for j in range(n) if audit_ids[j] != uniopd_ids[j]),
                          n)
            window = (max(0, div_at - 5), min(n, div_at + 5))
            print(f"  first token diff at pos {div_at}:")
            print(f"    audit  ids[{window[0]}:{window[1]}] = "
                  f"{audit_ids[window[0]:window[1]]}")
            print(f"    uni-opd ids[{window[0]}:{window[1]}] = "
                  f"{uniopd_ids[window[0]:window[1]]}")
            for j in range(window[0], window[1]):
                ja = audit_ids[j] if j < len(audit_ids) else None
                jb = uniopd_ids[j] if j < len(uniopd_ids) else None
                marker = " <-- DIFF" if ja != jb else ""
                print(f"      pos {j}: audit={ja!r}={tokenizer.decode([ja]) if ja is not None else None!r}  "
                      f"uniopd={jb!r}={tokenizer.decode([jb]) if jb is not None else None!r}{marker}")
            all_pass = False

    print()
    if all_pass:
        print(">>> All samples byte-identical between AUDIT and UNI-OPD chat-template renderings.")
        print(">>> Risk #2 closed. Punch list #3 PASSED.")
        sys.exit(0)
    else:
        print(">>> FAIL: chat-template renderings diverge. Do not launch T1 until reconciled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
