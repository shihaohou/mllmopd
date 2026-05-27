"""Unified per-prompt evaluation record schema (``v1``).

This is the single shape that all *new* eval runs starting from the
Rethinking-OPD-atlas pivot (2026-05-27) emit, one record per line in a
``.jsonl`` file. The full prose spec lives in ``docs/eval_schema_v1.md``;
this module exposes the same shape in code so callers can validate and
coerce records at runtime.

Why TypedDict and not Pydantic
------------------------------
The project ``pyproject.toml`` lists only stdlib-adjacent deps (``numpy``,
``pandas``, ``matplotlib``, ``pyyaml``, ``datasets``, ``huggingface-hub``,
``pillow``, ``tqdm``). Pydantic is not a current dependency, and the
maintainers want this subpackage to import on the Mac analysis box and
inside the audit / train venvs without pulling extra wheels. A
``TypedDict`` plus a coercion helper gives us:

* IDE / mypy-friendly typing for record producers,
* zero new runtime deps,
* easy round-tripping to/from plain ``dict`` (which is what ``json.dumps``
  and pandas / parquet want anyway).

If we later adopt Pydantic for richer validation, the ``TypedDict`` keys
match what a ``BaseModel`` would expose, so the migration is mechanical.

The schema mirrors ``docs/eval_schema_v1.md`` field-for-field; update both
files when adding or renaming fields.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict, get_type_hints

SCHEMA_VERSION: str = "v1"

# Tolerance for `hit_max_tokens` field across all eval drivers. A response
# is considered to have "hit max tokens" if num_tokens >= max_new_tokens
# - HIT_MAX_TOKENS_TOLERANCE. Pinned project-wide to prevent driver drift.
HIT_MAX_TOKENS_TOLERANCE: int = 4


def compute_hit_max_tokens(num_tokens: int, max_new_tokens: int) -> bool:
    """Whether a response is at or near the configured max_new_tokens cap.

    Uses the project-wide tolerance to be permissive about tokenizer-level
    off-by-ones (a response that emits exactly 2044 tokens when the cap is
    2048 should still count as "hit the cap" for length-cliff analyses).
    """
    return num_tokens >= max_new_tokens - HIT_MAX_TOKENS_TOLERANCE


# ---------------------------------------------------------------------------
# Enum-like value sets (string Literals)
# ---------------------------------------------------------------------------
TeacherMode = Literal[
    "full",
    "blank",
    "text",
    "offline_full",
    "offline_blank",
    "none",
]

Policy = Literal["on_policy", "off_policy", "none"]

Mode = Literal["full_image", "blank_image", "text_only"]


# ---------------------------------------------------------------------------
# Record TypedDict
# ---------------------------------------------------------------------------
class PromptEvalRecord(TypedDict, total=False):
    """One row of an eval jsonl, ``v1`` schema.

    ``total=False`` (every key is technically optional at the type level)
    because:

    * not every eval set has gold answers (open-ended HallusionBench rows),
    * some scorers don't emit ``parse_path`` (judge-only scoring),
    * older eval jsonls migrated through :func:`from_existing_record` may
      have unfilled fields.

    Producers of *new* records should fill **every** field — use ``None``
    or ``False`` for explicit "unknown" rather than dropping the key, so
    pandas / parquet consumers see a stable column set.
    """

    # ------ identity ------
    arm: str                  # e.g. "S0", "T1-Full", "T1-Blank", "T2-1", "T2-2", "Teacher"
    student: str              # student model name, e.g. "MMR1-3B-SFT"
    teacher_mode: TeacherMode
    policy: Policy
    step: int                 # training step (0 = base / pre-training, 230 = final)
    eval_set: str             # "dev_mmr1_v0_1k" | "level1_subset_v0" | "opd_target_133" | "lmmseval/<bench>"
    benchmark: str            # "MathVista" | "ChartQA" | "POPE_adv" | "HallusionBench" | "MMR1-RL_dev" | ...
    mode: Mode

    # ------ prompt + answer ------
    id: str                   # prompt id (dev_mmr1_v0_***, opd_target_***, level1 row id, ...)
    prediction: str           # full model output (raw, untrimmed)
    gold: str | None          # ground truth answer text; None for open-ended
    parsed_answer: str | None # extracted answer after parsing
    is_correct: bool | None   # True/False/None (None when scoring skipped / gold absent)
    scorer: str               # "claude-sonnet-4-6" | "rule_based" | "string_match" | "mathvista_rule" | "mcq_letter" | ...

    # ------ output statistics ------
    num_tokens: int           # output token count
    hit_max_tokens: bool      # num_tokens >= max_new_tokens - tolerance

    # ------ mechanism signals ------
    blankness: bool
    early_blankness: bool
    blankness_in_think: bool

    # ------ provenance ------
    parse_path: str           # "boxed_regex" | "answer_tag_split" | "raw" | scorer-specific tag
    extras: dict[str, Any]    # free-form, eval-set-specific (image_path, response_time, prompt_len, ...)


# Canonical key order — controls column order in pandas dumps + the
# canonical jsonl key order. Keep in sync with PromptEvalRecord above and
# docs/eval_schema_v1.md.
FIELD_ORDER: tuple[str, ...] = (
    "arm",
    "student",
    "teacher_mode",
    "policy",
    "step",
    "eval_set",
    "benchmark",
    "mode",
    "id",
    "prediction",
    "gold",
    "parsed_answer",
    "is_correct",
    "scorer",
    "num_tokens",
    "hit_max_tokens",
    "blankness",
    "early_blankness",
    "blankness_in_think",
    "parse_path",
    "extras",
)


# ---------------------------------------------------------------------------
# Legacy field aliases (older jsonls use different names)
# ---------------------------------------------------------------------------
#
# Each (alias -> v1-key) mapping is consulted when coercing an existing
# record via :func:`from_existing_record`. Only add entries here for fields
# we've actually observed in past eval jsonls — never speculative aliases.
#
_LEGACY_ALIASES: dict[str, str] = {
    # run_audit_pass.py / run_audit_pass_sglang.py emit "model" but v1
    # prefers "student" (the model under eval). The teacher arm uses the
    # same column.
    "model": "student",
    # Old runs emitted just "answer" instead of "gold".
    "answer": "gold",
    # Some lmms-eval wrappers emitted "response" / "output" instead of
    # "prediction".
    "response": "prediction",
    "output": "prediction",
}


def _default_extras_keys() -> set[str]:
    """Field names that should land in ``extras`` rather than be dropped
    when present in an incoming legacy record."""
    return {
        "prompt_len",
        "choices",
        "error",
        "image_path",
        "response_time",
        "rescored_is_correct",
        "rescore_changed",
    }


# ---------------------------------------------------------------------------
# Coercion: legacy dict -> v1 record
# ---------------------------------------------------------------------------
def from_existing_record(rec: dict[str, Any]) -> PromptEvalRecord:
    """Coerce a legacy per-prompt eval dict into the v1 schema shape.

    Behaviour:

    * Recognised legacy aliases (see :data:`_LEGACY_ALIASES`) are renamed.
    * Missing required-shape fields are filled with conservative defaults
      (``None`` for nullable strings, ``False`` for boolean mechanism
      signals, ``0`` for counts, empty string for identity slots we can't
      infer).
    * Any extra keys that aren't part of v1 land inside ``extras`` (the
      free-form bag) rather than being dropped silently — this preserves
      provenance for the audit pipeline.

    The function does **not** *compute* mechanism signals from the
    prediction string (``blankness`` etc.) — that's the producer's job at
    write time. If the legacy record didn't run a blankness scan, the
    fields stay ``False``. A separate pass can re-fill them by feeding
    each ``prediction`` through :func:`mllmopd.eval.blankness.analyze`.
    """
    if not isinstance(rec, dict):
        raise TypeError(f"from_existing_record expects a dict, got {type(rec).__name__}")

    # 1) Apply aliases. We don't mutate the caller's dict.
    src: dict[str, Any] = {}
    for k, v in rec.items():
        key = _LEGACY_ALIASES.get(k, k)
        # Don't clobber a real v1 key with an alias-renamed one (defensive
        # in case the caller already filled the v1 name).
        if key in src and key != k:
            continue
        src[key] = v

    # 2) Collect extras: anything not in the v1 field set goes into extras.
    #    Pre-existing ``extras`` dict (if any) gets merged in too.
    extras: dict[str, Any] = {}
    pre_extras = src.pop("extras", None)
    if isinstance(pre_extras, dict):
        extras.update(pre_extras)

    v1_keys = set(FIELD_ORDER)
    # ``_default_extras_keys`` is kept as a hint to future maintainers
    # that these names have an "official" home in ``extras`` — but any
    # unknown key is preserved regardless, since ``extras`` is free-form.
    unknown_keys = [k for k in list(src.keys()) if k not in v1_keys]
    for k in unknown_keys:
        extras[k] = src.pop(k)

    # 3) Fill required-ish defaults. None / False / 0 / "" preserves the
    #    "explicit unknown" semantics the producers should follow.
    out: PromptEvalRecord = {  # type: ignore[typeddict-item]
        "arm": src.get("arm", ""),
        "student": src.get("student", ""),
        "teacher_mode": src.get("teacher_mode", "none"),
        "policy": src.get("policy", "none"),
        "step": int(src["step"]) if isinstance(src.get("step"), (int, float)) else 0,
        "eval_set": src.get("eval_set", ""),
        "benchmark": src.get("benchmark", ""),
        "mode": src.get("mode", "full_image"),
        "id": str(src.get("id", "")),
        "prediction": src.get("prediction", "") or "",
        "gold": src.get("gold"),
        "parsed_answer": src.get("parsed_answer"),
        "is_correct": src.get("is_correct"),
        "scorer": src.get("scorer", ""),
        "num_tokens": int(src.get("num_tokens") or 0),
        "hit_max_tokens": bool(src.get("hit_max_tokens", False)),
        "blankness": bool(src.get("blankness", False)),
        "early_blankness": bool(src.get("early_blankness", False)),
        "blankness_in_think": bool(src.get("blankness_in_think", False)),
        "parse_path": src.get("parse_path", "") or "",
        "extras": extras,
    }
    return out


# ---------------------------------------------------------------------------
# Convenience: produce a freshly-defaulted record
# ---------------------------------------------------------------------------
def empty_record() -> PromptEvalRecord:
    """Return a new ``PromptEvalRecord`` filled with the v1 defaults.

    Useful as a starting point in eval drivers: fill the known fields and
    write out as jsonl without worrying about missing keys.
    """
    return from_existing_record({})


def schema_types() -> dict[str, Any]:
    """Return the resolved type hints for :class:`PromptEvalRecord`.

    Evaluated lazily — under ``from __future__ import annotations`` (which
    we use for forward-compatible ``str | None`` syntax) the annotations
    are strings until ``typing.get_type_hints`` resolves them, and that
    call needs Python 3.10+ to evaluate PEP 604 unions. Calling this from
    a 3.10+ runtime is fine; from 3.9 it will raise — which is exactly
    what ``pyproject.toml`` says (``requires-python = ">=3.10"``).
    """
    return get_type_hints(PromptEvalRecord)


__all__ = [
    "SCHEMA_VERSION",
    "FIELD_ORDER",
    "TeacherMode",
    "Policy",
    "Mode",
    "PromptEvalRecord",
    "empty_record",
    "from_existing_record",
    "schema_types",
]
