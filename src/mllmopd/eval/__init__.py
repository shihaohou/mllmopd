"""mllmopd.eval — unified evaluation utilities.

This subpackage hosts:

* :mod:`mllmopd.eval.blankness` — refusal / blank-template phrase detection
  on model outputs. The signal is a key mechanism marker for BlankTeacher
  students that develop "learned visual blindness" (cf. T1-3 trajectory).
* :mod:`mllmopd.eval.schema` — the unified per-prompt evaluation jsonl
  record schema (``v1``) used by all new eval runs starting from the
  Rethinking-OPD atlas pivot (2026-05-27).

Both modules are stdlib-only on purpose so they import cleanly on Mac
analysis boxes and inside the audit / train venvs alike.
"""

from mllmopd.eval.blankness import (
    BLANK_PATTERNS,
    BlanknessResult,
    analyze,
    detect_blankness,
    detect_blankness_in_think,
    detect_early_blankness,
)
from mllmopd.eval.schema import (
    SCHEMA_VERSION,
    PromptEvalRecord,
    from_existing_record,
)

__all__ = [
    "BLANK_PATTERNS",
    "BlanknessResult",
    "analyze",
    "detect_blankness",
    "detect_blankness_in_think",
    "detect_early_blankness",
    "SCHEMA_VERSION",
    "PromptEvalRecord",
    "from_existing_record",
]
