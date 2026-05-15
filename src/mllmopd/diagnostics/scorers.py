"""Benchmark-aware lightweight scorers for the audit pipeline.

These are triage-only. Real headline numbers must come from lmms-eval. The job
here is to be honest enough that a per-cell accuracy in summary.json is not
totally noise — in particular, NOT scoring `"yes"` as correct when the gold is
`"yes"` and the prediction is a long CoT that happens to contain the word "yes".

Each `score_*` function returns:
    True / False  — confident score
    None          — gold cannot be parsed for this scorer (skip this record)

`score_for_benchmark` dispatches and also records which scorer fired, so we can
filter `loose_contains` cells out of any plot.
"""

from __future__ import annotations

import re

_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_MCQ_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Ha-h])(?![A-Za-z])")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Benchmarks where gold is yes/no.
_YESNO_BENCHMARKS = {"pope_adversarial", "pope", "hallusionbench"}


def _parse_yesno(text: str) -> str | None:
    """Return last yes/no in the prediction, lowercased."""
    m = _YESNO_RE.findall(text)
    return m[-1].lower() if m else None


def _parse_mcq_letter(text: str) -> str | None:
    """Return last standalone A-H letter, uppercased."""
    m = _MCQ_LETTER_RE.findall(text)
    return m[-1].upper() if m else None


def _parse_number(text: str) -> float | None:
    m = _NUMBER_RE.findall(text)
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def score_yesno(pred: str, gold) -> bool | None:
    if gold is None:
        return None
    g = str(gold).strip().lower()
    if g not in {"yes", "no"}:
        return None
    p = _parse_yesno(pred)
    return False if p is None else (p == g)


def score_mcq_letter(pred: str, gold) -> bool | None:
    if gold is None:
        return None
    g = str(gold).strip().upper()
    if not (len(g) == 1 and "A" <= g <= "H"):
        return None
    p = _parse_mcq_letter(pred)
    return False if p is None else (p == g)


def score_numeric(pred: str, gold, rel_tol: float = 0.01, abs_tol: float = 1e-3) -> bool | None:
    if gold is None:
        return None
    try:
        g = float(str(gold).strip().rstrip("%"))
    except ValueError:
        return None
    p = _parse_number(pred)
    if p is None:
        return False
    if g == 0:
        return abs(p) < abs_tol
    return abs(p - g) / abs(g) < rel_tol


def score_for_benchmark(benchmark: str, pred: str, gold) -> tuple[bool | None, str]:
    """Dispatch on benchmark name + gold shape.

    Returns (is_correct, scorer_name). Scorer name is one of:
        yesno / mcq_letter / numeric / loose_contains / skip_empty_gold
    """
    if gold is None or str(gold).strip() == "":
        return None, "skip_empty_gold"

    b = benchmark.lower()
    if any(k in b for k in _YESNO_BENCHMARKS):
        return score_yesno(pred, gold), "yesno"

    g = str(gold).strip()
    if len(g) == 1 and "A" <= g.upper() <= "H":
        return score_mcq_letter(pred, gold), "mcq_letter"

    try:
        float(g.rstrip("%"))
        return score_numeric(pred, gold), "numeric"
    except ValueError:
        pass

    p_low = pred.strip().lower()
    g_low = g.lower()
    return (g_low == p_low or g_low in p_low), "loose_contains"
