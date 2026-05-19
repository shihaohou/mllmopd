"""Benchmark-aware lightweight scorers for the audit pipeline.

These are triage-only. Real headline numbers must come from lmms-eval. The job
here is to be honest enough that a per-cell accuracy in summary.json is not
totally noise — in particular, NOT scoring `"yes"` as correct when the gold is
`"yes"` and the prediction is a long CoT that happens to contain the word "yes".

Each `score_*` function returns:
    True / False  — confident score
    None          — gold cannot be parsed for this scorer (skip this record)

`score_for_benchmark` dispatches and also records which scorer fired plus the
parse path used (so we can filter low-confidence `last_letter_fallback` rows
out of any plot).
"""

from __future__ import annotations

import re

_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_MCQ_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Ha-h])(?![A-Za-z])")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LETTER_LABELS = "ABCDEFGHIJ"

# Benchmarks where gold is yes/no.
_YESNO_BENCHMARKS = {"pope_adversarial", "pope", "hallusionbench"}

# Priority patterns for MCQ letter extraction. Each is tried in order; the LAST
# match within a pattern is taken (more robust to long preambles that enumerate
# options before stating the final answer). The label flows out as parse_path
# so aggregations can separate high-confidence rows from fallback rows.
_MCQ_PRIORITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\boxed\{\s*\(?([A-Ha-h])\)?\s*\}"), "boxed"),
    (re.compile(r"final answer[\s:]*(?:is\s*)?\(?([A-Ha-h])\)?(?![A-Za-z])", re.IGNORECASE), "final_answer"),
    (re.compile(r"correct answer[\s:]*(?:is\s*)?\(?([A-Ha-h])\)?(?![A-Za-z])", re.IGNORECASE), "correct_answer"),
    (re.compile(r"correct (?:choice|option)[\s:]*(?:is\s*)?\(?([A-Ha-h])\)?(?![A-Za-z])", re.IGNORECASE), "correct_choice"),
    (re.compile(r"\banswer[\s:]+(?:is\s*)?\(?([A-Ha-h])\)?(?![A-Za-z])", re.IGNORECASE), "answer_phrase"),
    (re.compile(r"\boption\s+\(?([A-Ha-h])\)?(?![A-Za-z])", re.IGNORECASE), "option_phrase"),
]

# Refusal phrases — used by diagnostics, not for scoring.
_REFUSAL_RE = re.compile(
    r"(?:i\s+(?:cannot|can'?t|am\s+unable|am\s+not\s+able)\s+(?:see|view|access|analyze|examine))"
    r"|(?:as\s+an?\s+(?:ai|language\s+model))"
    r"|(?:i\s+don'?t\s+(?:have\s+(?:access|the\s+ability)|see))"
    r"|(?:without\s+(?:seeing|access\s+to|the\s+image))"
    r"|(?:no\s+image\s+(?:was|is)\s+provided)"
    r"|(?:please\s+provide\s+(?:an?\s+)?image)",
    re.IGNORECASE,
)


def _parse_yesno(text: str) -> str | None:
    m = _YESNO_RE.findall(text)
    return m[-1].lower() if m else None


def _parse_mcq_priority(text: str) -> tuple[str | None, str]:
    """MCQ letter extraction with priority patterns.

    Returns (letter, parse_path). Higher-priority matches override lower ones,
    so the parser is robust to long CoTs that list "(A)..(D)" before the final
    answer. Falls back to the legacy "last standalone A-H" only when nothing
    else fires — those rows can be filtered out via parse_path.
    """
    for pat, label in _MCQ_PRIORITY_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            return matches[-1].group(1).upper(), label

    # Medium-confidence: last parenthesized letter in the final 300 chars
    tail = text[-300:]
    paren = list(re.finditer(r"\(([A-Ha-h])\)", tail))
    if paren:
        return paren[-1].group(1).upper(), "paren_tail"

    # Low-confidence: last standalone A-H anywhere (legacy behavior)
    m = _MCQ_LETTER_RE.findall(text)
    if m:
        return m[-1].upper(), "last_letter_fallback"
    return None, "none"


# Parse paths considered "high confidence" — explicit final-answer markers.
HIGH_CONFIDENCE_PATHS = frozenset({
    "boxed", "final_answer", "correct_answer", "correct_choice",
    "answer_phrase", "option_phrase",
    "choice_text",  # mapped option text → letter via choices list
})


def _parse_mcq_with_choices(text: str, choices) -> tuple[str | None, str]:
    """Like `_parse_mcq_priority`, but inserts an option-text → letter mapping
    step between the high-confidence answer phrases and the paren_tail fallback.

    Use this when the source record carries the MCQ `choices` list. The
    motivation: long-CoT models often conclude with the option *text*
    ("Serrulate is the correct answer") instead of restating the letter, in
    which case `_parse_mcq_priority` falls through to paren_tail which picks up
    the last enumerated option letter from the option list — usually wrong.
    """
    # Step 1: high-confidence priority patterns (same as the no-choices parser)
    for pat, label in _MCQ_PRIORITY_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            return matches[-1].group(1).upper(), label

    # Step 2: option text appearing in the prediction tail. The latest match
    # wins (model usually states its choice last); ties broken by longest text
    # (so "Serrulate" beats "Serrate" when both are in the tail).
    if choices:
        tail = text[-400:].lower()
        hits: list[tuple[int, int, int]] = []  # (position, choice_idx, length)
        for i, ch in enumerate(choices):
            if not ch:
                continue
            c = str(ch).strip().lower()
            if len(c) < 2:  # too short, false-positive risk
                continue
            pos = tail.rfind(c)
            if pos >= 0:
                hits.append((pos, i, len(c)))
        if hits:
            hits.sort(key=lambda x: (-x[0], -x[2]))  # latest, then longest
            _, idx, _ = hits[0]
            if idx < len(_LETTER_LABELS):
                return _LETTER_LABELS[idx], "choice_text"

    # Step 3: paren_tail
    tail = text[-300:]
    paren = list(re.finditer(r"\(([A-Ha-h])\)", tail))
    if paren:
        return paren[-1].group(1).upper(), "paren_tail"

    # Step 4: last_letter_fallback
    m = _MCQ_LETTER_RE.findall(text)
    if m:
        return m[-1].upper(), "last_letter_fallback"
    return None, "none"


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
    """Back-compat: returns bool only. Prefer `score_for_benchmark` which
    also surfaces the parse path."""
    res, _ = score_mcq_letter_v2(pred, gold)
    return res


def score_mcq_letter_v2(pred: str, gold, choices=None) -> tuple[bool | None, str]:
    """Returns (is_correct, parse_path). When `choices` is provided, uses the
    extended parser that can recover option-text-only conclusions."""
    if gold is None:
        return None, "skip_empty_gold"
    g = str(gold).strip().upper()
    if not (len(g) == 1 and "A" <= g <= "H"):
        return None, "skip_empty_gold"
    if choices:
        p, path = _parse_mcq_with_choices(pred, choices)
    else:
        p, path = _parse_mcq_priority(pred)
    if p is None:
        return False, path
    return (p == g), path


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


def score_for_benchmark(
    benchmark: str, pred: str, gold, choices=None,
) -> tuple[bool | None, str, str]:
    """Dispatch on benchmark name + gold shape.

    Returns (is_correct, scorer_name, parse_path). scorer_name is one of:
        yesno / mcq_letter / numeric / loose_contains / skip_empty_gold
    parse_path is meaningful for mcq_letter; for other scorers it equals the
    scorer name. Pass `choices` (a list of option-text strings, in A/B/C/...
    order) when available; the MCQ parser uses them to recover predictions
    that conclude with option text instead of letter.
    """
    if gold is None or str(gold).strip() == "":
        return None, "skip_empty_gold", "skip_empty_gold"

    b = benchmark.lower()
    if any(k in b for k in _YESNO_BENCHMARKS):
        return score_yesno(pred, gold), "yesno", "yesno"

    g = str(gold).strip()
    if len(g) == 1 and "A" <= g.upper() <= "H":
        res, path = score_mcq_letter_v2(pred, gold, choices=choices)
        return res, "mcq_letter", path

    try:
        float(g.rstrip("%"))
        return score_numeric(pred, gold), "numeric", "numeric"
    except ValueError:
        pass

    p_low = pred.strip().lower()
    g_low = g.lower()
    return (g_low == p_low or g_low in p_low), "loose_contains", "loose_contains"


def is_refusal(pred: str) -> bool:
    """Diagnostic helper: detects "I cannot see / as an AI language model / …"
    patterns that indicate the model refused to answer (instead of attempting).
    Used by aggregate_audit for the refusal_rate per cell."""
    return bool(_REFUSAL_RE.search(pred or ""))
