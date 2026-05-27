"""Blank / refusal template detection for model output strings.

Motivation
----------
BlankTeacher OPD (T1-3) produces a student that, after enough steps, refuses
to read the image and emits stock phrases like ``"the image appears to be
blank"`` or ``"I cannot determine from the image"``. This module turns that
mechanism marker into a per-prompt boolean feature so downstream eval
dashboards can plot ``blankness_rate(step)`` alongside accuracy.

The detector is intentionally stdlib-only (``re`` + ``dataclasses``) so it
imports on the Mac analysis box and inside the audit / train venvs alike.

The pattern list is extensible — new phrases can be appended to
:data:`BLANK_PATTERNS` without touching the call sites. Prior art lives in
``mllmopd.analysis.t1_blankness_trajectory.BLANK_PHRASES``; the patterns
below are a superset reviewed for the unified v1 schema.

Three detection variants
------------------------
* :func:`detect_blankness` — anywhere in the output. The default signal.
* :func:`detect_early_blankness` — only the first ``max_prefix_chars`` of
  the output. Late-tail blankness can show up in a verbose CoT that still
  ends up correct; the early-prefix variant is the one that actually
  correlates with on-policy prefix self-conditioning (the student's own
  prefix steering it toward the blank template).
* :func:`detect_blankness_in_think` — only inside the first
  ``<think>...</think>`` block (MMR1-style CoT). Catches the case where
  the visible answer looks fine but the chain of thought has already
  declared the image blank.

All matching is case-insensitive against the lowercased text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Pattern table
# ---------------------------------------------------------------------------
#
# Curated list of refusal / blank-template phrases. Each entry is a *plain
# lowercase substring* — matching is performed against ``text.lower()`` and
# the lookup is a simple ``in`` test. We do not use regex here because
#
#   (a) every pattern in the seed list is a plain phrase, not a regex; and
#   (b) substring-on-lowered-text is reliably faster than ``re.search`` for
#       a couple of dozen short patterns, which matters when we sweep this
#       over millions of eval rows.
#
# If you later need word-boundary or alternation patterns (e.g. ``\bblank\b``
# from the prior-art ``t1_blankness_trajectory``), add a separate regex list
# and OR its result into the detector. Keep the plain-substring list the
# stable, easy-to-grep surface area.
#
BLANK_PATTERNS: list[str] = [
    "blank image",
    "blank white",
    "white image",
    "completely white",
    "plain white",
    "empty image",
    "no visible",
    "cannot see",
    "can't see",
    "there is no image",
    "image is missing",
    "no information in the image",
    "unable to determine from the image",
    "image appears to be blank",
    "image is blank",
    "i don't see any",
    "i do not see any",
    "no discernible content",
    "no recognizable",
    "without an image",
    "no image is provided",
    "image cannot be seen",
]

# Pre-lowered for paranoia (the seed list is already lowercase, but a future
# editor may add a capitalised entry without realising matching is on lowered
# text). De-duplicated while preserving order.
_LOWERED_PATTERNS: tuple[str, ...] = tuple(dict.fromkeys(p.lower() for p in BLANK_PATTERNS))


# ``<think>...</think>`` block — non-greedy, ``DOTALL`` so newlines inside the
# block don't break the match. We only ever look at the **first** think block;
# subsequent ones are rare and tend to be the model re-thinking itself, which
# is less mechanism-relevant.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class BlanknessResult:
    """Output of :func:`analyze`.

    Attributes
    ----------
    blankness:
        True iff *any* pattern matches anywhere in the full text.
    early_blankness:
        True iff *any* pattern matches in the first
        ``early_blankness_max_chars`` (default 400) characters of the text.
        Captures prefix-self-conditioning during on-policy rollout.
    blankness_in_think:
        True iff *any* pattern matches inside the first
        ``<think>...</think>`` block. False if no think block exists.
    matched_patterns:
        Unique patterns that fired across the full text, in the order they
        appear in :data:`BLANK_PATTERNS`. Useful for figure legends + audit.
    """

    blankness: bool
    early_blankness: bool
    blankness_in_think: bool
    matched_patterns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _iter_matches(lowered: str) -> list[str]:
    """Return the patterns (in canonical order) that occur in ``lowered``."""
    return [p for p in _LOWERED_PATTERNS if p in lowered]


def detect_blankness(text: str) -> bool:
    """True iff any :data:`BLANK_PATTERNS` entry appears in ``text``.

    Matching is case-insensitive. Non-string / empty inputs return ``False``
    so callers don't need to guard against ``None`` predictions.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(p in lowered for p in _LOWERED_PATTERNS)


def detect_early_blankness(text: str, max_prefix_chars: int = 400) -> bool:
    """Like :func:`detect_blankness` but restricted to the leading
    ``max_prefix_chars`` characters of the output.

    Rationale: on-policy prefix self-conditioning is the failure mode where
    the student's own opening words push it toward the blank template. Late
    blankness (e.g. a footnote in a 4k-token CoT that mostly answers the
    question correctly) is much less mechanism-relevant; the early-prefix
    variant is the one to plot against training step.

    Boundary semantics: a pattern must lie *entirely* inside the
    ``[:max_prefix_chars]`` slice to count as early. A phrase that starts
    at e.g. char 395 but extends past the boundary is treated as late.
    This is conservative on purpose — the mechanism marker we care about
    is a phrase that fits fully into the prefix.
    """
    if not text:
        return False
    # Slice *before* lowering — avoids allocating a giant lowered string when
    # ``text`` is long and we only need the prefix.
    prefix = text[:max_prefix_chars].lower()
    return any(p in prefix for p in _LOWERED_PATTERNS)


def detect_blankness_in_think(text: str) -> bool:
    """True iff a blank pattern occurs inside the first ``<think>...</think>``
    block. False if no think block is present (rather than scanning the
    whole text — the caller should use :func:`detect_blankness` for that)."""
    if not text:
        return False
    m = _THINK_RE.search(text)
    if not m:
        return False
    body = m.group(1).lower()
    return any(p in body for p in _LOWERED_PATTERNS)


def analyze(text: str, early_blankness_max_chars: int = 400) -> BlanknessResult:
    """Run all three detectors + collect matched patterns in one pass.

    More efficient than calling the three ``detect_*`` functions separately
    when the caller needs the full result — we only lower the text once.
    """
    if not text:
        return BlanknessResult(
            blankness=False,
            early_blankness=False,
            blankness_in_think=False,
            matched_patterns=[],
        )

    lowered = text.lower()
    matches = _iter_matches(lowered)

    prefix_lowered = lowered[:early_blankness_max_chars]
    early = any(p in prefix_lowered for p in _LOWERED_PATTERNS)

    think_m = _THINK_RE.search(text)
    if think_m:
        # Use the lowered version of the captured group, computed once.
        think_lowered = think_m.group(1).lower()
        in_think = any(p in think_lowered for p in _LOWERED_PATTERNS)
    else:
        in_think = False

    return BlanknessResult(
        blankness=bool(matches),
        early_blankness=early,
        blankness_in_think=in_think,
        matched_patterns=matches,
    )
