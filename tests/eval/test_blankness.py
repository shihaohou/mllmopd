"""Unit tests for ``mllmopd.eval.blankness``.

Coverage:

* every seed pattern in :data:`mllmopd.eval.blankness.BLANK_PATTERNS` is
  detected at least once;
* clean text (no pattern) returns all-False;
* case insensitivity (upper/mixed case input);
* the early-prefix detector honours the ``max_prefix_chars`` boundary;
* the think-block detector fires only when the phrase is inside the
  first ``<think>...</think>`` block;
* :func:`analyze` returns all three signals plus the matched pattern
  list in canonical order.
"""

from __future__ import annotations

import pytest

from mllmopd.eval.blankness import (
    BLANK_PATTERNS,
    BlanknessResult,
    analyze,
    detect_blankness,
    detect_blankness_in_think,
    detect_early_blankness,
)


# ---------------------------------------------------------------------------
# detect_blankness — positive coverage of every seed pattern
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pattern", BLANK_PATTERNS)
def test_every_seed_pattern_fires(pattern: str) -> None:
    sample = f"My answer: {pattern}, therefore none."
    assert detect_blankness(sample) is True, f"pattern not detected: {pattern!r}"


# ---------------------------------------------------------------------------
# detect_blankness — negative cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "clean_text",
    [
        "The cat is on the mat.",
        "The answer is 42.",
        "The chart shows a steady increase from 10 to 80.",
        "I see a red car next to the building.",
        "",  # empty
    ],
)
def test_clean_text_returns_false(clean_text: str) -> None:
    assert detect_blankness(clean_text) is False


def test_none_safe() -> None:
    # The detectors must not crash on falsy / non-string-looking inputs the
    # producer might forward when ``prediction`` is missing.
    assert detect_blankness("") is False
    # mypy would complain about passing None, but real eval pipelines do
    # this in practice — the function should swallow it.
    assert detect_blankness(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "BLANK IMAGE provided.",
        "The Image Is Blank.",
        "I CANNOT SEE anything here.",
        "i Don't see ANY chart in the figure.",
    ],
)
def test_case_insensitive(text: str) -> None:
    assert detect_blankness(text) is True


# ---------------------------------------------------------------------------
# detect_early_blankness — boundary handling
# ---------------------------------------------------------------------------
def test_early_blankness_within_default_window() -> None:
    text = "the image is blank, sorry I cannot help."
    assert detect_early_blankness(text) is True


def test_early_blankness_outside_default_window() -> None:
    # Push the phrase past the 400-char default window.
    filler = "x" * 500
    text = filler + " " + "the image is blank"
    assert detect_blankness(text) is True
    assert detect_early_blankness(text) is False


def test_early_blankness_custom_window() -> None:
    # Same text, but pass a bigger window so the late phrase is captured.
    filler = "x" * 500
    text = filler + " " + "the image is blank"
    assert detect_early_blankness(text, max_prefix_chars=600) is True


def test_early_blankness_fully_inside_boundary() -> None:
    # The pattern fits entirely within the 400-char prefix.
    # "the image is blank" is 18 chars; with 380 chars of filler the phrase
    # ends at char 398, inside the [:400] slice.
    filler = "x" * 380
    text = filler + "the image is blank and so on."
    assert detect_early_blankness(text) is True


def test_early_blankness_straddling_boundary_is_excluded() -> None:
    # The pattern starts before char 400 but extends past it. By design
    # ``detect_early_blankness`` uses a plain ``text[:max_prefix_chars]``
    # slice — patterns must fit *entirely* inside the prefix to count.
    # This is the conservative choice (the prefix-self-conditioning
    # mechanism is about the phrase appearing fully early, not just
    # starting to appear).
    filler = "x" * 395
    text = filler + "the image is blank and so on."
    assert detect_early_blankness(text) is False
    # The full-text detector still fires.
    assert detect_blankness(text) is True


def test_early_blankness_just_outside_boundary() -> None:
    # The pattern starts AT char 400 (the slice is exclusive) — outside.
    filler = "x" * 400
    text = filler + "the image is blank"
    assert detect_early_blankness(text) is False
    # But the same content fires the full-text detector.
    assert detect_blankness(text) is True


# ---------------------------------------------------------------------------
# detect_blankness_in_think
# ---------------------------------------------------------------------------
def test_blankness_only_in_think_block() -> None:
    text = (
        "<think>The image is blank, so I really cannot determine the answer.</think>\n"
        "The answer is 7."
    )
    assert detect_blankness_in_think(text) is True
    assert detect_blankness(text) is True


def test_blankness_only_after_think_block() -> None:
    # Think block is innocuous; the blank phrase is in the visible answer
    # afterwards. The think-block detector must return False.
    text = (
        "<think>Let me solve this step by step. The chart shows X.</think>\n"
        "I cannot see the chart clearly."
    )
    assert detect_blankness_in_think(text) is False
    assert detect_blankness(text) is True


def test_no_think_block_returns_false() -> None:
    text = "the image is blank and I am stuck."
    assert detect_blankness_in_think(text) is False
    # ...but the regular detector still fires.
    assert detect_blankness(text) is True


def test_think_block_case_insensitive() -> None:
    text = "<THINK>The Image Is Blank.</THINK>\nanswer."
    assert detect_blankness_in_think(text) is True


def test_only_first_think_block_is_consulted() -> None:
    # The mechanism marker is *prefix* self-conditioning, so we look at the
    # first think block. A clean first think + a blank-mentioning second
    # think shouldn't fire the in-think detector.
    text = (
        "<think>All looks fine, the bar chart goes 1, 2, 3.</think>\n"
        "Some answer here.\n"
        "<think>Actually wait — the image is blank.</think>"
    )
    assert detect_blankness_in_think(text) is False
    # The full-text detector still catches it.
    assert detect_blankness(text) is True


# ---------------------------------------------------------------------------
# analyze — composite return
# ---------------------------------------------------------------------------
def test_analyze_returns_dataclass() -> None:
    r = analyze("the image is blank")
    assert isinstance(r, BlanknessResult)
    assert r.blankness is True
    assert r.early_blankness is True
    assert r.blankness_in_think is False
    assert r.matched_patterns == ["image is blank"]


def test_analyze_collects_multiple_patterns_in_canonical_order() -> None:
    # Order in the input is the reverse of canonical order — the result
    # should still be in canonical (BLANK_PATTERNS) order, deduplicated.
    text = "I cannot see the chart; the image is blank; cannot see again."
    r = analyze(text)
    # "cannot see" comes before "image is blank" in BLANK_PATTERNS.
    assert r.matched_patterns == ["cannot see", "image is blank"]
    # Each unique pattern appears at most once.
    assert len(r.matched_patterns) == len(set(r.matched_patterns))


def test_analyze_empty_string() -> None:
    r = analyze("")
    assert r.blankness is False
    assert r.early_blankness is False
    assert r.blankness_in_think is False
    assert r.matched_patterns == []


def test_analyze_think_and_early_signals_independent() -> None:
    # Phrase only inside think -> in_think True, early True (think block
    # opens at char 0), blankness True.
    text = "<think>the image is blank</think>\nanswer."
    r = analyze(text)
    assert r.blankness is True
    assert r.early_blankness is True
    assert r.blankness_in_think is True


def test_analyze_late_phrase_only() -> None:
    filler = "x" * 600
    text = filler + " the image is blank."
    r = analyze(text)
    assert r.blankness is True
    assert r.early_blankness is False
    assert r.blankness_in_think is False
    assert r.matched_patterns == ["image is blank"]


def test_analyze_custom_early_window_kw() -> None:
    filler = "x" * 600
    text = filler + " the image is blank."
    r = analyze(text, early_blankness_max_chars=1000)
    assert r.early_blankness is True
