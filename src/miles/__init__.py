"""Wrapper for the Uni-OPD `miles` package — makes BOTH `miles.X` and
`miles.miles.X` import forms resolve.

Why this exists:
  - The Uni-OPD submodule has `third_party/Uni-OPD/miles/miles/` as the
    real Python package (with `__init__.py`), and `third_party/Uni-OPD/miles/`
    as the outer container directory (sibling of `Uni_OPD_utils/`, no
    `__init__.py`).
  - Most of Uni-OPD's code imports as `from miles.X import ...` which
    works if PYTHONPATH points at the outer dir.
  - But `Uni_OPD_utils/margin_calibration/margin_shift.py` (loaded by
    `miles.backends.training_utils.loss`) uses `from miles.miles.X import ...`
    (double prefix), which only works if PYTHONPATH points at
    `third_party/Uni-OPD/`. The two layouts contradict each other.

Fix: make `src/miles/` (this directory) the first `miles` hit on
PYTHONPATH (`src/` is at the front of the launcher's PYTHONPATH), and
extend `__path__` to include both the inner `miles/miles/` (so
`miles.backends.X` resolves to the real package contents) and the
outer `miles/` (so `miles.miles` resolves as a sub-package whose own
`__path__` is the inner directory). Result: both single- and
double-prefix import forms route to the same real code.

Caveats: this wrapper assumes the repo layout
  <repo>/src/miles/__init__.py        (this file)
  <repo>/third_party/Uni-OPD/miles/   (outer; has Uni_OPD_utils/ as sibling-style submodule)
  <repo>/third_party/Uni-OPD/miles/miles/  (the real Python package)
If the submodule is checked out at a different path (e.g. via
`MILES_DIR` env override), this file does NOT respect it. Override via
`MLLMOPD_MILES_OUTER` env var if needed.
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Repo root is two levels up from this file: src/miles/__init__.py.
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))

# Outer Uni-OPD miles dir (contains Uni_OPD_utils/, scripts/, the inner
# miles/ package, etc.). Override via MLLMOPD_MILES_OUTER if the
# submodule is checked out elsewhere.
_MILES_OUTER = os.environ.get(
    "MLLMOPD_MILES_OUTER",
    os.path.normpath(os.path.join(_REPO_ROOT, "third_party", "Uni-OPD", "miles")),
)
_MILES_INNER = os.path.join(_MILES_OUTER, "miles")

if not os.path.isdir(_MILES_INNER):
    # Don't crash on import — let the eventual ImportError surface in the
    # right place with the right module-level traceback. But do print a
    # clear hint so debugging the path issue isn't grep-archeology.
    sys.stderr.write(
        f"[mllmopd src/miles wrapper] WARNING: inner miles package not found "
        f"at {_MILES_INNER!r}. The wrapper will not extend __path__; "
        f"most `import miles.*` statements will fail. Set "
        f"MLLMOPD_MILES_OUTER to the dir containing the real miles/ subdir.\n"
    )
else:
    # Order matters: inner first so `miles.backends`, `miles.utils`, etc.
    # resolve to the real package's contents on the FIRST PYTHONPATH lookup.
    # Outer second so `miles.miles` (the rare double-prefix form used only
    # in margin_shift.py) resolves to `outer/miles/` which IS the inner
    # package, giving the double-prefix import the same content as the
    # single-prefix one.
    __path__ = [_MILES_INNER, _MILES_OUTER]
