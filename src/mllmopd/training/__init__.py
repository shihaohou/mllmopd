"""T1 (and beyond) OPD training-side extension modules.

These wrap Uni-OPD's `--custom-rm-path` and `--custom-reward-post-process-path`
extension hooks. No upstream Uni-OPD code is modified — we plug in via the
existing CLI flags.

Layout:
  blank_teacher_payload.py   — helper: produce a blank-image variant of a Sample
  dual_teacher_get_reward.py — entry point for `--custom-rm-path`; chooses
                               between full / blank / both teacher scoring per
                               the `OPD_TEACHER_IMAGE_MODE` env var
  opd_diagnostics_hook.py    — entry point for `--custom-reward-post-process-path`;
                               dumps per-token VD / OPD-reward arrays to JSONL
                               for later analysis
"""
