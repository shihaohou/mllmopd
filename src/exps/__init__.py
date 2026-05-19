"""Minimal shim package for Uni-OPD's stale `exps.OPD.utils.reward.*` imports.

Uni-OPD's `Uni_OPD_utils/OPD_reward/{reward_manager,get_reward,post_process_rewards}.py`
contain hard-coded `from exps.OPD.utils.reward.X import Y` imports that
reference an older Tencent-internal layout. The actual modules live at
`Uni_OPD_utils.OPD_reward.X` in the current submodule checkout. We can't
modify the submodule files (treating Uni-OPD as read-only upstream), so
this shim provides the legacy import paths and re-exports each name
from its real Uni-OPD location.

We only ship the SUBSET of leaves that our T1 code transitively touches:
  - exps.OPD.utils.reward.teacher          (needed by reward_manager)
  - exps.OPD.utils.reward.utils            (needed by reward_manager)

The other Uni-OPD modules (`rule_base_reward`, `get_reward`,
`post_process_rewards`) are NOT shimmed because:
  - `rule_base_reward.py` itself depends on `exps.RL.utils.reward.*` AND
    `Math.generate.verify_deepmath.reward_func`, neither of which is in
    the submodule checkout. Skipping rule-based correctness in T1 is
    acceptable (it's only used as a metadata tag in diagnostics).
  - `get_reward.py` and `post_process_rewards.py` chain through
    `rule_base_reward.py` and would inherit the same failure. Instead,
    our `mllmopd.training.dual_teacher_get_reward` reimplements the
    needed pieces directly.
"""
