"""Shim: forwards `exps.OPD.utils.reward.teacher` to the real module
at `Uni_OPD_utils.OPD_reward.teacher`. See `exps/__init__.py`."""

from Uni_OPD_utils.OPD_reward.teacher import *  # noqa: F401, F403
from Uni_OPD_utils.OPD_reward.teacher import TeacherState  # noqa: F401  (explicit re-export)
