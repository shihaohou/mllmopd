"""See exps/__init__.py for the shim rationale.

Only `teacher` and `utils` are provided as leaves — these are the two
modules that `Uni_OPD_utils.OPD_reward.reward_manager` imports at
module load time. Other modules (`get_reward`, `post_process_rewards`,
`rule_base_reward`, `session_manager`) are intentionally absent.
"""
