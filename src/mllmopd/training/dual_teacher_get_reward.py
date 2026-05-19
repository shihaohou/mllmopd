"""T1 custom-rm-path: dual-teacher OPD reward fetcher.

Replaces Uni-OPD's default `Uni_OPD_utils.OPD_reward.get_reward.get_reward`
for the T1 FullTeacher-vs-BlankTeacher negative control. Wired in via
the launcher's `--custom-rm-path mllmopd.training.dual_teacher_get_reward.get_reward`
flag (set by `scripts/train/opd_mmr1_3b_baseline.sh`).

Behavior:
  - For every sample, build TWO teacher payloads — one with the canonical
    full-image content, one with a same-size white-blank substitution.
  - Dispatch both to (likely different) teacher URLs in parallel via
    asyncio.gather. Each is a logprob-scoring call (`max_new_tokens=0`,
    `return_logprob=True`), so total wall time is ~max(full, blank)
    instead of sum.
  - `OPD_TEACHER_IMAGE_MODE` selects which logprobs become the training
    reward:
        full   → `meta_info` = lp_full,  `meta_info_diagnostic` = lp_blank
        blank  → `meta_info` = lp_blank, `meta_info_diagnostic` = lp_full
  - Both arms always log both quantities, so the diagnostics hook can
    compute per-token VD = lp_full - lp_blank regardless of which arm.

Failure handling matches Uni-OPD's default: if either teacher call
exhausts retries we mark the sample with REWARD_FAILED_KEY=True, and
post_process_rewards will fill the row with -100 sentinel and mask
it from the PG loss.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import random
import time
from argparse import Namespace

import aiohttp

# Uni-OPD's own files use stale `from exps.OPD.utils.reward.*` imports
# that don't resolve in the current layout. We can't import
# `Uni_OPD_utils.OPD_reward.{get_reward,rule_base_reward,post_process_rewards}`
# at all because their module-load triggers the broken chain:
#   - get_reward.py top-level imports rule_base_reward
#   - rule_base_reward.py top-level imports `exps.RL.utils.reward.PRIME_code_server.server`
#     and `Math.generate.verify_deepmath.reward_func` — neither present here.
#
# So we import only the two modules whose dependency closure IS satisfied
# by our minimal `src/exps/OPD/utils/reward/{teacher,utils}.py` shim
# (the only `exps.OPD.utils.reward.*` paths reward_manager.py needs):
#   - Uni_OPD_utils.OPD_reward.reward_manager  → uses exps.OPD.utils.reward.{teacher,utils}
#   - Uni_OPD_utils.OPD_reward.session_manager → clean
# REWARD_FAILED_KEY is just a string constant; we inline it. The rule-based
# reward is skipped entirely (response_correct=None is acceptable for T1
# diagnostics — see comment in get_reward() below).
from Uni_OPD_utils.OPD_reward.reward_manager import RMSystemManager
from Uni_OPD_utils.OPD_reward.session_manager import RewardSessionManager

# Verbatim from Uni_OPD_utils/OPD_reward/get_reward.py:22 — kept in sync
# manually because that module can't be imported (see chain above).
REWARD_FAILED_KEY = "__opd_reward_failed__"

from miles.utils.types import Sample

from mllmopd.training.blank_teacher_payload import make_blank_image_data

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_DELAY = 2.0

# Match Uni-OPD's default of clearing inherited proxy envs — sglang
# teacher servers run on the same intranet and proxying would route
# them through an external hop.
os.environ["http_proxy"], os.environ["https_proxy"] = "", ""


def _resolve_image_mode() -> str:
    mode = os.environ.get("OPD_TEACHER_IMAGE_MODE", "full").lower()
    if mode not in ("full", "blank"):
        raise ValueError(
            f"OPD_TEACHER_IMAGE_MODE must be 'full' or 'blank' (T1-2 / T1-3 arm); "
            f"got {mode!r}. The 'both' alias is reserved — both arms always "
            f"dual-call for diagnostics; this knob picks the *primary* one."
        )
    return mode


async def _post_with_retry(session_manager, url: str, payload: dict, tag: str) -> dict:
    """POST one payload, retry on transient errors. Returns parsed JSON
    or raises after _MAX_RETRIES. `tag` is just for log labelling."""
    last_exception = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with session_manager.inflight_sem:
                session = await session_manager.get_session()
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            last_exception = e
            logger.warning(
                f"[dual_teacher/{tag}] attempt {attempt}/{_MAX_RETRIES} on "
                f"{url}: {type(e).__name__}: {e}"
            )
            is_bad_fd = isinstance(e, aiohttp.ClientOSError) and e.errno == 9
            is_session_closed = (
                isinstance(e, RuntimeError) and "Session is closed" in str(e)
            )
            is_fd_reuse = (
                isinstance(e, RuntimeError)
                and "File descriptor" in str(e)
                and "used by transport" in str(e)
            )
            if is_bad_fd or is_session_closed or is_fd_reuse:
                await session_manager.reset_session()
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY * attempt + random.uniform(0, 0.2))
        except Exception as e:
            last_exception = e
            logger.warning(
                f"[dual_teacher/{tag}] unexpected error attempt {attempt}/{_MAX_RETRIES} "
                f"on {url}: {type(e).__name__}: {e}"
            )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY * attempt + random.uniform(0, 0.2))
    raise RuntimeError(
        f"[dual_teacher/{tag}] all {_MAX_RETRIES} attempts failed: "
        f"{type(last_exception).__name__}: {last_exception}"
    )


async def get_reward(args: Namespace, sample: Sample, **kwargs) -> dict:
    """T1 dual-teacher reward entry point.

    Scores `sample` against the full image and a same-size white blank.
    Returns a reward dict in Uni-OPD's expected schema:
      - meta_info.input_token_logprobs  → primary (training-reward) logprobs
      - meta_info_diagnostic.input_token_logprobs → the other one, logged
    Plus the standard `response_correct`, `rule_based_metadata`,
    `reward_time`, `teacher_url` fields the default post-processor reads.
    """
    start_time = time.time()
    image_mode = _resolve_image_mode()

    rm_manager = RMSystemManager(args)
    session_manager = RewardSessionManager()

    # T1 doesn't consume rule-based correctness — `response_correct` is
    # only a diagnostic tag in the per-step JSONL. The default Uni-OPD
    # rule_base_reward.py at module-level loads `Math.generate.verify_deepmath`
    # and `exps.RL.utils.reward.PRIME_code_server.server`, neither of
    # which is in our checkout. Skipping the call avoids the chain.
    response_correct: bool | None = None
    rule_based_metadata: dict = {}

    # Canonical full-image payload (drives `payload["input_ids"]`,
    # `image_data`, etc. via Uni-OPD's regular path).
    payload_full = rm_manager.build_payload(sample)

    # Blank variant: deepcopy + swap image_data with same-size white
    # blanks. Input_ids, sampling_params, etc. are identical so the
    # teacher's prefix tokens are the same and the only intentional
    # difference is the pixel content.
    payload_blank = copy.deepcopy(payload_full)
    blank_image_data = make_blank_image_data(sample)
    if blank_image_data:
        payload_blank["image_data"] = blank_image_data
    elif "image_data" in payload_blank:
        # Defensive: if the sample had no PIL images but the payload
        # has image_data (shouldn't happen for MLLM rollouts), drop it.
        del payload_blank["image_data"]

    # Fire both in parallel against possibly-different teacher URLs.
    # On a single-server config they share the URL; sglang's continuous
    # batching handles concurrency.
    url_full = rm_manager.get_next_url(sample)
    url_blank = rm_manager.get_next_url(sample)

    try:
        res_full, res_blank = await asyncio.gather(
            _post_with_retry(session_manager, url_full, payload_full, "full"),
            _post_with_retry(session_manager, url_blank, payload_blank, "blank"),
        )
    except Exception as e:
        logger.error(f"[dual_teacher] one or both arms failed permanently: {e}")
        return {
            REWARD_FAILED_KEY: True,
            "response_correct": response_correct,
            "rule_based_metadata": rule_based_metadata,
            "reward_time": time.time() - start_time,
            "meta_info": {"input_token_logprobs": []},
            "meta_info_diagnostic": {"input_token_logprobs": []},
            "image_mode": image_mode,
        }

    if image_mode == "full":
        primary, diagnostic = res_full, res_blank
        primary_url, diagnostic_url = url_full, url_blank
    else:  # blank
        primary, diagnostic = res_blank, res_full
        primary_url, diagnostic_url = url_blank, url_full

    # Reuse the primary response's top-level shape so Uni-OPD's default
    # post_process_rewards reads meta_info.input_token_logprobs as the
    # training reward. The diagnostic counterpart is attached as a
    # parallel key for our custom diagnostics hook.
    primary["meta_info_diagnostic"] = diagnostic.get("meta_info", {})
    primary["teacher_url_diagnostic"] = diagnostic_url
    primary["reward_time"] = time.time() - start_time
    primary["teacher_url"] = primary_url
    primary["response_correct"] = response_correct
    primary["rule_based_metadata"] = rule_based_metadata
    primary["image_mode"] = image_mode
    return primary
