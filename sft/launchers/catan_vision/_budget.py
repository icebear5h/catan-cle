"""The spending plan for a budgeted launch and its bounded wait."""

from __future__ import annotations

import math
import time

from modal import FunctionCall

from sft.json_types import JsonLikeDict
from sft.launchers.catan_vision._config import (
    BUDGET_EVAL_RESERVE_USD,
    BUDGET_RATE_PER_SECOND,
    BUDGET_STARTUP_SECONDS,
    BUDGET_TIMEOUT_SECONDS,
)


def training_budget_plan(budget_usd: float) -> JsonLikeDict | None:
    """Reserve eval money and validate the bounded training envelope before upload.

    This is a one-launch resource/time bound, not an account billing limit.
    A separate CPU watchdog cancels even rescheduled/crash-looping GPU calls at
    an absolute deadline. No evaluation or second training run is auto-launched.
    """
    if not math.isfinite(budget_usd) or budget_usd < 0:
        raise ValueError("budget_usd must be finite and nonnegative")
    if budget_usd == 0:
        return None
    maximum = BUDGET_RATE_PER_SECOND * (BUDGET_TIMEOUT_SECONDS + BUDGET_STARTUP_SECONDS)
    remainder = budget_usd - maximum - BUDGET_EVAL_RESERVE_USD
    if remainder < 5.0:
        raise ValueError("budget is too small for the bounded training profile, $25 eval reserve and $5+ safety margin")
    return {
        "schema": "catan_single_run_budget/v1",
        "budget_usd": budget_usd,
        "maximum_training_window_usd_at_list_rates": maximum,
        "standalone_eval_reserve_usd": BUDGET_EVAL_RESERVE_USD,
        "remaining_headroom_usd": remainder,
        "gpu": "H200", "cpu_core_limit": 16, "memory_gib_limit": 128,
        "execution_timeout_seconds": BUDGET_TIMEOUT_SECONDS,
        "startup_timeout_seconds": BUDGET_STARTUP_SECONDS,
        "rate_per_second": BUDGET_RATE_PER_SECOND,
        "pricing_source": "https://modal.com/pricing",
        "pricing_checked": "2026-09-05",
        "function_retries": 0,
        "absolute_deadline_watchdog": True,
        "evals_auto_launched": False,
        "excludes": ["prior spending", "shared storage", "workspace subscription"],
    }


def wait_for_budgeted_call(call: FunctionCall[JsonLikeDict], deadline_unix: float) -> JsonLikeDict:
    """Observe one known call, cancelling its containers on deadline or failure."""
    if not math.isfinite(deadline_unix):
        raise ValueError("budget deadline must be finite")
    try:
        while (remaining := deadline_unix - time.time()) > 0:
            try:
                result = call.get(timeout=min(30.0, remaining))
                return {"status": "finished", "result": result}
            except TimeoutError:
                continue
        call.cancel(terminate_containers=True)
        return {"status": "budget_deadline_cancelled"}
    except BaseException:
        # If the watcher itself is interrupted, fail closed. A later restart of
        # this watcher may observe cancellation but cannot start more training.
        call.cancel(terminate_containers=True)
        raise
