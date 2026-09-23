"""Explicit native-reasoning request and evidence contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from cle.game_engine.models.enums import ActionType
from cle.players.contracts import PlayerContext
from cle.players.data import JsonValue

# One normalized native-reasoning request: only scalars cross the provider wire.
NativeReasoningRequest: TypeAlias = dict[str, bool | int | str]

DEFAULT_NATIVE_REASONING_EFFORT = "xhigh"
NATIVE_REASONING_EFFORTS = frozenset(
    {"minimal", "low", "medium", "high", "xhigh", "max"}
)

# Game start never reasons below high; one batched setup call commits both
# the settlement and its road, so every setup key holds the top budget.
_SETUP_PROMPT_KEYS = frozenset(
    {
        "initial_placement",
        "initial_road",
        "initial_settlement_1",
        "initial_settlement_2",
        "initial_road_1",
        "initial_road_2",
    }
)
_OPENING_TURNS = 8
_ENDGAME_VP = 7

# Menus containing any of these hold high/4096: trading decisions compare
# several offers at once and demonstrably ramble past medium budgets.
_TRADE_ACTION_TYPES = frozenset(
    {
        ActionType.OFFER_TRADE,
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
        ActionType.COUNTER_OFFER,
        ActionType.CONFIRM_TRADE,
        ActionType.CANCEL_TRADE,
    }
)

# Menus containing any of these hold medium/2048; anything smaller is routine.
_SIGNIFICANT_ACTION_TYPES = frozenset(
    {
        ActionType.BUILD_ROAD,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_CITY,
        ActionType.BUY_DEVELOPMENT_CARD,
        ActionType.PLAY_KNIGHT_CARD,
        ActionType.PLAY_YEAR_OF_PLENTY,
        ActionType.PLAY_MONOPOLY,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.MARITIME_TRADE,
        ActionType.MOVE_ROBBER,
        ActionType.STEAL,
        ActionType.DISCARD,
    }
)


@dataclass(frozen=True, slots=True)
class ReasoningBudget:
    """Per-decision reasoning effort plus total-completion token ceiling.

    max_tokens None means uncapped: the effort tier and the pacing line do
    the work, and retries escalate effort instead of a wall.
    """

    effort: str
    max_tokens: int | None


def reasoning_budget_for_context(context: PlayerContext) -> ReasoningBudget:
    """Select one reasoning budget from only visible decision facts.

    Completion is uncapped everywhere: effort tiers pace the model, the
    prompt line states the expected scale, and retries escalate effort.
    """
    if context.prompt_key in _SETUP_PROMPT_KEYS:
        return ReasoningBudget(effort="high", max_tokens=None)
    if context.turn_number < _OPENING_TURNS:
        return ReasoningBudget(effort="high", max_tokens=None)
    vps = (context.observation.my_vp or 0, *context.observation.opponent_vps.values())
    if any(vp >= _ENDGAME_VP for vp in vps):
        return ReasoningBudget(effort="high", max_tokens=None)
    if context.prompt_key in ("discarding", "robber"):
        return ReasoningBudget(effort="medium", max_tokens=None)
    menu = {action.action_type for action in context.legal_actions}
    if menu & _TRADE_ACTION_TYPES:
        return ReasoningBudget(effort="high", max_tokens=None)
    if menu & _SIGNIFICANT_ACTION_TYPES:
        return ReasoningBudget(effort="medium", max_tokens=None)
    return ReasoningBudget(effort="medium", max_tokens=None)


_EFFORT_LADDER = ("minimal", "low", "medium", "high", "xhigh", "max")


def escalate_budget(budget: ReasoningBudget) -> ReasoningBudget:
    """Raise one budget tier for a retry: retries must not re-hit the same wall."""
    try:
        tier = _EFFORT_LADDER.index(budget.effort)
    except ValueError:
        return budget
    effort = _EFFORT_LADDER[min(tier + 1, len(_EFFORT_LADDER) - 1)]
    max_tokens = budget.max_tokens * 2 if budget.max_tokens is not None else None
    return ReasoningBudget(effort=effort, max_tokens=max_tokens)


def reasoning_budget_for_attempt(
    context: PlayerContext, feedback: str | None
) -> ReasoningBudget:
    """Select the budget for one attempt; a retry never re-hits the same wall."""
    budget = reasoning_budget_for_context(context)
    if feedback is not None:
        budget = escalate_budget(budget)
    return budget


def reasoning_budget_instruction(budget: ReasoningBudget) -> str:
    """Render one pacing line so the model can see its reasoning budget."""
    if budget.max_tokens is None:
        return (
            f"Reasoning effort is {budget.effort}; keep deliberation tight, "
            "front-load the decisive considerations, and commit to exactly "
            "one response promptly."
        )
    return (
        f"Reasoning budget is about {budget.max_tokens} completion tokens "
        f"at {budget.effort} effort. Pace yourself against that budget and "
        "commit to exactly one response before it runs out."
    )


def native_reasoning_request(
    effort: str = DEFAULT_NATIVE_REASONING_EFFORT,
) -> NativeReasoningRequest:
    """Build one explicit OpenRouter reasoning request."""
    normalized = effort.strip().lower()
    if normalized == "off":
        return {"enabled": False}
    if normalized not in NATIVE_REASONING_EFFORTS:
        choices = ", ".join(["off", *sorted(NATIVE_REASONING_EFFORTS)])
        raise ValueError(f"Unknown native reasoning effort {effort!r}; expected: {choices}")
    return {"effort": normalized, "exclude": False}


def validate_native_reasoning_request(value: object) -> NativeReasoningRequest:
    """Normalize a caller request and never delegate mode selection to a provider."""
    if value is None:
        return native_reasoning_request()
    if not isinstance(value, Mapping):
        raise ValueError("reasoning must be an object")

    unknown = set(value) - {"enabled", "effort", "max_tokens", "exclude"}
    if unknown:
        raise ValueError(f"Unknown reasoning fields: {sorted(unknown)}")

    enabled = value.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("reasoning.enabled must be a boolean")
    if enabled is False:
        if any(key in value for key in ("effort", "max_tokens")):
            raise ValueError(
                "Disabled reasoning cannot also set effort or max_tokens"
            )
        return {"enabled": False}

    exclude = value.get("exclude", False)
    if not isinstance(exclude, bool):
        raise ValueError("reasoning.exclude must be a boolean")
    if exclude:
        raise ValueError(
            "reasoning.exclude must be false so native reasoning evidence is retained"
        )

    effort = value.get("effort")
    max_tokens = value.get("max_tokens")
    if effort is not None and max_tokens is not None:
        raise ValueError("reasoning.effort and reasoning.max_tokens are mutually exclusive")

    if effort is not None:
        if not isinstance(effort, str):
            raise ValueError("reasoning.effort must be a string")
        return native_reasoning_request(effort)

    if max_tokens is not None:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("reasoning.max_tokens must be a positive integer")
        return {"max_tokens": max_tokens, "exclude": False}

    return native_reasoning_request()


def native_reasoning_enabled(request: Mapping[str, JsonValue]) -> bool:
    """Return whether the normalized request asks the model to reason natively."""
    return request.get("enabled") is not False


def reasoning_token_count(usage: Mapping[str, JsonValue]) -> int | None:
    """Extract provider-reported reasoning tokens when present and valid."""
    details = usage.get("completion_tokens_details")
    if not isinstance(details, Mapping):
        return None
    value = details.get("reasoning_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def native_reasoning_returned(
    native_reasoning: str,
    native_reasoning_details: tuple[JsonValue, ...],
    usage: Mapping[str, JsonValue],
) -> bool:
    """Return whether a response contains any provider evidence of native reasoning."""
    tokens = reasoning_token_count(usage)
    return bool(native_reasoning or native_reasoning_details or (tokens is not None and tokens > 0))
