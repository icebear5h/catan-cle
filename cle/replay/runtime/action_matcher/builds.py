"""Match Colonist build, roll, discard and end-of-turn rows."""

from __future__ import annotations

from cle.game_engine.models.enums import Action, ActionType
from cle.replay.colonist.constants import ENGINE_RESOURCES

from .context import UNDECIDED, Decision, MatchContext, decide

__all__ = ["BUILD_MATCHERS"]


def _roll(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "ROLL" not in action_str:
        return UNDECIDED
    dice = ctx.action_hint.get("dice")
    if dice:
        return decide(Action(action.color, action.action_type, tuple(dice)))
    return decide(action)


def _at_target_node(action: Action, ctx: MatchContext) -> Decision:
    if ctx.target_node is not None:
        if hasattr(action, "value") and action.value == ctx.target_node:
            return decide(action)
        return UNDECIDED
    return decide(action)


def _build_settlement(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "BUILD_SETTLEMENT" not in action_str:
        return UNDECIDED
    return _at_target_node(action, ctx)


def _build_city(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "BUILD_CITY" not in action_str:
        return UNDECIDED
    return _at_target_node(action, ctx)


def _build_road(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "BUILD_ROAD" not in action_str:
        return UNDECIDED
    if ctx.target_edge is not None:
        if hasattr(action, "value"):
            action_edge = action.value
            normalized_action_edge = (
                (min(action_edge), max(action_edge)) if action_edge else None
            )
            if normalized_action_edge == ctx.target_edge:
                return decide(action)
        return UNDECIDED
    return decide(action)


def _discard(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "DISCARD" not in action_str:
        return UNDECIDED
    cards = ctx.action_hint.get("cards")
    if cards:
        discarded: list[str] = []
        for i, count in enumerate(cards):
            discarded.extend([ENGINE_RESOURCES[i]] * count)
        return decide(Action(action.color, ActionType.DISCARD, discarded))
    return decide(action)


def _end_turn(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "END_TURN" not in action_str:
        return UNDECIDED
    return decide(action)


BUILD_MATCHERS = {
    "ROLL": _roll,
    "BUILD_SETTLEMENT": _build_settlement,
    "BUILD_CITY": _build_city,
    "BUILD_ROAD": _build_road,
    "DISCARD": _discard,
    "END_TURN": _end_turn,
}
