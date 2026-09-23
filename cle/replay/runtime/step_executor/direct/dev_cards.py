"""Direct execution for Colonist development-card rows."""

from __future__ import annotations

from typing import Final

from cle.game_engine.models.enums import Action, ActionType
from cle.replay.contracts import ReplayPayload
from cle.replay.runtime.audit import record_replay_issue

from ..context import engine_of, seat_index, seating_map
from .outcome import DirectContext

__all__ = [
    "announce_dev_card",
    "buy_development_card",
    "monopoly_resource",
    "play_dev_card",
    "year_of_plenty_resources",
]

_EXPECTED_FOLLOWUP: Final[dict[str, str]] = {
    "PLAY_MONOPOLY": "MONOPOLY_RESOURCE",
    "PLAY_YEAR_OF_PLENTY": "YEAR_OF_PLENTY_RESOURCES",
}
_PLAY_ACTIONS: Final[dict[str, ActionType]] = {
    "PLAY_KNIGHT_CARD": ActionType.PLAY_KNIGHT_CARD,
    "PLAY_ROAD_BUILDING": ActionType.PLAY_ROAD_BUILDING,
}


def announce_dev_card(ctx: DirectContext) -> ReplayPayload:
    """Colonist logs the announcement separately from the resource choice."""
    action_type = ctx.action_type
    expected_followup = _EXPECTED_FOLLOWUP[action_type]
    ctx.state.replay_pending_dev_card = {
        "announcement_type": action_type,
        "expected_followup": expected_followup,
        "player": ctx.action_hint.get("player"),
        "step": ctx.state.replay_index,
    }
    print(f"[Replay] Deferred {action_type} announcement until {expected_followup}")
    return ctx.finish(
        "deferred",
        0,
        f"Deferred {action_type} announcement until {expected_followup}",
        {"deferred": True},
    )


def play_dev_card(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    action_type = ctx.action_type
    game = engine_of(state)
    colonist_player = ctx.action_hint.get("player")
    player_idx = seat_index(seating_map(state), colonist_player)
    action_type_enum = _PLAY_ACTIONS[action_type]

    if player_idx is not None:
        player_color = game.state.colors[player_idx]
        dev_action = Action(player_color, action_type_enum, None)
        print(f"[Replay] Executing {action_type} directly: player={player_color}")
        try:
            game.step(dev_action, force=True)
            return ctx.finish("ok", 1)
        except ValueError as e:
            print(f"[Replay] Skipping {action_type}: direct execution failed: {e}")
            return ctx.finish("skipped", 0, f"Skipped {action_type} (direct execution failed)")

    print(f"[Replay] Skipping {action_type}: couldn't map player {colonist_player}")
    return ctx.finish("skipped", 0, f"Skipped {action_type} (player mapping failed)")


def _clear_pending_announcement(ctx: DirectContext, expected: str, message: str) -> None:
    pending_dev = getattr(ctx.state, "replay_pending_dev_card", None)
    if pending_dev:
        if (
            pending_dev.get("expected_followup") != expected
            or pending_dev.get("player") != ctx.action_hint.get("player")
        ):
            record_replay_issue(
                ctx.state,
                kind="dev_card_announcement_mismatch",
                action_hint=ctx.action_hint,
                message=message,
                severity="error",
                details={"pending": pending_dev},
            )
        ctx.state.replay_pending_dev_card = None


def monopoly_resource(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    action_hint = ctx.action_hint
    game = engine_of(state)
    resource = action_hint.get("resource")
    colonist_player = action_hint.get("player")
    seating = seating_map(state)
    player_idx = seat_index(seating, colonist_player)
    _clear_pending_announcement(
        ctx,
        "MONOPOLY_RESOURCE",
        "MONOPOLY_RESOURCE did not match pending dev-card announcement",
    )

    print(f"[DEBUG MONOPOLY] colonist_player={colonist_player} (type: {type(colonist_player)})")
    print(f"[DEBUG MONOPOLY] colonist_to_engine mapping: {seating}")
    print(f"[DEBUG MONOPOLY] Lookup result: player_idx={player_idx}")
    print(f"[DEBUG MONOPOLY] Engine colors: {game.state.colors}")

    if player_idx is not None and resource is not None:
        player_color = game.state.colors[player_idx]
        monopoly_action = Action(player_color, ActionType.PLAY_MONOPOLY, resource)
        amount = action_hint.get("amount", "?")
        print(f"[Replay] Executing PLAY_MONOPOLY: player_color={player_color}, player_idx={player_idx}, resource={resource}, amount={amount}")
        game.step(monopoly_action, force=True)
        return ctx.finish("ok", 1)
    else:
        print(f"[Replay] Skipping MONOPOLY_RESOURCE: player={colonist_player}, resource={resource}")
        return ctx.finish("skipped", 0, "Skipped MONOPOLY_RESOURCE (invalid params)")


def year_of_plenty_resources(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    action_hint = ctx.action_hint
    game = engine_of(state)
    resources = action_hint.get("resources", [])
    colonist_player = action_hint.get("player")
    player_idx = seat_index(seating_map(state), colonist_player)
    _clear_pending_announcement(
        ctx,
        "YEAR_OF_PLENTY_RESOURCES",
        "YEAR_OF_PLENTY_RESOURCES did not match pending dev-card announcement",
    )

    if player_idx is not None and len(resources) >= 1:
        player_color = game.state.colors[player_idx]
        resource_tuple = tuple(resources[:2]) if len(resources) >= 2 else (resources[0],)
        yop_action = Action(player_color, ActionType.PLAY_YEAR_OF_PLENTY, resource_tuple)
        print(f"[Replay] Executing PLAY_YEAR_OF_PLENTY: player={player_color}, resources={resource_tuple}")
        game.step(yop_action, force=True)
        return ctx.finish("ok", 1)
    else:
        print(f"[Replay] Skipping YEAR_OF_PLENTY_RESOURCES: player={colonist_player}, resources={resources}")
        return ctx.finish("skipped", 0, "Skipped YEAR_OF_PLENTY_RESOURCES (invalid params)")


def buy_development_card(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    game = engine_of(state)
    colonist_player = ctx.action_hint.get("player")
    player_idx = seat_index(seating_map(state), colonist_player)

    if player_idx is not None:
        player_color = game.state.colors[player_idx]
        card_type = ctx.action_hint.get("card_type", None)
        buy_dev_action = Action(player_color, ActionType.BUY_DEVELOPMENT_CARD, card_type)
        print(f"[Replay] Executing BUY_DEVELOPMENT_CARD: player={player_color}, card_type={card_type}")
        game.step(buy_dev_action, force=True)
        return ctx.finish("ok", 1)
    else:
        print(f"[Replay] Skipping BUY_DEVELOPMENT_CARD: couldn't map player {colonist_player}")
        return ctx.finish("skipped", 0, "Skipped BUY_DEVELOPMENT_CARD (player mapping failed)")
