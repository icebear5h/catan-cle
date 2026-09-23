"""Direct execution for recorded closures and bank trades."""

from __future__ import annotations

import time

from cle.game_engine.models.enums import Action, ActionType
from cle.replay.colonist.helpers import format_resources, get_player_color_name
from cle.replay.contracts import ReplayPayload
from cle.replay.runtime.audit import record_replay_issue

from ..context import engine_of, seat_index, seating_map
from ..forcing import apply_trade_closures
from .outcome import DirectContext

__all__ = ["close_trade", "maritime_trade"]


def close_trade(ctx: DirectContext) -> ReplayPayload:
    """CLOSE_TRADE is a replay lifecycle event, not a resource transaction."""
    state = ctx.state
    action_hint = ctx.action_hint
    applied_closures = apply_trade_closures(action_hint, state)
    if not applied_closures:
        return ctx.finish(
            "skipped",
            0,
            f"Could not apply trade closure {action_hint.get('trade_id')}",
        )

    record_replay_issue(
        state,
        kind="replayed_trade_closure",
        action_hint=action_hint,
        message=f"Replayed closure for trade {action_hint.get('trade_id')}",
        severity="info",
        details={"closures": applied_closures},
    )
    return ctx.finish(
        "closed",
        0,
        f"Closed trade {action_hint.get('trade_id')}",
    )


def maritime_trade(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    action_hint = ctx.action_hint
    game = engine_of(state)
    given = action_hint.get("given", (0, 0, 0, 0, 0))
    received = action_hint.get("received", (0, 0, 0, 0, 0))
    colonist_player = action_hint.get("player")
    player_idx = seat_index(seating_map(state), colonist_player)

    if player_idx is not None:
        player_color = game.state.colors[player_idx]
        given_resource_count = sum(1 for x in given if x > 0)
        received_resource_count = sum(1 for x in received if x > 0)
        is_multi_resource = given_resource_count > 1 or received_resource_count > 1

        maritime_action = Action(player_color, ActionType.MARITIME_TRADE, (given, received))

        if is_multi_resource:
            print(f"[Replay] Executing multi-resource MARITIME_TRADE: {format_resources(given)} -> {format_resources(received)}")
        else:
            print(f"[Replay] Executing MARITIME_TRADE: {format_resources(given)} -> {format_resources(received)}")

        applied_closures = apply_trade_closures(action_hint, state)
        game.step(maritime_action, force=True)
        if applied_closures:
            record_replay_issue(
                state,
                kind="replayed_trade_closure",
                action_hint=action_hint,
                message="Replayed offer closures attached to maritime trade",
                severity="info",
                details={"closures": applied_closures},
            )

        state.game_log.append({
            "type": "general",
            "timestamp": time.time(),
            "message": f"[{state.replay_index+1}/{len(ctx.parsed_actions)}] MARITIME_TRADE: {format_resources(given)} -> {format_resources(received)}",
            "color": get_player_color_name(colonist_player),
        })
        return ctx.finish("ok", 1)
    else:
        print(f"[Replay] Skipping MARITIME_TRADE: couldn't map player {colonist_player}")
        return ctx.finish("skipped", 0, "Skipped MARITIME_TRADE (player mapping failed)")
