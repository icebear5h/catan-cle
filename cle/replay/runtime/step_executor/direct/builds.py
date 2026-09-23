"""Direct execution for recorded builds and end-of-turn rows."""

from __future__ import annotations

import time
from typing import Final

from cle.game_engine.models.enums import Action, ActionType
from cle.replay.contracts import ReplayPayload
from cle.replay.runtime.audit import record_replay_issue
from cle.replay.runtime.checkpoint import ReplayStepCheckpoint

from ..context import (
    engine_color_for_colonist,
    engine_of,
    record_forced_overlay,
    seat_index,
    seating_map,
)
from ..forcing import force_record_road
from ..publishing import publish_replay_action
from .outcome import DirectContext

__all__ = ["BUILD_ACTIONS", "build", "end_turn"]

BUILD_ACTIONS: Final[list[str]] = ["BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"]
_BUILD_TYPES: Final[dict[str, ActionType]] = {
    "BUILD_ROAD": ActionType.BUILD_ROAD,
    "BUILD_SETTLEMENT": ActionType.BUILD_SETTLEMENT,
    "BUILD_CITY": ActionType.BUILD_CITY,
}


def build(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    game = engine_of(state)
    action_hint = ctx.action_hint
    action_type = ctx.action_type
    colonist_corner = action_hint.get("colonist_corner")
    colonist_edge = action_hint.get("colonist_edge")
    target_node = state.corner_to_node_map.get(f"_{colonist_corner}") if colonist_corner is not None else None
    target_edge = state.edge_to_edge_map.get(f"_{colonist_edge}") if colonist_edge is not None else None

    colonist_player = action_hint.get("player")
    player_idx = (
        seat_index(seating_map(state), colonist_player)
        if colonist_player is not None
        else None
    )

    if player_idx is None:
        player_idx = game.state.current_player_index
        print(f"[Replay] {action_type} has no player info, inferring from context: trying player {player_idx}")

    player_color = game.state.colors[player_idx]

    location: object = None
    edge_location: tuple[int, int] | None = None
    if action_type in ["BUILD_SETTLEMENT", "BUILD_CITY"]:
        location = target_node
    elif action_type == "BUILD_ROAD" and target_edge is not None:
        edge_location = (target_edge[0], target_edge[1])
        location = edge_location

    if location is None:
        print(f"[Replay] Skipping {action_type}: no location mapping (colonist corner={colonist_corner}, edge={colonist_edge})")
        state.game_log.append({
            "type": "warning",
            "timestamp": time.time(),
            "message": f"[{state.replay_index+1}/{len(ctx.parsed_actions)}] Skipped {action_type} (no location mapping)",
        })
        return ctx.finish("skipped", 0, f"Skipped {action_type} (no location mapping)")

    build_action = Action(player_color, _BUILD_TYPES[action_type], location)
    print(f"[Replay] Executing {action_type}: player={player_color}, location={location}")

    build_checkpoint = ReplayStepCheckpoint.capture(state)
    try:
        game.step(build_action, force=True)
    except ValueError as e:
        build_checkpoint.restore(state)
        error_msg = str(e)
        if action_type == "BUILD_CITY" and "no player settlement" in error_msg:
            print(f"[Replay] BUILD_CITY failed with player {player_idx}, searching for correct player...")
            found_player = None
            for try_idx in range(len(game.state.colors)):
                try_color = game.state.colors[try_idx]
                try_action = Action(try_color, ActionType.BUILD_CITY, location)
                try:
                    game.step(try_action, force=True)
                    found_player = try_idx
                    print(f"[Replay] BUILD_CITY succeeded with player {try_idx}")
                    break
                except ValueError:
                    build_checkpoint.restore(state)
                    continue
            if found_player is None:
                raise ValueError(f"Could not find valid player for BUILD_CITY at {location}")
        else:
            if action_type == "BUILD_ROAD" and edge_location is not None:
                return _force_road(ctx, build_action, edge_location, error_msg)

            print(f"[Replay] Skipping {action_type}: direct execution failed: {error_msg}")
            state.game_log.append({
                "type": "warning",
                "timestamp": time.time(),
                "message": f"[{state.replay_index+1}/{len(ctx.parsed_actions)}] Skipped {action_type} (direct execution failed: {error_msg})",
            })
            return ctx.finish("skipped", 0, f"Skipped {action_type} (direct execution failed)")

    return ctx.finish("ok", 1)


def _force_road(
    ctx: DirectContext,
    build_action: Action,
    location: tuple[int, int],
    error_msg: str,
) -> ReplayPayload:
    state = ctx.state
    game = engine_of(state)
    is_free = bool(
        game.state.is_initial_build_phase
        or (
            game.state.is_road_building
            and game.state.free_roads_available > 0
        )
    )
    print(f"[Replay] Forcing BUILD_ROAD record after placement failure: {error_msg}")
    force_record_road(game.state, build_action.color, location, is_free)
    publish_replay_action(game, build_action)
    record_forced_overlay(
        state,
        ctx.action_hint,
        "Forced BUILD_ROAD board record after engine placement failure",
        details={"location": location, "error": error_msg},
    )
    state.game_log.append({
        "type": "warning",
        "timestamp": time.time(),
        "message": f"[{state.replay_index+1}/{len(ctx.parsed_actions)}] Forced {ctx.action_type} from Colonist replay (direct execution failed: {error_msg})",
    })
    return ctx.finish("ok", 1, f"Forced {ctx.action_type} from Colonist replay")


def end_turn(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    game = engine_of(state)
    action_hint = ctx.action_hint
    colonist_player = action_hint.get("player")
    _, player_idx = engine_color_for_colonist(state, colonist_player)
    if player_idx is not None and game.state.current_turn_index != player_idx:
        print("[Replay] END_TURN already satisfied by engine turn advance")
        return ctx.finish(
            "already_satisfied",
            0,
            "END_TURN already satisfied by engine turn advance",
        )

    issue_message = "END_TURN unavailable while engine requires another action"
    print(f"[Replay] {issue_message}")
    record_replay_issue(
        state,
        kind="unmatched_end_turn",
        action_hint=action_hint,
        message=issue_message,
        severity="error",
        details={
            "current_prompt": str(game.state.current_prompt),
            "current_turn_index": game.state.current_turn_index,
            "colonist_player": colonist_player,
        },
    )
    state.game_log.append({
        "type": "general",
        "timestamp": time.time(),
        "message": f"[{state.replay_index+1}/{len(ctx.parsed_actions)}] {issue_message}",
    })
    return ctx.finish("unmatched", 0, issue_message)
