"""Resolve a replay row that no playable engine action matched."""

from __future__ import annotations

import time
from collections.abc import Sequence

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action
from cle.replay.colonist.helpers import get_player_color_name
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import (
    ParsedActions,
    ReplayOutcome,
    ReplayPayload,
    ReplayRuntimeState,
)
from cle.replay.runtime.audit import record_replay_issue

from ..context import mark_finished_if_needed, record_forced_overlay
from ..direct import handle_direct_execute
from ..forcing import force_apply_trade_overlay
from ..publishing import publish_trade_overlay
from ..trade_steps import handle_confirm_trade
from .prepare import REQUIRED_ACTION_TYPES, SKIP_ACTIONS
from .skip_reason import skip_message, skip_reason_for

__all__ = ["handle_unmatched"]


def handle_unmatched(
    state: ReplayRuntimeState,
    game: GameEngine,
    action_hint: ActionHint,
    action_type: str | None,
    parsed_actions: ParsedActions,
    playable: Sequence[Action],
    engine_requires: str | None,
    lookahead_executed_type: str | None,
) -> ReplayOutcome:
    if action_type in (
        "DISCARD", "MOVE_ROBBER", "STEAL", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING",
    ):
        result, handled = handle_direct_execute(action_type, action_hint, state)
        if handled and result is not None:
            return result

    # Required actions can be executed by lookahead to satisfy engine prompts.
    # When the replay cursor later reaches that already-satisfied hint, do not
    # fall back to the first playable action; that mutates resources incorrectly.
    if action_type in REQUIRED_ACTION_TYPES:
        return _finish_required(
            state, action_hint, action_type, parsed_actions,
            engine_requires, lookahead_executed_type,
        )

    # Handle skippable trade actions
    if action_type in SKIP_ACTIONS:
        return _finish_trade_overlay(
            state, game, action_hint, action_type, parsed_actions, playable
        )

    # Handle CONFIRM_TRADE
    if action_type == "CONFIRM_TRADE":
        return handle_confirm_trade(action_hint, state)

    # Handle direct-execute types
    result, handled = handle_direct_execute(action_type, action_hint, state)
    if handled and result is not None:
        return result

    return _finish_unresolved(state, action_hint, action_type, parsed_actions, playable)


def _finish_required(
    state: ReplayRuntimeState,
    action_hint: ActionHint,
    action_type: str | None,
    parsed_actions: ParsedActions,
    engine_requires: str | None,
    lookahead_executed_type: str | None,
) -> ReplayPayload:
    skip_reason = (
        "already satisfied by lookahead"
        if lookahead_executed_type == action_type
        else "no matching required action"
    )
    status = "already_satisfied" if lookahead_executed_type == action_type else "unmatched"
    if status == "unmatched":
        record_replay_issue(
            state,
            kind="unmatched_required_action",
            action_hint=action_hint,
            message=f"No matching required action for {action_type}",
            severity="error",
            details={"engine_requires": engine_requires},
        )
    print(f"[Replay] {action_type}: {skip_reason}")
    state.game_log.append({
        "type": "general",
        "timestamp": time.time(),
        "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {action_type} ({skip_reason})",
        "color": get_player_color_name(action_hint.get("player")),
    })
    state.replay_actions_per_step.append(0)
    state.replay_index += 1
    finished = mark_finished_if_needed(state, parsed_actions)
    return {
        "status": status,
        "event_index": state.replay_index,
        "total_events": len(parsed_actions),
        "message": f"{action_type} ({skip_reason})",
        "finished": finished,
    }


def _finish_trade_overlay(
    state: ReplayRuntimeState,
    game: GameEngine,
    action_hint: ActionHint,
    action_type: str,
    parsed_actions: ParsedActions,
    playable: Sequence[Action],
) -> ReplayPayload:
    trade_num = action_hint.get("trade_num", 0)
    trade_label = f"Trade #{trade_num}" if trade_num else "Trade"
    skip_msg = skip_message(action_hint, action_type, trade_label)
    skip_reason = skip_reason_for(state, game, action_hint, action_type, playable)

    overlay_msg = force_apply_trade_overlay(action_type, action_hint, state)
    if overlay_msg:
        publish_trade_overlay(state, action_hint)
        print(f"[Replay] {overlay_msg}: {skip_msg}{skip_reason}")
        record_forced_overlay(
            state,
            action_hint,
            overlay_msg,
            details={"reason": skip_reason.strip(" -")},
        )
        state.game_log.append({
            "type": "trade",
            "timestamp": time.time(),
            "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {overlay_msg}: {skip_msg}",
            "color": get_player_color_name(action_hint.get("player")),
        })
        status = "overlay_applied"
        message = overlay_msg
    else:
        message = f"Unmatched {action_type}{skip_reason}"
        print(f"[Replay] {message}: {skip_msg}")
        record_replay_issue(
            state,
            kind="unmatched_trade_overlay",
            action_hint=action_hint,
            message=message,
            severity="error",
            details={"reason": skip_reason.strip(" -")},
        )
        state.game_log.append({
            "type": "warning",
            "timestamp": time.time(),
            "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {message}: {skip_msg}",
            "color": get_player_color_name(action_hint.get("player")),
        })
        status = "unmatched"

    state.replay_actions_per_step.append(0)
    state.replay_index += 1
    finished = mark_finished_if_needed(state, parsed_actions)
    return {
        "status": status,
        "event_index": state.replay_index,
        "total_events": len(parsed_actions),
        "message": message,
        "finished": finished,
    }


def _finish_unresolved(
    state: ReplayRuntimeState,
    action_hint: ActionHint,
    action_type: str | None,
    parsed_actions: ParsedActions,
    playable: Sequence[Action],
) -> ReplayPayload:
    message = f"No engine action or replay force path matched {action_type}"
    print(f"[Replay] {message}")
    record_replay_issue(
        state,
        kind="unmatched_replay_action",
        action_hint=action_hint,
        message=message,
        severity="error",
        details={
            "playable_types": [
                str(a.action_type).replace("ActionType.", "")
                for a in playable
                if hasattr(a, "action_type")
            ][:20],
        },
    )
    state.game_log.append({
        "type": "warning",
        "timestamp": time.time(),
        "message": f"[{state.replay_index+1}/{len(parsed_actions)}] {message}",
    })
    state.replay_actions_per_step.append(0)
    state.replay_index += 1
    finished = mark_finished_if_needed(state, parsed_actions)
    return {
        "status": "unmatched",
        "event_index": state.replay_index,
        "total_events": len(parsed_actions),
        "message": message,
        "finished": finished,
    }
