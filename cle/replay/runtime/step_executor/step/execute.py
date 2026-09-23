"""Apply the matched engine action and report the resulting replay step."""

from __future__ import annotations

import time
import traceback

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action
from cle.game_engine.state import apply_counter_offer, apply_offer_trade
from cle.replay.colonist.constants import RESOURCE_EMOJIS
from cle.replay.colonist.helpers import (
    colonist_cards_to_freqdeck,
    get_engine_player_resources,
    validate_resources_match,
)
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import (
    ParsedActions,
    ReplayArchive,
    ReplayOutcome,
    ReplayRuntimeState,
    list_field,
    mapping_field,
)

from ..context import (
    mark_finished_if_needed,
    regenerate_playable_actions,
    seat_index,
)
from ..forcing import apply_trade_closures
from ..offers import project_trade_record
from ..publishing import publish_trade_overlay
from .logging import log_matched_action, log_resource_gains

__all__ = ["execute_matched_action"]

_RESOURCE_KEYS = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]


def execute_matched_action(
    state: ReplayRuntimeState,
    game: GameEngine,
    replay_data: ReplayArchive,
    action_hint: ActionHint,
    action: Action,
    parsed_actions: ParsedActions,
    auto_actions_taken: int,
) -> ReplayOutcome:
    action_type = log_matched_action(state, action, action_hint, parsed_actions)

    # Snapshot resources before execution for diff
    before: dict[int, dict[str, int]] = {}
    for idx in range(len(game.state.colors)):
        before[idx] = {
            r: int(game.state.player_state.get(f"P{idx}_{r}_IN_HAND", 0))
            for r in _RESOURCE_KEYS
        }

    # Execute the action
    try:
        print(f"[Replay] Executing action: {action}")
        apply_trade_closures(action_hint, state)
        if action_type in ("OFFER_TRADE", "COUNTER_OFFER"):
            # Full source snapshots include responses, unlike a live offer intent.
            # Materialize and project them before publishing the single offer fact.
            apply_offer = apply_offer_trade if action_type == "OFFER_TRADE" else apply_counter_offer
            apply_offer(game.state, action, force=True)
            record = state.replay_trade_ledger.get(action_hint.get("trade_id"))
            if record is not None:
                project_trade_record(state, record)
            regenerate_playable_actions(game.state)
            publish_trade_overlay(state, action_hint)
        else:
            game.step(action, force=True)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Action failed: {e}", "action": str(action), "hint": action_hint}, 500

    log_resource_gains(state, game, before, RESOURCE_EMOJIS, _RESOURCE_KEYS)
    _validate_expected_resources(state, game, replay_data, action_hint, action_type)

    # Track engine actions for undo
    actions_this_step = auto_actions_taken + 1
    state.replay_actions_per_step.append(actions_this_step)
    state.replay_index += 1
    replay_finished = mark_finished_if_needed(state, parsed_actions)

    # Recorded source completion, not the live-game threshold, ends a replay.
    if replay_finished:
        state.game_running = False
        state.game_log.append({
            "type": "general",
            "timestamp": time.time(),
            "message": "Replay complete!",
        })

    return _step_result(state, replay_data, action, action_hint, parsed_actions, replay_finished)


def _step_result(
    state: ReplayRuntimeState,
    replay_data: ReplayArchive,
    action: Action,
    action_hint: ActionHint,
    parsed_actions: ParsedActions,
    replay_finished: bool,
) -> ReplayOutcome:
    # Get raw Colonist event for comparison
    raw_events = list_field(replay_data, "events")
    raw_event_idx = action_hint.get("index", state.replay_index - 1)
    raw_event = raw_events[raw_event_idx] if 0 <= raw_event_idx < len(raw_events) else None

    return {
        "status": "ok",
        "event_index": state.replay_index,
        "total_events": len(parsed_actions),
        "action": str(action),
        "finished": replay_finished,
        "colonist_event": raw_event,
        "engine_translation": action_hint,
    }


def _validate_expected_resources(
    state: ReplayRuntimeState,
    game: GameEngine,
    replay_data: ReplayArchive,
    action_hint: ActionHint,
    action_type: str,
) -> None:
    """Validate resources match (divergence detection)."""
    expected_resources = action_hint.get("expected_resources", {})
    colonist_to_engine = mapping_field(replay_data, "colonist_color_to_engine_idx")
    if expected_resources and action_type == "ROLL":
        print(f"[DEBUG] Step {state.replay_index}: {action_type} dice={action_hint.get('dice')}")
        for colonist_id, card_list in expected_resources.items():
            engine_idx = seat_index(colonist_to_engine, colonist_id)
            if engine_idx is not None:
                expected = colonist_cards_to_freqdeck(card_list)
                actual = get_engine_player_resources(game, engine_idx)
                match = "OK" if expected == actual else "MISMATCH"
                print(f"  P{engine_idx} (c{colonist_id}): engine={actual} expected={expected} [{match}]")
    if not expected_resources:
        return
    mismatches = validate_resources_match(
        game, expected_resources, colonist_to_engine,
        step_info=f"After {action_type} at step {state.replay_index}"
    )
    if not mismatches:
        return
    res_names = _RESOURCE_KEYS
    for m in mismatches:
        diff_str = ", ".join(f"{res_names[i]}:{d:+d}" for i, d in enumerate(m["diff"]) if d != 0)
        player_key = f"P{m['player_idx']}"
        if player_key not in state.first_divergence_step:
            state.first_divergence_step[player_key] = state.replay_index
            print(f"[DIVERGENCE STARTED] Player {m['player_idx']} first diverged at step {state.replay_index} "
                  f"after {action_type}: {diff_str}")
            print(f"  Engine has: {m['engine_has']}, Colonist expects: {m['colonist_expects']}")
        print(f"[DIVERGENCE] Player {m['player_idx']} (colonist {m['colonist_id']}): "
              f"engine={m['engine_has']}, expected={m['colonist_expects']}, diff=[{diff_str}]")
        state.game_log.append({
            "type": "warning",
            "timestamp": time.time(),
            "message": f"[DIVERGENCE after step {state.replay_index}] Player {m['player_idx']}: {diff_str}",
            "details": {
                "action_type": action_type,
                "engine_resources": m["engine_has"],
                "expected_resources": m["colonist_expects"],
                "diff": m["diff"],
                "first_divergence": state.first_divergence_step.get(player_key),
            }
        })
