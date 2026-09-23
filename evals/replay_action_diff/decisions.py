"""Decision-point construction over a replayed game."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping

from cle.env.observation_formatter import CatanObservationFormatter
from cle.replay.contracts import ReplayPayload, ReplayRuntimeState, as_list, as_mapping
from cle.replay.runtime.step_executor import replay_step_logic
from cle.sandbox.decision import build_decision_context
from evals.decision_buckets import decision_state_features
from evals.json_types import JsonDict, JsonList, as_dict
from evals.replay_action_diff.actors import _compound_label, canonicalize_policy_action_order
from evals.replay_action_diff.contracts import (
    SCHEMA_VERSION,
    DecisionPoint,
    replay_parsed_actions,
    require_archive,
    require_engine,
    to_json,
)
from evals.replay_action_diff.identity import _color_name
from evals.replay_action_diff.matching import (
    _decision_signature,
    match_human_action,
    prepare_decision_game,
)
from playground.game_viewer.app import app
from playground.game_viewer.state import server_state


def build_decision_point(
    state: ReplayRuntimeState,
    replay_index: int,
    actor_index: int,
    actor_source: str,
) -> tuple[JsonDict, DecisionPoint | None, int | None]:
    parsed_actions = replay_parsed_actions(state)
    source_hint: Mapping[str, object] = parsed_actions[replay_index]
    effective_hint, followup_index, compound_error = _compound_label(
        parsed_actions, replay_index
    )
    actor_color = require_engine(state).state.colors[actor_index]
    archive = require_archive(state)
    base_record: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": f"{archive['game_id']}:{replay_index}",
        "game_id": to_json(archive["game_id"], "game_id"),
        "replay_index": replay_index,
        "source_replay_index": to_json(
            source_hint.get("_source_replay_index", replay_index), "_source_replay_index"
        ),
        "source_event_index": to_json(source_hint.get("index"), "index"),
        "source_action_type": to_json(source_hint.get("type"), "type"),
        "effective_action_type": to_json(effective_hint.get("type"), "type"),
        "compound_followup_replay_index": followup_index,
        "actor": {
            "colonist_player": to_json(source_hint.get("player"), "player"),
            "engine_index": actor_index,
            "engine_color": _color_name(actor_color),
            "source": actor_source,
        },
    }

    if compound_error is not None:
        base_record.update(
            {
                "classification": "unmappable",
                "reason": compound_error,
                "available_actions": [],
            }
        )
        return base_record, None, followup_index

    decision_game, preparation_error = prepare_decision_game(
        state, source_hint, actor_index
    )
    if preparation_error is not None or decision_game is None:
        classification = (
            "coarse"
            if "exact indexed interface" in str(preparation_error)
            else "unmappable"
        )
        base_record.update(
            {
                "classification": classification,
                "reason": preparation_error,
                "available_actions": [],
            }
        )
        return base_record, None, followup_index

    context = build_decision_context(decision_game, actor_color)
    base_record["state_features"] = decision_state_features(
        decision_game.state,
        actor_color,
        context.legal_actions,
    )
    formatter = CatanObservationFormatter()
    available_actions: JsonList = [
        {
            "index": index,
            "action": str(action),
            "description": formatter._format_single_action(
                action,
                context.observation,
            ),
        }
        for index, action in enumerate(context.legal_actions)
    ]
    match = match_human_action(context.legal_actions, effective_hint, state)
    event_sequences = [event.sequence for event in context.events]
    base_record.update(
        {
            "classification": match["status"],
            "reason": match.get("reason"),
            "phase": context.phase,
            "forced": len(available_actions) == 1,
            "available_actions": available_actions,
            "activity_window": {
                "scope": "complete_visible_game_events",
                "row_count": len(context.events),
                "start_sequence": min(event_sequences) if event_sequences else None,
                "end_sequence": max(event_sequences) if event_sequences else None,
                "truncated": False,
            },
        }
    )

    if match["status"] != "exact":
        return base_record, None, followup_index

    human_index = match["action_index"]
    if human_index is None:
        raise RuntimeError(f"Exact match at replay row {replay_index} has no action index")
    selected = as_dict(available_actions[human_index], "selected action")
    base_record["human"] = {
        "action_index": human_index,
        "action": selected["action"],
        "description": selected["description"],
        "normalization": match["normalization"],
    }
    base_record["signature"] = _decision_signature(base_record)
    return base_record, DecisionPoint(base_record, context), followup_index


def _load_replay(game_id: str) -> ReplayRuntimeState:
    app.config["TESTING"] = True
    server_state.reset()
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        response = app.test_client().post(
            "/api/load-replay", json={"game_id": game_id}
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Could not load replay {game_id}: {response.get_json()}\n"
            f"{output.getvalue()[-4_000:]}"
        )
    archive = require_archive(server_state)
    if not isinstance(archive, dict):
        raise TypeError(f"Loaded replay archive is not a dict: {type(archive).__name__}")
    canonical_actions, changes = canonicalize_policy_action_order(
        [
            as_mapping(row, "parsed_actions entry")
            for row in as_list(archive["parsed_actions"], "parsed_actions")
        ]
    )
    archive["parsed_actions"] = canonical_actions
    archive["evaluation_canonicalizations"] = changes
    return server_state


def _step_replay(state: ReplayRuntimeState) -> ReplayPayload:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        result = replay_step_logic(state, lambda: None, allow_lookahead=False)
    if isinstance(result, tuple):
        payload, status = result
        raise RuntimeError(
            f"Replay step {state.replay_index} failed with HTTP-like status {status}: "
            f"{payload}\n{output.getvalue()[-4_000:]}"
        )
    if result.get("error"):
        raise RuntimeError(
            f"Replay step {state.replay_index} failed: {result}\n"
            f"{output.getvalue()[-4_000:]}"
        )
    return result


def _resolve_target_player(state: ReplayRuntimeState, target_player: str) -> int:
    player_id: object
    if target_player == "captured":
        player_id = require_archive(state).get("player_perspective")
    else:
        try:
            player_id = int(target_player)
        except ValueError as exc:
            raise ValueError("target_player must be 'captured' or a Colonist color ID") from exc
    if player_id is None:
        raise ValueError("Replay does not declare a captured player perspective")
    if not isinstance(player_id, (str, int, float)):
        raise TypeError(f"Replay player perspective is not numeric: {player_id!r}")
    return int(player_id)

