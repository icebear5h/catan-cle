"""Causal full-replay action-selection comparisons against recorded humans."""

from __future__ import annotations

from cle.replay.runtime.access import get_game_engine

import asyncio
import contextlib
import hashlib
import io
import json
import math
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cle.env.observation_formatter import CatanObservationFormatter
from cle.harness.decision import request_player_attempt
from cle.harness.providers import OpenRouterConfig, OpenRouterTransport
from cle.harness.reasoning import (
    native_reasoning_request,
    native_reasoning_returned,
    reasoning_token_count,
)
from cle.harness.suite import load_context_suite
from cle.players.validation import action_from_choice
from cle.sandbox.decision import build_decision_context
from evals.decision_buckets import (
    classify_decision_records,
    decision_state_features,
    load_decision_bucket_suite,
)
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import RESOURCES
from playground.game_viewer.app import app
from cle.replay.runtime.action_matcher import _colonist_xy_to_engine_coord
from cle.replay.runtime.step_executor import (
    TURN_OWNER_ACTIONS,
    _ensure_root_offer,
    replay_step_logic,
)
from playground.game_viewer.state import server_state

SCHEMA_VERSION = "replay-action-diff-v2"
COMPARISON_PARSER_VERSION = "shared-action-parser-v3"
COMPOUND_ACTIONS = {
    "PLAY_MONOPOLY": "MONOPOLY_RESOURCE",
    "PLAY_YEAR_OF_PLENTY": "YEAR_OF_PLENTY_RESOURCES",
}
COMPOUND_FOLLOWUPS = set(COMPOUND_ACTIONS.values())
ASYNC_TRADE_RESPONSES = {"ACCEPT_TRADE", "REJECT_TRADE"}
# Historical comparisons stay index-only; exact new payloads do not relabel old decisions.
COARSE_ACTIONS = {
    "OFFER_TRADE": "trade terms are not represented by the indexed meta-action",
    "COUNTER_OFFER": "counter-offer terms are not represented by the indexed meta-action",
    "DISCARD": "the indexed interface omits the human's controlled card selection",
}
LIFECYCLE_ACTIONS = {
    "CLEAR_TRADE_RESPONSE": "trade lifecycle update, not an indexed decision",
    "CLOSE_TRADE": "trade lifecycle closure, not an indexed decision",
}


@dataclass
class DecisionPoint:
    """One exact human decision and its shared provider-safe player context."""

    record: Dict[str, Any]
    context: Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _color_name(color: Any) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _action_type_name(action: Any) -> str:
    action_type = getattr(action, "action_type", None)
    if hasattr(action_type, "value"):
        return str(action_type.value)
    return str(action_type).replace("ActionType.", "")


def _normalize_edge(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return min(value), max(value)


def _resource_sort_key(resource: Any) -> int:
    try:
        return RESOURCES.index(resource)
    except ValueError:
        return len(RESOURCES)


def _same_resource_choice(left: Any, right: Any) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    return sorted(left, key=_resource_sort_key) == sorted(
        right, key=_resource_sort_key
    )


def _maritime_value_matches(value: Any, given: Any, received: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 5:
        return False
    if not isinstance(given, (list, tuple)) or len(given) != len(RESOURCES):
        return False
    if not isinstance(received, (list, tuple)) or len(received) != len(RESOURCES):
        return False

    outgoing = [0] * len(RESOURCES)
    for resource in value[:4]:
        if resource is None:
            continue
        if resource not in RESOURCES:
            return False
        outgoing[RESOURCES.index(resource)] += 1

    incoming = [0] * len(RESOURCES)
    incoming_resource = value[4]
    if incoming_resource not in RESOURCES:
        return False
    incoming[RESOURCES.index(incoming_resource)] = 1
    return tuple(outgoing) == tuple(given) and tuple(incoming) == tuple(received)


def _engine_color_for_colonist(state: Any, player_id: Any) -> Optional[Any]:
    mapping = state.replay_data.get("colonist_color_to_engine_idx", {})
    player_index = mapping.get(str(player_id))
    colors = get_game_engine(state).state.colors
    if not isinstance(player_index, int) or not 0 <= player_index < len(colors):
        return None
    return colors[player_index]


def infer_actor(state: Any, action_hint: Dict[str, Any]) -> Tuple[Optional[int], str]:
    """Resolve the acting engine seat without guessing from turn order."""
    player_id = action_hint.get("player")
    mapping = state.replay_data.get("colonist_color_to_engine_idx", {})
    player_index = mapping.get(str(player_id))
    if isinstance(player_index, int):
        return player_index, "replay_player"

    if action_hint.get("type") != "BUILD_CITY":
        return None, "unresolved"

    corner = action_hint.get("colonist_corner")
    node = state.corner_to_node_map.get(f"_{corner}")
    building = get_game_engine(state).state.board.buildings.get(node)
    if not building:
        return None, "unresolved"
    owner_color = building[0]
    inferred_index = get_game_engine(state).state.color_to_index.get(owner_color)
    if not isinstance(inferred_index, int):
        return None, "unresolved"
    return inferred_index, "pre_action_building_owner"


def canonicalize_policy_action_order(
    parsed_actions: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Put same-event robber movement before its resulting steal decision."""
    canonical = []
    for source_index, action in enumerate(parsed_actions):
        copied = deepcopy(action)
        copied["_source_replay_index"] = source_index
        canonical.append(copied)

    changes = []
    index = 0
    while index + 1 < len(canonical):
        first = canonical[index]
        second = canonical[index + 1]
        should_swap = (
            first.get("type") == "STEAL"
            and second.get("type") == "MOVE_ROBBER"
            and first.get("index") == second.get("index")
            and first.get("player") == second.get("player")
        )
        if should_swap:
            canonical[index], canonical[index + 1] = second, first
            changes.append(
                {
                    "canonical_indices": [index, index + 1],
                    "source_replay_indices": [
                        first["_source_replay_index"],
                        second["_source_replay_index"],
                    ],
                    "raw_event_index": first.get("index"),
                    "reason": "MOVE_ROBBER is the causal prerequisite of STEAL",
                }
            )
            index += 2
            continue
        index += 1
    return canonical, changes


def _compound_label(
    parsed_actions: Sequence[Dict[str, Any]], replay_index: int
) -> Tuple[Dict[str, Any], Optional[int], Optional[str]]:
    source = parsed_actions[replay_index]
    expected_followup = COMPOUND_ACTIONS.get(source.get("type"))
    if expected_followup is None:
        return source, None, None
    if replay_index + 1 >= len(parsed_actions):
        return source, None, "missing compound follow-up"

    followup = parsed_actions[replay_index + 1]
    if (
        followup.get("type") != expected_followup
        or followup.get("player") != source.get("player")
    ):
        return source, None, f"expected adjacent {expected_followup} follow-up"
    return followup, replay_index + 1, None


def _response_menu(
    game_state: Any,
    action_hint: Dict[str, Any],
    creator_color: Any,
) -> List[Any]:
    """Return only the actions scoped to one asynchronous trade response."""
    _ensure_root_offer(game_state, creator_color, action_hint)
    generated = generate_playable_actions(game_state)
    scoped = []
    counter_offer_added = False
    for action in generated:
        action_type = _action_type_name(action)
        if action_type in ASYNC_TRADE_RESPONSES and action.value == creator_color:
            scoped.append(action)
        elif action_type == "COUNTER_OFFER" and not counter_offer_added:
            scoped.append(action)
            counter_offer_added = True
    return scoped


def prepare_decision_game(
    state: Any,
    action_hint: Dict[str, Any],
    actor_index: int,
) -> Tuple[Optional[Any], Optional[str]]:
    """Create an isolated actor-perspective state before the human action."""
    decision_game = get_game_engine(state).copy()
    game_state = decision_game.state
    game_state.current_player_index = actor_index
    action_type = action_hint.get("type")

    if action_type in ASYNC_TRADE_RESPONSES:
        if action_hint.get("is_counter_offer"):
            return None, "counter-offer responses lack an exact indexed interface"
        creator_color = _engine_color_for_colonist(state, action_hint.get("creator"))
        if creator_color is None:
            return None, "trade creator does not map to an engine color"
        game_state.playable_actions = _response_menu(
            game_state, action_hint, creator_color
        )
    else:
        if action_type in TURN_OWNER_ACTIONS or action_type == "BUILD_CITY":
            game_state.current_turn_index = actor_index
        game_state.playable_actions = generate_playable_actions(game_state)

    if not game_state.playable_actions:
        return None, "prepared decision state has no indexed legal actions"
    return decision_game, None


def _candidate_indices(
    actions: Sequence[Any],
    action_hint: Dict[str, Any],
    state: Any,
) -> Tuple[List[int], str]:
    action_type = action_hint.get("type")
    candidates = []
    normalization = "exact type and value"

    if action_type == "ROLL":
        normalization = "ROLL intent; dice outcome omitted"
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "ROLL"
        ]
    elif action_type == "BUY_DEVELOPMENT_CARD":
        normalization = "BUY_DEVELOPMENT_CARD intent; random card identity omitted"
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "BUY_DEVELOPMENT_CARD"
        ]
    elif action_type == "STEAL":
        normalization = "STEAL victim; random stolen resource omitted"
        victim_color = _engine_color_for_colonist(state, action_hint.get("victim"))
        for index, action in enumerate(actions):
            if _action_type_name(action) != "STEAL":
                continue
            value = action.value
            if isinstance(value, (list, tuple)) and value and value[0] == victim_color:
                candidates.append(index)
    elif action_type in {"BUILD_SETTLEMENT", "BUILD_CITY"}:
        target = state.corner_to_node_map.get(
            f"_{action_hint.get('colonist_corner')}"
        )
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == action_type and action.value == target
        ]
    elif action_type == "BUILD_ROAD":
        target = _normalize_edge(
            state.edge_to_edge_map.get(f"_{action_hint.get('colonist_edge')}")
        )
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "BUILD_ROAD"
            and _normalize_edge(action.value) == target
        ]
    elif action_type == "MOVE_ROBBER":
        tile_info = action_hint.get("tile_info") or {}
        x = tile_info.get("x")
        y = tile_info.get("y")
        if x is None or y is None:
            return [], "robber destination is unavailable"
        target = _colonist_xy_to_engine_coord(x, y)
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "MOVE_ROBBER" and action.value == target
        ]
    elif action_type == "MARITIME_TRADE":
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "MARITIME_TRADE"
            and _maritime_value_matches(
                action.value,
                action_hint.get("given"),
                action_hint.get("received"),
            )
        ]
    elif action_type == "MONOPOLY_RESOURCE":
        normalization = "combined PLAY_MONOPOLY plus resource choice"
        resource = action_hint.get("resource")
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "PLAY_MONOPOLY"
            and action.value == resource
        ]
    elif action_type == "YEAR_OF_PLENTY_RESOURCES":
        normalization = "combined PLAY_YEAR_OF_PLENTY plus resource choice"
        resources = action_hint.get("resources") or []
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "PLAY_YEAR_OF_PLENTY"
            and _same_resource_choice(action.value, resources)
        ]
    elif action_type in ASYNC_TRADE_RESPONSES:
        creator_color = _engine_color_for_colonist(state, action_hint.get("creator"))
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == action_type
            and action.value == creator_color
        ]
    elif action_type == "CONFIRM_TRADE":
        acceptor_color = _engine_color_for_colonist(
            state, action_hint.get("acceptor")
        )
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "CONFIRM_TRADE"
            and action.value == acceptor_color
        ]
    else:
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == action_type
            and action.value is None
        ]

    return candidates, normalization


def match_human_action(
    actions: Sequence[Any],
    action_hint: Dict[str, Any],
    state: Any,
) -> Dict[str, Any]:
    """Map a recorded human choice to exactly one indexed interface action."""
    action_type = str(action_hint.get("type") or "")
    if action_type in COARSE_ACTIONS:
        return {
            "status": "coarse",
            "reason": COARSE_ACTIONS[action_type],
            "action_index": None,
        }
    if action_type in LIFECYCLE_ACTIONS:
        return {
            "status": "lifecycle",
            "reason": LIFECYCLE_ACTIONS[action_type],
            "action_index": None,
        }
    if action_type in COMPOUND_ACTIONS:
        return {
            "status": "unmappable",
            "reason": "compound announcement is missing its resource-choice label",
            "action_index": None,
        }
    if action_hint.get("is_counter_offer") and action_type in ASYNC_TRADE_RESPONSES:
        return {
            "status": "coarse",
            "reason": "counter-offer response is not represented exactly by the engine menu",
            "action_index": None,
        }

    candidates, normalization = _candidate_indices(actions, action_hint, state)
    if len(candidates) == 1:
        return {
            "status": "exact",
            "reason": None,
            "action_index": candidates[0],
            "normalization": normalization,
        }
    if not candidates:
        return {
            "status": "unmappable",
            "reason": f"no indexed legal action matched {action_type}",
            "action_index": None,
            "normalization": normalization,
        }
    return {
        "status": "ambiguous",
        "reason": f"{len(candidates)} indexed legal actions matched {action_type}",
        "action_index": None,
        "candidate_indices": candidates,
        "normalization": normalization,
    }


def _decision_signature(record: Dict[str, Any]) -> str:
    payload = {
        "replay_index": record["replay_index"],
        "human_action_index": record.get("human", {}).get("action_index"),
        "available_actions": record.get("available_actions", []),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def semantic_manifest_hash(manifest: Sequence[Dict[str, Any]]) -> str:
    """Hash decision meaning while ignoring unstable legal-menu ordering."""
    payload = []
    for record in manifest:
        payload.append(
            {
                "decision_id": record.get("decision_id"),
                "source_event_index": record.get("source_event_index"),
                "source_action_type": record.get("source_action_type"),
                "effective_action_type": record.get("effective_action_type"),
                "classification": record.get("classification"),
                "reason": record.get("reason"),
                "forced": record.get("forced"),
                "human_action": (record.get("human") or {}).get("action"),
                "available_actions": sorted(
                    action.get("action")
                    for action in record.get("available_actions", [])
                ),
            }
        )
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_fingerprint(state: Any) -> str:
    game_state = get_game_engine(state).state
    board = game_state.board
    roads = tuple(
        sorted(
            (min(left, right), max(left, right), _color_name(color))
            for (left, right), color in board.roads.items()
            if left < right
        )
    )
    buildings = tuple(
        sorted(
            (node, _color_name(color), str(building))
            for node, (color, building) in board.buildings.items()
        )
    )
    payload = (
        state.replay_index,
        len(game_state.actions),
        game_state.current_player_index,
        game_state.current_turn_index,
        str(game_state.current_prompt),
        tuple(sorted(str(action) for action in game_state.playable_actions)),
        tuple(sorted(game_state.player_state.items())),
        tuple(game_state.resource_freqdeck),
        tuple(game_state.development_listdeck),
        repr(game_state.trade_window),
        buildings,
        roads,
        board.robber_coordinate,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def build_decision_point(
    state: Any,
    replay_index: int,
    actor_index: int,
    actor_source: str,
) -> Tuple[Dict[str, Any], Optional[DecisionPoint], Optional[int]]:
    parsed_actions = state.replay_data["parsed_actions"]
    source_hint = parsed_actions[replay_index]
    effective_hint, followup_index, compound_error = _compound_label(
        parsed_actions, replay_index
    )
    actor_color = get_game_engine(state).state.colors[actor_index]
    base_record = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": f"{state.replay_data['game_id']}:{replay_index}",
        "game_id": state.replay_data["game_id"],
        "replay_index": replay_index,
        "source_replay_index": source_hint.get("_source_replay_index", replay_index),
        "source_event_index": source_hint.get("index"),
        "source_action_type": source_hint.get("type"),
        "effective_action_type": effective_hint.get("type"),
        "compound_followup_replay_index": followup_index,
        "actor": {
            "colonist_player": source_hint.get("player"),
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
    available_actions = [
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
    selected = available_actions[human_index]
    base_record["human"] = {
        "action_index": human_index,
        "action": selected["action"],
        "description": selected["description"],
        "normalization": match["normalization"],
    }
    base_record["signature"] = _decision_signature(base_record)
    return base_record, DecisionPoint(base_record, context), followup_index


def _load_replay(game_id: str) -> Any:
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
    canonical_actions, changes = canonicalize_policy_action_order(
        server_state.replay_data["parsed_actions"]
    )
    server_state.replay_data["parsed_actions"] = canonical_actions
    server_state.replay_data["evaluation_canonicalizations"] = changes
    return server_state


def _step_replay(state: Any) -> Dict[str, Any]:
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


def _resolve_target_player(state: Any, target_player: str) -> int:
    if target_player == "captured":
        player_id = state.replay_data.get("player_perspective")
    else:
        try:
            player_id = int(target_player)
        except ValueError as exc:
            raise ValueError("target_player must be 'captured' or a Colonist color ID") from exc
    if player_id is None:
        raise ValueError("Replay does not declare a captured player perspective")
    return int(player_id)


def scan_replay(game_id: str, target_player: str = "captured") -> Dict[str, Any]:
    """Replay the entire game locally and classify the target human's actions."""
    state = _load_replay(game_id)
    try:
        target_player_id = _resolve_target_player(state, target_player)
        mapping = state.replay_data.get("colonist_color_to_engine_idx", {})
        target_index = mapping.get(str(target_player_id))
        if not isinstance(target_index, int):
            raise ValueError(
                f"Target Colonist player {target_player_id} is not in replay play order"
            )
        target_color = get_game_engine(state).state.colors[target_index]
        parsed_actions = state.replay_data["parsed_actions"]
        records = []
        step_statuses = Counter()
        compound_followups = set()

        while state.replay_index < len(parsed_actions):
            replay_index = state.replay_index
            hint = parsed_actions[replay_index]
            actor_index, actor_source = infer_actor(state, hint)

            if replay_index in compound_followups and actor_index == target_index:
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "decision_id": f"{game_id}:{replay_index}",
                        "game_id": game_id,
                        "replay_index": replay_index,
                        "source_replay_index": hint.get(
                            "_source_replay_index", replay_index
                        ),
                        "source_event_index": hint.get("index"),
                        "source_action_type": hint.get("type"),
                        "effective_action_type": hint.get("type"),
                        "classification": "lifecycle",
                        "reason": "compound decision follow-up scored at the prior announcement",
                        "available_actions": [],
                        "actor": {
                            "colonist_player": hint.get("player"),
                            "engine_index": actor_index,
                            "engine_color": _color_name(target_color),
                            "source": actor_source,
                        },
                    }
                )
            elif actor_index == target_index:
                record, _, followup_index = build_decision_point(
                    state, replay_index, actor_index, actor_source
                )
                records.append(record)
                if followup_index is not None:
                    compound_followups.add(followup_index)

            result = _step_replay(state)
            step_statuses[result.get("status", "unknown")] += 1

        if state.replay_index != len(parsed_actions):
            raise RuntimeError(
                f"Replay stopped at {state.replay_index}/{len(parsed_actions)}"
            )
        bucket_assignments = classify_decision_records(records)
        for record, assignment in zip(records, bucket_assignments, strict=True):
            record["bucket_assignment"] = assignment

        semantic_errors = [
            issue
            for issue in state.replay_semantic_issues
            if issue.get("severity") == "error"
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "game_id": game_id,
            "target_player_id": target_player_id,
            "target_engine_index": target_index,
            "target_engine_color": _color_name(target_color),
            "replay_file": state.replay_data.get("file"),
            "parsed_action_count": len(parsed_actions),
            "canonicalizations": deepcopy(
                state.replay_data.get("evaluation_canonicalizations", [])
            ),
            "records": records,
            "step_statuses": dict(step_statuses),
            "semantic_errors": deepcopy(semantic_errors),
        }
    finally:
        server_state.reset()


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _latest_responses(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    latest = {}
    for row in rows:
        latest[(row["decision_id"], row["model_id"])] = row
    return latest


def _parse_shared_action_index(
    raw_response: str,
    action_count: int,
) -> Tuple[Optional[int], Optional[str]]:
    """Parse only the shared indexed-action contract from immutable raw text."""
    action_tags = re.findall(
        r"<action>\s*(.*?)\s*</action>",
        raw_response or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    numeric_tags = [value for value in action_tags if value.strip().isdigit()]
    errors = []
    if len(set(numeric_tags)) > 1:
        errors.append("Model returned conflicting numeric action tags; used the last one")
    action_text = numeric_tags[-1] if numeric_tags else None
    if action_text is None:
        direct = re.findall(
            r"(?:action|move)(?:_index)?\s*[:=]\s*(\d+)",
            raw_response or "",
            flags=re.IGNORECASE,
        )
        action_text = direct[-1] if direct else None
    if action_text is None:
        errors.append("Model did not return a parseable action index")
        return None, "; ".join(errors)
    candidate = int(action_text.strip())
    if not 0 <= candidate < action_count:
        errors.append(
            f"Model selected action {candidate}, outside the valid range 0-{action_count - 1}"
        )
        return None, "; ".join(errors)
    return candidate, "; ".join(errors) if errors else None


def normalize_response_selection(row: Dict[str, Any]) -> Dict[str, Any]:
    """Reparse immutable raw output with the current comparison parser."""
    normalized = deepcopy(row)
    normalized["comparison_parser_version"] = COMPARISON_PARSER_VERSION
    if normalized.get("error") or not normalized.get("result"):
        return normalized

    result = normalized["result"]
    if "raw_response" not in result:
        return normalized
    available_actions = result.get("available_actions") or []
    action_index, parse_error = _parse_shared_action_index(
        str(result.get("raw_response") or ""),
        len(available_actions),
    )
    selected = (
        available_actions[action_index]
        if isinstance(action_index, int) and 0 <= action_index < len(available_actions)
        else None
    )
    result.update(
        {
            "action_index": action_index,
            "action": selected.get("action") if selected else None,
            "action_description": selected.get("description") if selected else None,
            "parse_error": parse_error,
        }
    )
    normalized["model_action_index"] = action_index
    normalized["agreement"] = action_index == normalized.get("human_action_index")
    return normalized


def validate_response_compatibility(
    manifest: Sequence[Dict[str, Any]],
    response_rows: Sequence[Dict[str, Any]],
) -> None:
    """Verify stored calls against current order-invariant decision semantics."""
    exact = {
        row["decision_id"]: row
        for row in manifest
        if row.get("classification") == "exact"
    }
    for response in _latest_responses(response_rows).values():
        decision = exact.get(response.get("decision_id"))
        result = response.get("result") or {}
        if decision is None or not result:
            continue
        stored_actions = Counter(
            action.get("action") for action in result.get("available_actions", [])
        )
        current_actions = Counter(
            action.get("action") for action in decision.get("available_actions", [])
        )
        if stored_actions != current_actions:
            raise RuntimeError(
                f"Stored response menu changed semantically for {response['decision_id']}"
            )
        human_index = response.get("human_action_index")
        stored_menu = result.get("available_actions", [])
        if not isinstance(human_index, int) or not 0 <= human_index < len(stored_menu):
            raise RuntimeError(
                f"Stored human action index is invalid for {response['decision_id']}"
            )
        if stored_menu[human_index].get("action") != decision["human"]["action"]:
            raise RuntimeError(
                f"Stored human action changed semantically for {response['decision_id']}"
            )


def reasoning_request_for_model(
    model_id: str,
    effort: str = "xhigh",
) -> Dict[str, Any]:
    """Return the explicit native-reasoning condition recorded for a model."""
    if not model_id:
        raise ValueError("model_id is required")
    return native_reasoning_request(effort)


def _response_cost_usd(row: Dict[str, Any]) -> float:
    result = row.get("result")
    usage = result.get("usage") if isinstance(result, dict) else None
    value = usage.get("cost") if isinstance(usage, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return 0.0
    return float(value)


def _query_model(
    model_id: str,
    context: Any,
    reasoning_effort: str,
    max_tokens: int,
) -> Dict[str, Any]:
    reasoning = reasoning_request_for_model(model_id, reasoning_effort)
    suite = load_context_suite()
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model=model_id,
            temperature=0.2,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )
    )

    async def request_attempt() -> Any:
        try:
            return await request_player_attempt(
                context,
                transport,
                game_plan="",
                session_id=f"{context.context_id}:{model_id}",
                suite=suite,
            )
        finally:
            await transport.aclose()

    attempt = asyncio.run(request_attempt())
    response = attempt.model_response
    if response is None:
        raise RuntimeError("Agent attempt did not retain its model response")
    choice = attempt.choice
    usage = dict(response.usage)
    formatter = CatanObservationFormatter()
    available_actions = [
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
    action_index = choice.action_index if choice is not None else None
    selected = (
        action_from_choice(context, choice)
        if choice is not None and attempt.validation_error is None
        else None
    )
    native_returned = native_reasoning_returned(
        response.native_reasoning,
        response.native_reasoning_details,
        usage,
    )
    messages = attempt.model_request.messages if attempt.model_request is not None else ()
    return {
        "context_version": f"{suite.id}@{suite.version}",
        "game_plan": choice.game_plan if choice is not None else "",
        "action_index": action_index,
        "action": str(selected) if selected is not None else None,
        "action_description": (
            formatter._format_single_action(selected, context.observation)
            if selected is not None
            else None
        ),
        "parse_error": attempt.validation_error,
        "finish_reason": response.finish_reason,
        "provider_native_finish_reason": response.provider_native_finish_reason,
        "provider_response_id": response.provider_response_id,
        "provider_request_id": response.provider_request_id,
        "response_truncated": response.finish_reason == "length",
        "available_actions": available_actions,
        "raw_response": response.content,
        "latency_ms": response.latency_ms,
        "usage": usage,
        "requested_model": model_id,
        "model": response.model or model_id,
        "native_reasoning": response.native_reasoning,
        "native_reasoning_details": list(response.native_reasoning_details),
        "native_reasoning_returned": native_returned,
        "native_reasoning_missing": not native_returned,
        "reasoning_request": reasoning,
        "reasoning_tokens": reasoning_token_count(usage),
        "system_prompt": messages[0].content if messages else "",
        "context_prompt": messages[-1].content if messages else "",
    }


def query_replay(
    game_id: str,
    target_player: str,
    manifest: Sequence[Dict[str, Any]],
    models: Sequence[str],
    responses_path: Path,
    reasoning_effort: str = "xhigh",
    max_tokens: int = 8_192,
    max_requests_per_model: Optional[int] = None,
    max_cost_usd: Optional[float] = None,
    retry_errors: bool = False,
) -> Dict[str, int]:
    """Query each model at every exact point, then advance only the human replay."""
    exact_by_index = {
        row["replay_index"]: row
        for row in manifest
        if row.get("classification") == "exact"
    }
    existing_rows = _read_jsonl(responses_path)
    latest = _latest_responses(existing_rows)
    new_requests = Counter()
    manifest_ids = {record["decision_id"] for record in manifest}
    spent_usd = sum(
        _response_cost_usd(row)
        for (decision_id, model_id), row in latest.items()
        if decision_id in manifest_ids and model_id in models
    )
    state = _load_replay(game_id)

    try:
        target_player_id = _resolve_target_player(state, target_player)
        target_index = state.replay_data["colonist_color_to_engine_idx"].get(
            str(target_player_id)
        )
        parsed_actions = state.replay_data["parsed_actions"]

        while state.replay_index < len(parsed_actions):
            replay_index = state.replay_index
            expected = exact_by_index.get(replay_index)
            if expected is not None:
                hint = parsed_actions[replay_index]
                actor_index, actor_source = infer_actor(state, hint)
                if actor_index != target_index:
                    raise RuntimeError(
                        f"Decision actor changed at replay row {replay_index}: "
                        f"expected seat {target_index}, got {actor_index}"
                    )
                record, point, _ = build_decision_point(
                    state, replay_index, actor_index, actor_source
                )
                if point is None or record.get("classification") != "exact":
                    raise RuntimeError(
                        f"Exact decision no longer maps at replay row {replay_index}: {record}"
                    )
                if record["signature"] != expected["signature"]:
                    raise RuntimeError(
                        f"Decision menu changed at replay row {replay_index}"
                    )

                models_to_query = []
                for model_id in models:
                    prior = latest.get((record["decision_id"], model_id))
                    if prior is not None and not (retry_errors and prior.get("error")):
                        continue
                    if (
                        max_requests_per_model is not None
                        and new_requests[model_id] >= max_requests_per_model
                    ):
                        continue
                    if max_cost_usd is not None and spent_usd >= max_cost_usd:
                        continue
                    models_to_query.append(model_id)

                before = _state_fingerprint(state)
                response_rows = []
                if models_to_query:
                    with ThreadPoolExecutor(max_workers=len(models_to_query)) as executor:
                        futures = {
                            executor.submit(
                                _query_model,
                                model_id,
                                point.context,
                                reasoning_effort,
                                max_tokens,
                            ): model_id
                            for model_id in models_to_query
                        }
                        for future in as_completed(futures):
                            model_id = futures[future]
                            new_requests[model_id] += 1
                            try:
                                result = future.result()
                                model_action_index = result.get("action_index")
                                row = {
                                    "schema_version": SCHEMA_VERSION,
                                    "recorded_at": utc_now(),
                                    "decision_id": record["decision_id"],
                                    "game_id": game_id,
                                    "replay_index": replay_index,
                                    "model_id": model_id,
                                    "human_action_index": record["human"]["action_index"],
                                    "model_action_index": model_action_index,
                                    "agreement": model_action_index
                                    == record["human"]["action_index"],
                                    "error": None,
                                    "result": result,
                                }
                            except Exception as exc:
                                row = {
                                    "schema_version": SCHEMA_VERSION,
                                    "recorded_at": utc_now(),
                                    "decision_id": record["decision_id"],
                                    "game_id": game_id,
                                    "replay_index": replay_index,
                                    "model_id": model_id,
                                    "human_action_index": record["human"]["action_index"],
                                    "model_action_index": None,
                                    "agreement": False,
                                    "error": {
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                    "result": None,
                                }
                            response_rows.append(row)
                            latest[(record["decision_id"], model_id)] = row
                            spent_usd += _response_cost_usd(row)
                    _append_jsonl(responses_path, response_rows)

                after = _state_fingerprint(state)
                if after != before:
                    raise RuntimeError(
                        f"Model query mutated replay state at row {replay_index}"
                    )

            _step_replay(state)

        if state.replay_index != len(parsed_actions):
            raise RuntimeError(
                f"Query replay stopped at {state.replay_index}/{len(parsed_actions)}"
            )
        semantic_errors = [
            issue
            for issue in state.replay_semantic_issues
            if issue.get("severity") == "error"
        ]
        if semantic_errors:
            raise RuntimeError(
                f"Causal query replay produced {len(semantic_errors)} semantic errors"
            )
        return dict(new_requests)
    finally:
        server_state.reset()


def _percent(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def build_comparisons(
    manifest: Sequence[Dict[str, Any]],
    response_rows: Sequence[Dict[str, Any]],
    models: Sequence[str],
) -> List[Dict[str, Any]]:
    latest = _latest_responses(
        [normalize_response_selection(row) for row in response_rows]
    )
    comparisons = []
    for decision in manifest:
        if decision.get("classification") != "exact":
            continue
        model_results = {}
        human_candidates = []
        menu_orders = []
        for model_id in models:
            response = latest.get((decision["decision_id"], model_id))
            result = response.get("result") if response else None
            menu = result.get("available_actions", []) if result else []
            model_index = response.get("model_action_index") if response else None
            human_index = response.get("human_action_index") if response else None
            selected = (
                menu[model_index]
                if isinstance(model_index, int) and 0 <= model_index < len(menu)
                else None
            )
            stored_human = (
                menu[human_index]
                if isinstance(human_index, int) and 0 <= human_index < len(menu)
                else None
            )
            if stored_human is not None:
                human_candidates.append((human_index, stored_human))
            if menu:
                menu_orders.append([action.get("action") for action in menu])
            agreement = (
                selected.get("action") == stored_human.get("action")
                if selected and stored_human
                else bool(response and response.get("agreement"))
            )
            model_results[model_id] = {
                "response_present": response is not None,
                "error": response.get("error") if response else None,
                "parse_error": result.get("parse_error") if result else None,
                "action_index": model_index,
                "action": selected.get("action") if selected else None,
                "description": selected.get("description") if selected else None,
                "agreement": agreement,
            }

        if human_candidates:
            human_actions = {
                candidate.get("action") for _, candidate in human_candidates
            }
            if len(human_actions) != 1:
                raise RuntimeError(
                    f"Stored models disagree on the human label for {decision['decision_id']}"
                )
            human_index, human_selected = human_candidates[0]
            human = {
                "action_index": human_index,
                "action": human_selected.get("action"),
                "description": human_selected.get("description"),
                "normalization": decision["human"]["normalization"],
            }
            menu_size = len(menu_orders[0])
        else:
            human = decision["human"]
            menu_size = len(decision["available_actions"])

        comparisons.append(
            {
                "decision_id": decision["decision_id"],
                "replay_index": decision["replay_index"],
                "source_replay_index": decision["source_replay_index"],
                "source_event_index": decision["source_event_index"],
                "source_action_type": decision["source_action_type"],
                "effective_action_type": decision["effective_action_type"],
                "phase": decision.get("phase"),
                "forced": decision.get("forced"),
                "bucket_assignment": decision.get("bucket_assignment"),
                "menu_size": menu_size,
                "menu_order_consistent_across_models": all(
                    menu == menu_orders[0] for menu in menu_orders[1:]
                ),
                "human": human,
                "models": model_results,
            }
        )
    return comparisons


def summarize_run(
    scan: Dict[str, Any],
    response_rows: Sequence[Dict[str, Any]],
    models: Sequence[str],
) -> Dict[str, Any]:
    manifest = scan["records"]
    normalized_rows = [normalize_response_selection(row) for row in response_rows]
    comparisons = build_comparisons(manifest, normalized_rows, models)
    latest = _latest_responses(normalized_rows)
    classification_counts = Counter(row["classification"] for row in manifest)
    summary_models = {}

    for model_id in models:
        eligible = len(comparisons)
        present_rows = [
            latest[(item["decision_id"], model_id)]
            for item in comparisons
            if (item["decision_id"], model_id) in latest
        ]
        api_errors = sum(bool(row.get("error")) for row in present_rows)
        valid_rows = [
            row
            for row in present_rows
            if not row.get("error")
            and isinstance(row.get("model_action_index"), int)
        ]
        agreements = sum(bool(row.get("agreement")) for row in valid_rows)
        parse_errors = sum(
            bool((row.get("result") or {}).get("parse_error"))
            for row in present_rows
            if not row.get("error")
        )
        usage_rows = [
            (row.get("result") or {}).get("usage") or {}
            for row in present_rows
            if not row.get("error")
        ]
        latencies = [
            float((row.get("result") or {}).get("latency_ms"))
            for row in present_rows
            if (row.get("result") or {}).get("latency_ms") is not None
        ]
        by_type = {}
        for action_type in sorted(
            {item["effective_action_type"] for item in comparisons}
        ):
            type_items = [
                item for item in comparisons if item["effective_action_type"] == action_type
            ]
            type_rows = [
                latest[(item["decision_id"], model_id)]
                for item in type_items
                if (item["decision_id"], model_id) in latest
                and not latest[(item["decision_id"], model_id)].get("error")
                and isinstance(
                    latest[(item["decision_id"], model_id)].get("model_action_index"),
                    int,
                )
            ]
            type_agreements = sum(row.get("agreement", False) for row in type_rows)
            by_type[action_type] = {
                "eligible": len(type_items),
                "valid": len(type_rows),
                "agreements": type_agreements,
                "agreement_rate_valid": _percent(type_agreements, len(type_rows)),
            }

        forced_items = [item for item in comparisons if item["forced"]]
        nontrivial_items = [item for item in comparisons if not item["forced"]]

        def subset_stats(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            rows = [
                latest[(item["decision_id"], model_id)]
                for item in items
                if (item["decision_id"], model_id) in latest
                and not latest[(item["decision_id"], model_id)].get("error")
                and isinstance(
                    latest[(item["decision_id"], model_id)].get("model_action_index"),
                    int,
                )
            ]
            agree = sum(row.get("agreement", False) for row in rows)
            return {
                "eligible": len(items),
                "valid": len(rows),
                "agreements": agree,
                "agreement_rate_valid": _percent(agree, len(rows)),
                "strict_agreement_rate": _percent(agree, len(items)),
            }

        summary_models[model_id] = {
            "eligible_decisions": eligible,
            "responses_present": len(present_rows),
            "valid_selections": len(valid_rows),
            "coverage": _percent(len(valid_rows), eligible),
            "api_errors": api_errors,
            "parse_errors": parse_errors,
            "agreements": agreements,
            "agreement_rate_valid": _percent(agreements, len(valid_rows)),
            "strict_agreement_rate": _percent(agreements, eligible),
            "forced": subset_stats(forced_items),
            "nontrivial": subset_stats(nontrivial_items),
            "by_action_type": by_type,
            "usage": {
                "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in usage_rows),
                "completion_tokens": sum(
                    int(row.get("completion_tokens", 0)) for row in usage_rows
                ),
                "total_tokens": sum(int(row.get("total_tokens", 0)) for row in usage_rows),
                "cost_usd": sum(float(row.get("cost", 0.0)) for row in usage_rows),
            },
            "latency_ms": {
                "mean": statistics.mean(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies) if latencies else None,
            },
        }

    model_pair = None
    if len(models) == 2:
        left, right = models
        both_valid = 0
        same_selection = 0
        both_human = 0
        left_only_human = 0
        right_only_human = 0
        neither_human = 0
        same_alternative = 0
        for item in comparisons:
            left_result = item["models"][left]
            right_result = item["models"][right]
            if not isinstance(left_result["action_index"], int) or not isinstance(
                right_result["action_index"], int
            ):
                continue
            both_valid += 1
            same_model_choice = (
                left_result["action"] == right_result["action"]
                if left_result["action"] is not None
                and right_result["action"] is not None
                else left_result["action_index"] == right_result["action_index"]
            )
            if same_model_choice:
                same_selection += 1
            left_human = left_result["agreement"]
            right_human = right_result["agreement"]
            if left_human and right_human:
                both_human += 1
            elif left_human:
                left_only_human += 1
            elif right_human:
                right_only_human += 1
            else:
                neither_human += 1
                if same_model_choice:
                    same_alternative += 1
        model_pair = {
            "models": [left, right],
            "both_valid": both_valid,
            "same_selection": same_selection,
            "same_selection_rate": _percent(same_selection, both_valid),
            "both_match_human": both_human,
            "left_only_matches_human": left_only_human,
            "right_only_matches_human": right_only_human,
            "neither_matches_human": neither_human,
            "same_nonhuman_alternative": same_alternative,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_parser_version": COMPARISON_PARSER_VERSION,
        "generated_at": utc_now(),
        "game_id": scan["game_id"],
        "target_player_id": scan["target_player_id"],
        "target_engine_color": scan["target_engine_color"],
        "parsed_action_count": scan["parsed_action_count"],
        "canonicalization_count": len(scan.get("canonicalizations", [])),
        "target_action_records": len(manifest),
        "classification_counts": dict(classification_counts),
        "exact_decisions": len(comparisons),
        "forced_exact_decisions": sum(item["forced"] for item in comparisons),
        "nontrivial_exact_decisions": sum(not item["forced"] for item in comparisons),
        "step_statuses": scan["step_statuses"],
        "semantic_error_count": len(scan["semantic_errors"]),
        "models": summary_models,
        "model_pair": model_pair,
    }


def _rate_text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _short_action(text: Optional[str], limit: int = 96) -> str:
    if not text:
        return "—"
    normalized = " ".join(str(text).split()).replace("|", "\\|")
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def render_report(
    summary: Dict[str, Any],
    comparisons: Sequence[Dict[str, Any]],
    models: Sequence[str],
) -> str:
    lines = [
        "# Full-game model vs human action-selection diff",
        "",
        f"Generated: {summary['generated_at']}",
        f"Game: `{summary['game_id']}`",
        f"Comparison parser: `{summary['comparison_parser_version']}`",
        (
            f"Human: Colonist color {summary['target_player_id']} / "
            f"engine {summary['target_engine_color']}"
        ),
        "Policy calls: stateless; model actions were never executed",
        "Replay stepping: `allow_lookahead=False`",
        "",
        "## Decision coverage",
        "",
        f"- Parsed replay rows: {summary['parsed_action_count']}",
        (
            "- Same-event robber/steal order canonicalizations: "
            f"{summary['canonicalization_count']}"
        ),
        f"- Human-seat records: {summary['target_action_records']}",
        f"- Exact indexed choices: {summary['exact_decisions']}",
        f"- Forced exact choices: {summary['forced_exact_decisions']}",
        f"- Nontrivial exact choices: {summary['nontrivial_exact_decisions']}",
        f"- Other classifications: `{json.dumps(summary['classification_counts'], sort_keys=True)}`",
        f"- Replay semantic errors: {summary['semantic_error_count']}",
        "",
        "## Headline agreement",
        "",
        "| Model | Valid / exact | All exact | Forced | Nontrivial | Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id in models:
        metrics = summary["models"][model_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model_id}`",
                    (
                        f"{metrics['agreements']}/{metrics['valid_selections']} "
                        f"({_rate_text(metrics['agreement_rate_valid'])})"
                    ),
                    (
                        f"{metrics['agreements']}/{metrics['eligible_decisions']} "
                        f"({_rate_text(metrics['strict_agreement_rate'])})"
                    ),
                    (
                        f"{metrics['forced']['agreements']}/{metrics['forced']['valid']} "
                        f"({_rate_text(metrics['forced']['agreement_rate_valid'])})"
                    ),
                    (
                        f"{metrics['nontrivial']['agreements']}/{metrics['nontrivial']['valid']} "
                        f"({_rate_text(metrics['nontrivial']['agreement_rate_valid'])})"
                    ),
                    f"${metrics['usage']['cost_usd']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Output and usage health",
            "",
            "| Model | Responses | Valid actions | Format warnings | API errors | Tokens in/out | Median / p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_id in models:
        metrics = summary["models"][model_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model_id}`",
                    str(metrics["responses_present"]),
                    str(metrics["valid_selections"]),
                    str(metrics["parse_errors"]),
                    str(metrics["api_errors"]),
                    (
                        f"{metrics['usage']['prompt_tokens']:,} / "
                        f"{metrics['usage']['completion_tokens']:,}"
                    ),
                    (
                        "n/a"
                        if metrics["latency_ms"]["median"] is None
                        else (
                            f"{metrics['latency_ms']['median'] / 1000:.2f}s / "
                            f"{metrics['latency_ms']['p95'] / 1000:.2f}s"
                        )
                    ),
                ]
            )
            + " |"
        )

    pair = summary.get("model_pair")
    if pair:
        lines.extend(
            [
                "",
                "## Model-to-model diff",
                "",
                (
                    f"- Same selection: {pair['same_selection']}/{pair['both_valid']} "
                    f"({_rate_text(pair['same_selection_rate'])})"
                ),
                f"- Both match human: {pair['both_match_human']}",
                f"- Only `{pair['models'][0]}` matches human: {pair['left_only_matches_human']}",
                f"- Only `{pair['models'][1]}` matches human: {pair['right_only_matches_human']}",
                f"- Neither matches human: {pair['neither_matches_human']}",
                f"- Both choose the same nonhuman alternative: {pair['same_nonhuman_alternative']}",
            ]
        )

    action_types = sorted(
        {
            action_type
            for model_id in models
            for action_type in summary["models"][model_id]["by_action_type"]
        }
    )
    lines.extend(
        [
            "",
            "## Agreement by human action type",
            "",
            "| Human action | " + " | ".join(f"`{model}`" for model in models) + " |",
            "| --- | " + " | ".join("---:" for _ in models) + " |",
        ]
    )
    for action_type in action_types:
        cells = []
        for model_id in models:
            metrics = summary["models"][model_id]["by_action_type"][action_type]
            cells.append(
                f"{metrics['agreements']}/{metrics['valid']} "
                f"({_rate_text(metrics['agreement_rate_valid'])})"
            )
        lines.append(f"| `{action_type}` | " + " | ".join(cells) + " |")

    differing = [
        item
        for item in comparisons
        if any(
            not item["models"][model_id]["agreement"] for model_id in models
        )
    ]
    lines.extend(
        [
            "",
            "## Decisions with at least one model-human difference",
            "",
            (
                "| Replay row | Human type | Menu | Human | "
                + " | ".join(f"`{model}`" for model in models)
                + " |"
            ),
            "| ---: | --- | ---: | --- | " + " | ".join("---" for _ in models) + " |",
        ]
    )
    for item in differing:
        model_cells = []
        for model_id in models:
            result = item["models"][model_id]
            marker = "✓" if result["agreement"] else "✗"
            model_cells.append(
                f"{marker} {result['action_index']}: {_short_action(result['description'])}"
            )
        lines.append(
            f"| {item['replay_index']} | `{item['effective_action_type']}` | "
            f"{item['menu_size']} | {item['human']['action_index']}: "
            f"{_short_action(item['human']['description'])} | "
            + " | ".join(model_cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `plan.json`: immutable run inputs and packet settings",
            "- `decision_manifest.jsonl`: every classified action for the human seat",
            "- `responses.jsonl`: append-only raw model calls and prompts",
            "- `comparisons.jsonl`: one normalized human/model comparison per exact choice",
            "- `summary.json`: machine-readable aggregate metrics",
            "",
        ]
    )
    return "\n".join(lines)


def run_action_diff(
    game_id: str,
    models: Sequence[str],
    output_dir: Path,
    target_player: str = "captured",
    dry_run: bool = False,
    reasoning_effort: str = "xhigh",
    max_tokens: int = 8_192,
    max_requests_per_model: Optional[int] = None,
    max_cost_usd: Optional[float] = None,
    retry_errors: bool = False,
) -> Dict[str, Any]:
    """Preflight a full replay, optionally query models, and write artifacts."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive")
    reasoning = native_reasoning_request(reasoning_effort)
    suite = load_context_suite()
    bucket_suite = load_decision_bucket_suite()
    output_dir.mkdir(parents=True, exist_ok=True)
    scan = scan_replay(game_id, target_player)
    if scan["semantic_errors"]:
        _write_json(output_dir / "preflight_errors.json", scan["semantic_errors"])
        raise RuntimeError(
            f"Preflight found {len(scan['semantic_errors'])} replay semantic errors"
        )

    manifest = scan["records"]
    _write_jsonl(output_dir / "decision_manifest.jsonl", manifest)
    manifest_hash = hashlib.sha256(
        (output_dir / "decision_manifest.jsonl").read_bytes()
    ).hexdigest()
    manifest_semantic_hash = semantic_manifest_hash(manifest)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "comparison_parser_version": COMPARISON_PARSER_VERSION,
        "generated_at": utc_now(),
        "game_id": game_id,
        "models": list(models),
        "target_player": target_player,
        "target_player_id": scan["target_player_id"],
        "target_engine_color": scan["target_engine_color"],
        "parsed_action_count": scan["parsed_action_count"],
        "canonicalizations": scan.get("canonicalizations", []),
        "exact_decision_count": sum(
            row.get("classification") == "exact" for row in manifest
        ),
        "manifest_sha256": manifest_hash,
        "manifest_semantic_sha256": manifest_semantic_hash,
        "settings": {
            "context_version": f"{suite.id}@{suite.version}",
            "decision_bucket_suite": f"{bucket_suite.id}@{bucket_suite.version}",
            "stateless_game_plan": True,
            "allow_lookahead": False,
            "execute_model_actions": False,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "max_cost_usd": max_cost_usd,
            "native_reasoning_request": {
                model_id: dict(reasoning)
                for model_id in models
            },
            "headline_includes_forced_exact_choices": True,
            "coarse_actions_excluded": True,
            "same_event_robber_order": "MOVE_ROBBER then STEAL",
        },
    }
    plan_path = output_dir / "plan.json"
    responses_path = output_dir / "responses.jsonl"
    if responses_path.exists() and plan_path.exists():
        existing_plan = json.loads(plan_path.read_text())
        existing_responses = _read_jsonl(responses_path)
        validate_response_compatibility(manifest, existing_responses)
        invariant_fields = (
            "schema_version",
            "game_id",
            "models",
            "target_player",
            "target_player_id",
            "target_engine_color",
            "settings",
        )
        mismatches = [
            field
            for field in invariant_fields
            if existing_plan.get(field) != plan.get(field)
        ]
        existing_semantic_hash = existing_plan.get("manifest_semantic_sha256")
        if (
            existing_semantic_hash is not None
            and existing_semantic_hash != manifest_semantic_hash
        ):
            mismatches.append("manifest_semantic_sha256")
        if mismatches:
            raise RuntimeError(
                "Cannot resume response artifact after run inputs changed: "
                + ", ".join(mismatches)
            )
    _write_json(plan_path, plan)

    new_requests = {}
    if not dry_run:
        new_requests = query_replay(
            game_id=game_id,
            target_player=target_player,
            manifest=manifest,
            models=models,
            responses_path=responses_path,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            max_requests_per_model=max_requests_per_model,
            max_cost_usd=max_cost_usd,
            retry_errors=retry_errors,
        )

    response_rows = _read_jsonl(responses_path)
    comparisons = build_comparisons(manifest, response_rows, models)
    summary = summarize_run(scan, response_rows, models)
    summary["new_requests_this_invocation"] = new_requests
    summary["complete"] = all(
        summary["models"][model_id]["responses_present"]
        == summary["exact_decisions"]
        and summary["models"][model_id]["api_errors"] == 0
        for model_id in models
    )
    _write_jsonl(output_dir / "comparisons.jsonl", comparisons)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        render_report(summary, comparisons, models)
    )
    return summary
