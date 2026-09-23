"""Name, colour, edge, and resource normalization plus manifest identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone

from cle.game_engine.models.enums import RESOURCES
from cle.game_engine.models.player import Color
from cle.replay.contracts import ReplayRuntimeState, mapping_field
from evals.json_types import JsonDict, as_dicts, as_str
from evals.replay_action_diff.contracts import object_or_empty, require_archive, require_engine


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _color_name(color: object) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _action_type_name(action: object) -> str:
    action_type: object = getattr(action, "action_type", None)
    if hasattr(action_type, "value"):
        return str(action_type.value)
    return str(action_type).replace("ActionType.", "")


def _normalize_edge(value: object) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return min(value), max(value)


def _resource_sort_key(resource: object) -> int:
    for position, candidate in enumerate(RESOURCES):
        if candidate == resource:
            return position
    return len(RESOURCES)


def _same_resource_choice(left: object, right: object) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    return sorted(left, key=_resource_sort_key) == sorted(
        right, key=_resource_sort_key
    )


def _maritime_value_matches(value: object, given: object, received: object) -> bool:
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


def _engine_color_for_colonist(state: ReplayRuntimeState, player_id: object) -> Color | None:
    mapping = mapping_field(require_archive(state), "colonist_color_to_engine_idx")
    player_index = mapping.get(str(player_id))
    colors = require_engine(state).state.colors
    if not isinstance(player_index, int) or not 0 <= player_index < len(colors):
        return None
    return colors[player_index]


def semantic_manifest_hash(manifest: Sequence[JsonDict]) -> str:
    """Hash decision meaning while ignoring unstable legal-menu ordering."""
    payload: list[dict[str, object]] = []
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
                "human_action": object_or_empty(record.get("human"), "human").get("action"),
                "available_actions": sorted(
                    as_str(action.get("action"), "available action")
                    for action in as_dicts(
                        record.get("available_actions", []), "available_actions"
                    )
                ),
            }
        )
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_fingerprint(state: ReplayRuntimeState) -> str:
    game_state = require_engine(state).state
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

