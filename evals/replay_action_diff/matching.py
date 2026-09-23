"""Candidate enumeration and matching of the recorded human action."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action
from cle.replay.contracts import ReplayRuntimeState, optional_mapping_field
from cle.replay.runtime.action_matcher import _colonist_xy_to_engine_coord
from cle.replay.runtime.step_executor import TURN_OWNER_ACTIONS
from evals.json_types import JsonDict, as_dict
from evals.replay_action_diff.actors import _response_menu
from evals.replay_action_diff.contracts import (
    ASYNC_TRADE_RESPONSES,
    COARSE_ACTIONS,
    COMPOUND_ACTIONS,
    LIFECYCLE_ACTIONS,
    HumanMatch,
    require_engine,
)
from evals.replay_action_diff.identity import (
    _action_type_name,
    _engine_color_for_colonist,
    _maritime_value_matches,
    _normalize_edge,
    _same_resource_choice,
)


def prepare_decision_game(
    state: ReplayRuntimeState,
    action_hint: Mapping[str, object],
    actor_index: int,
) -> tuple[GameEngine | None, str | None]:
    """Create an isolated actor-perspective state before the human action."""
    decision_game = require_engine(state).copy()
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
    actions: Sequence[Action],
    action_hint: Mapping[str, object],
    state: ReplayRuntimeState,
) -> tuple[list[int], str]:
    action_type = action_hint.get("type")
    candidates: list[int] = []
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
        node_target = state.corner_to_node_map.get(
            f"_{action_hint.get('colonist_corner')}"
        )
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == action_type and action.value == node_target
        ]
    elif action_type == "BUILD_ROAD":
        edge_target = _normalize_edge(
            state.edge_to_edge_map.get(f"_{action_hint.get('colonist_edge')}")
        )
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "BUILD_ROAD"
            and _normalize_edge(action.value) == edge_target
        ]
    elif action_type == "MOVE_ROBBER":
        tile_info = optional_mapping_field(action_hint, "tile_info")
        x = tile_info.get("x")
        y = tile_info.get("y")
        if x is None or y is None:
            return [], "robber destination is unavailable"
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError(f"robber destination is not integral: {x!r}, {y!r}")
        robber_target = _colonist_xy_to_engine_coord(x, y)
        candidates = [
            index
            for index, action in enumerate(actions)
            if _action_type_name(action) == "MOVE_ROBBER" and action.value == robber_target
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
    actions: Sequence[Action],
    action_hint: Mapping[str, object],
    state: ReplayRuntimeState,
) -> HumanMatch:
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


def _decision_signature(record: JsonDict) -> str:
    payload = {
        "replay_index": record["replay_index"],
        "human_action_index": as_dict(record.get("human", {}), "human").get("action_index"),
        "available_actions": record.get("available_actions", []),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

