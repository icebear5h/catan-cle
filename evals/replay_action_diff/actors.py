"""Actor inference, policy action ordering, and response menus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState
from cle.replay.contracts import ParsedActions, ReplayRuntimeState, mapping_field
from cle.replay.runtime.step_executor import _ensure_root_offer
from evals.replay_action_diff.contracts import (
    ASYNC_TRADE_RESPONSES,
    COMPOUND_ACTIONS,
    require_archive,
    require_engine,
)
from evals.replay_action_diff.identity import _action_type_name


def infer_actor(
    state: ReplayRuntimeState, action_hint: Mapping[str, object]
) -> tuple[int | None, str]:
    """Resolve the acting engine seat without guessing from turn order."""
    player_id = action_hint.get("player")
    mapping = mapping_field(require_archive(state), "colonist_color_to_engine_idx")
    player_index = mapping.get(str(player_id))
    if isinstance(player_index, int):
        return player_index, "replay_player"

    if action_hint.get("type") != "BUILD_CITY":
        return None, "unresolved"

    corner = action_hint.get("colonist_corner")
    node = state.corner_to_node_map.get(f"_{corner}")
    building = (
        require_engine(state).state.board.buildings.get(node) if node is not None else None
    )
    if not building:
        return None, "unresolved"
    owner_color = building[0]
    inferred_index = require_engine(state).state.color_to_index.get(owner_color)
    if not isinstance(inferred_index, int):
        return None, "unresolved"
    return inferred_index, "pre_action_building_owner"


def canonicalize_policy_action_order(
    parsed_actions: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Put same-event robber movement before its resulting steal decision."""
    canonical: list[dict[str, object]] = []
    for source_index, action in enumerate(parsed_actions):
        copied = dict(deepcopy(action))
        copied["_source_replay_index"] = source_index
        canonical.append(copied)

    changes: list[dict[str, object]] = []
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
    parsed_actions: ParsedActions | Sequence[Mapping[str, object]], replay_index: int
) -> tuple[Mapping[str, object], int | None, str | None]:
    source: Mapping[str, object] = parsed_actions[replay_index]
    source_type = source.get("type")
    expected_followup = (
        COMPOUND_ACTIONS.get(source_type) if isinstance(source_type, str) else None
    )
    if expected_followup is None:
        return source, None, None
    if replay_index + 1 >= len(parsed_actions):
        return source, None, "missing compound follow-up"

    followup: Mapping[str, object] = parsed_actions[replay_index + 1]
    if (
        followup.get("type") != expected_followup
        or followup.get("player") != source.get("player")
    ):
        return source, None, f"expected adjacent {expected_followup} follow-up"
    return followup, replay_index + 1, None


def _response_menu(
    game_state: GameState,
    action_hint: Mapping[str, object],
    creator_color: Color,
) -> list[Action]:
    """Return only the actions scoped to one asynchronous trade response."""
    _ensure_root_offer(game_state, creator_color, action_hint)
    generated = generate_playable_actions(game_state)
    scoped: list[Action] = []
    counter_offer_added = False
    for action in generated:
        action_type = _action_type_name(action)
        if action_type in ASYNC_TRADE_RESPONSES and action.value == creator_color:
            scoped.append(action)
        elif action_type == "COUNTER_OFFER" and not counter_offer_added:
            scoped.append(action)
            counter_offer_added = True
    return scoped

