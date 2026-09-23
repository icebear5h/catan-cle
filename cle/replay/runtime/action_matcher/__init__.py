"""Find matching engine action from playable actions based on Colonist action hint."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from cle.game_engine.models.enums import Action
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ReplayRuntimeState
from cle.replay.runtime.access import get_game_engine

from .builds import BUILD_MATCHERS
from .context import (
    Decision,
    MatchContext,
)
from .context import (
    colonist_xy_to_engine_coord as _colonist_xy_to_engine_coord,
)
from .context import (
    offer_from_source as _offer_from_source,
)
from .plays import PLAY_MATCHERS
from .trades import TRADE_MATCHERS

__all__ = [
    "_colonist_xy_to_engine_coord",
    "_offer_from_source",
    "find_matching_action",
]

Matcher = Callable[[Action, str, MatchContext], Decision]

MATCHERS: Final[dict[str, Matcher]] = {
    **BUILD_MATCHERS,
    **TRADE_MATCHERS,
    **PLAY_MATCHERS,
}


def find_matching_action(
    playable_actions: Sequence[Action],
    action_hint: ActionHint,
    state: ReplayRuntimeState | None = None,
) -> Action | None:
    """Find an action from playable_actions that matches the replay hint.

    Args:
        playable_actions: List of engine Action objects
        action_hint: Dict with parsed Colonist action data
        state: ServerState instance (needed for mapping lookups and game access)
    """
    action_type = action_hint.get("type")

    if not action_type:
        return None

    # Get target coordinates from mappings
    colonist_corner = action_hint.get("colonist_corner")
    colonist_edge = action_hint.get("colonist_edge")

    target_node: int | None = None
    target_edge: tuple[int, ...] | None = None

    if state is not None:
        if colonist_corner is not None:
            target_node = state.corner_to_node_map.get(f"_{colonist_corner}")
            if target_node is not None:
                print(f"[Mapping] Colonist corner {colonist_corner} -> Engine node {target_node}")

        if colonist_edge is not None:
            edge_tuple = state.edge_to_edge_map.get(f"_{colonist_edge}")
            if edge_tuple:
                target_edge = tuple(edge_tuple)
                print(f"[Mapping] Colonist edge {colonist_edge} -> Engine edge {target_edge}")

    context = MatchContext(
        action_hint=action_hint,
        state=state,
        game=get_game_engine(state) if state else None,
        replay_data=state.replay_data if state else None,
        target_node=target_node,
        target_edge=target_edge,
    )
    matcher = MATCHERS.get(action_type)

    # Find matching action by type and coordinates
    if matcher is not None:
        for action in playable_actions:
            action_str = (
                str(action.action_type.value)
                if hasattr(action, "action_type")
                else str(action)
            )
            decided, matched = matcher(action, action_str, context)
            if decided:
                return matched

    # If no exact match found but we had a target, log it
    if target_node is not None or target_edge is not None:
        print(f"[Mapping] No exact match found for {action_type} with target node={target_node}, edge={target_edge}")
        matching_type_actions = [a for a in playable_actions
                                 if action_type.replace("_", "") in str(a.action_type.value).replace("_", "").upper()]
        if matching_type_actions:
            print(f"[Mapping] Available {action_type} actions: {[a.value for a in matching_type_actions[:5]]}")
        else:
            all_types = set(str(a.action_type).replace("ActionType.", "") for a in playable_actions)
            print(f"[Mapping] No {action_type} available! Playable types: {all_types}")

    return None
