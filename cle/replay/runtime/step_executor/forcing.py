"""Force Colonist facts onto engine state the typed rules cannot reach."""

from __future__ import annotations

from collections.abc import Sequence

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.decks import ROAD_COST_FREQDECK, freqdeck_add
from cle.game_engine.models.enums import ROAD, ActionPrompt
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState
from cle.game_engine.state_functions import maintain_longest_road
from cle.game_engine.trading import TradeOfferStatus
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ReplayRuntimeState
from cle.replay.runtime.audit import record_replay_issue

from .context import (
    TURN_OWNER_ACTIONS,
    engine_color_for_colonist,
    engine_of,
    record_forced_overlay,
    regenerate_playable_actions,
    seat_index,
    seating_map,
)
from .offers import ensure_counter_offer, ensure_root_offer, project_trade_record
from .publishing import recorded_trade_payload

__all__ = [
    "apply_trade_closures",
    "force_apply_trade_overlay",
    "force_clear_stale_steal_prompt",
    "force_record_road",
    "sync_turn_owner_from_hint",
]


def force_apply_trade_overlay(
    action_type: str,
    action_hint: ActionHint,
    state: ReplayRuntimeState,
) -> str | None:
    """Apply one authoritative Colonist trade event to the typed offer board."""
    game_state = engine_of(state).state
    player, _ = engine_color_for_colonist(state, action_hint.get("player"))
    offered_by, _ = engine_color_for_colonist(
        state,
        action_hint.get("creator"),
    )

    if action_type in {"OFFER_TRADE", "COUNTER_OFFER"}:
        if player is None:
            return None
        if action_type == "OFFER_TRADE":
            ensure_root_offer(game_state, player, action_hint)
        else:
            ensure_counter_offer(game_state, player, action_hint)
        record = state.replay_trade_ledger.get(action_hint.get("trade_id"))
        if record is not None:
            project_trade_record(state, record)
        regenerate_playable_actions(game_state)
        return f"Applied {action_type} to typed trade window"

    if action_type not in {
        "ACCEPT_TRADE",
        "REJECT_TRADE",
        "CLEAR_TRADE_RESPONSE",
    }:
        return None
    if player is None or offered_by is None:
        return None

    offer = (
        ensure_counter_offer(game_state, offered_by, action_hint)
        if action_hint.get("is_counter_offer")
        else ensure_root_offer(game_state, offered_by, action_hint)
    )
    if offer is None:
        return f"Recorded source-only {action_type} without a materialized offer"
    if player not in offer.audience:
        return f"Recorded source-only {action_type} outside the offer audience"
    if action_type == "ACCEPT_TRADE":
        offer.declined_by.discard(player)
        offer.willing_by.add(player)
    elif action_type == "REJECT_TRADE":
        offer.willing_by.discard(player)
        offer.declined_by.add(player)
    else:
        offer.willing_by.discard(player)
        offer.declined_by.discard(player)
    regenerate_playable_actions(game_state)
    return f"Applied {action_type} to typed trade window"


def _trade_closures_from_hint(action_hint: ActionHint) -> list[ActionHint]:
    if action_hint.get("trade_closures_preceded"):
        return []
    closures = action_hint.get("closed_trades")
    if closures is not None:
        return closures
    if action_hint.get("type") == "CLOSE_TRADE":
        return [action_hint]
    return []


def apply_trade_closures(
    action_hint: ActionHint,
    state: ReplayRuntimeState,
) -> list[dict[str, object]]:
    """Apply authoritative Colonist offer closures to the typed window."""
    game = engine_of(state)
    game_state = game.state
    window = game_state.trade_window
    applied: list[dict[str, object]] = []

    for closure in _trade_closures_from_hint(action_hint):
        trade_id = closure.get("trade_id")
        key = trade_id if isinstance(trade_id, str) else None
        offer = window.offers.get(key) if window is not None and key is not None else None
        existed = offer is not None and offer.active
        if existed and offer is not None:
            offer.status = TradeOfferStatus.WITHDRAWN
        creator, _ = engine_color_for_colonist(state, closure.get("creator"))
        if creator is None and offer is not None:
            creator = offer.offered_by
        if creator is not None:
            terms = recorded_trade_payload(game, trade_id)
            game.publish_event(
                "CLOSE_TRADE",
                creator,
                {
                    "offer_id": trade_id,
                    "parent_offer_id": closure.get("counter_offer_to"),
                    "reason": closure.get("reason"),
                    **({"offer": terms} if terms is not None else {}),
                },
                causation_id=f"replay:{state.replay_index}:event:{action_hint.get('index')}",
            )
        else:
            record_replay_issue(
                state,
                kind="unmapped_trade_closure",
                action_hint=action_hint,
                message=f"Could not map creator of closed trade {trade_id}",
                severity="warning",
            )
        applied.append({
            "trade_id": trade_id,
            "creator": closure.get("creator"),
            "is_counter_offer": bool(closure.get("is_counter_offer")),
            "reason": closure.get("reason"),
            "offer_existed": existed,
        })

    if applied:
        regenerate_playable_actions(game_state)
    return applied


def force_clear_stale_steal_prompt(
    state: ReplayRuntimeState,
    action_hint: ActionHint,
    reason: str,
) -> None:
    game_state = engine_of(state).state
    game_state.current_player_index = game_state.current_turn_index
    game_state.current_prompt = ActionPrompt.PLAY_TURN
    game_state.is_moving_knight = False
    regenerate_playable_actions(game_state)
    record_forced_overlay(
        state,
        action_hint,
        "Cleared stale STEAL prompt from replay state",
        details={"reason": reason},
    )


def force_record_road(
    game_state: GameState,
    player_color: Color,
    location: Sequence[int],
    is_free: bool,
) -> None:
    """Record a Colonist road when engine topology rejects the placement."""
    edge = (location[0], location[1])
    inverted_edge = (edge[1], edge[0])
    game_state.board.roads[edge] = player_color
    game_state.board.roads[inverted_edge] = player_color
    game_state.board.buildable_edges_cache = {}

    maintain_longest_road(game_state, *game_state.board.recompute_road_state())

    if edge not in game_state.buildings_by_color[player_color][ROAD]:
        game_state.buildings_by_color[player_color][ROAD].append(edge)

    key = f"P{game_state.color_to_index[player_color]}"
    game_state.player_state[f"{key}_ROADS_AVAILABLE"] -= 1
    if is_free:
        if game_state.is_road_building and game_state.free_roads_available > 0:
            game_state.free_roads_available -= 1
            if game_state.free_roads_available == 0:
                game_state.is_road_building = False
    else:
        game_state.player_state[f"{key}_WOOD_IN_HAND"] -= 1
        game_state.player_state[f"{key}_BRICK_IN_HAND"] -= 1
        game_state.resource_freqdeck = freqdeck_add(
            game_state.resource_freqdeck,
            ROAD_COST_FREQDECK,
        )

    game_state.current_prompt = ActionPrompt.PLAY_TURN
    game_state.playable_actions = generate_playable_actions(game_state)


def sync_turn_owner_from_hint(
    action_hint: ActionHint,
    state: ReplayRuntimeState,
) -> None:
    """Align engine turn indices to the Colonist actor for turn-owned actions."""
    action_type = action_hint.get("type")
    if action_type not in TURN_OWNER_ACTIONS:
        return

    colonist_player = action_hint.get("player")
    if colonist_player is None:
        return

    game = engine_of(state)
    player_idx = seat_index(seating_map(state), colonist_player)
    if player_idx is None or player_idx >= len(game.state.colors):
        return

    if action_type == "DISCARD":
        if game.state.current_player_index != player_idx:
            player_color = game.state.colors[player_idx]
            print(
                f"[Replay] Syncing discard actor: "
                f"colonist {colonist_player} -> {player_color}"
            )
            game.state.current_player_index = player_idx
            game.state.playable_actions = generate_playable_actions(game.state)
        return

    if (
        game.state.current_player_index != player_idx
        or game.state.current_turn_index != player_idx
    ):
        player_color = game.state.colors[player_idx]
        print(
            f"[Replay] Syncing turn owner for {action_type}: "
            f"colonist {colonist_player} -> {player_color}"
        )
        game.state.current_player_index = player_idx
        game.state.current_turn_index = player_idx
        game.state.playable_actions = generate_playable_actions(game.state)
