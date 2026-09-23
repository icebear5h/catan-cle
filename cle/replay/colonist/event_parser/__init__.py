"""Parse Colonist events into action type hints for the engine.

Colonist archives are delta encoded and untrusted, so every nested field is
read through the archive narrowing helpers. A fragment whose shape is wrong
(a mapping where a list belongs, say) now reads as empty instead of raising
``AttributeError`` part way through the event; well-formed archives decode
exactly as before.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from cle.replay.colonist.constants import COLONIST_DEV_CARD
from cle.replay.colonist.types import ActionHint, ColonistHands, ColonistTileState
from cle.replay.contracts import (
    as_mapping,
    mapping_field,
    optional_list_field,
)

from .game_log import decode_game_log
from .resources import ResourceTracker
from .trades import TradeTracker

__all__ = ["parse_colonist_events_to_actions"]


def parse_colonist_events_to_actions(
    events: Sequence[Mapping[str, object]],
    tile_hex_states: Mapping[str, ColonistTileState] | None = None,
    player_ids: Iterable[int | str] | None = None,
) -> list[ActionHint]:
    """Parse Colonist events into action type hints for our engine.

    Note: Colonist uses delta encoding - only changed values are included.
    We track state across events to get complete dice values and trade states.
    Also tracks Colonist's resource state per player (source of truth for syncing).

    Each action dict includes 'expected_resources': {player_id: [card_ids...]}
    representing the expected state AFTER that action executes.

    Args:
        events: List of Colonist events
        tile_hex_states: Optional dict mapping tile_index -> {type, diceNumber, x, y}
            for matching MOVE_ROBBER actions
        player_ids: Optional authoritative player-color IDs. Required before a
            roll payout can be marked complete.
    """
    if tile_hex_states is None:
        tile_hex_states = {}
    resources = ResourceTracker(player_ids)
    trades = TradeTracker()
    actions: list[ActionHint] = []
    # Track last known dice values (Colonist sends deltas)
    last_dice = [1, 1]  # Default to (1, 1)

    # Track current player to detect turn changes and emit END_TURN
    last_player: object = None
    # Track action state to only emit END_TURN during main game (not setup)
    # actionState 0 = main turn, 1/3 = setup, 24 = must roll
    is_main_game = False

    for i, event in enumerate(events):
        # Update resource tracking from playerStates FIRST (before detecting actions)
        # This gives us the expected state AFTER this event's action executes
        state_change = mapping_field(event, "stateChange")
        player_states = mapping_field(state_change, "playerStates")
        resources_before_event = resources.snapshot()

        # Detect turn changes and emit END_TURN (only in main game)
        current_state = mapping_field(state_change, "currentState")
        action_state = current_state.get("actionState")
        if action_state is not None:
            # Main game starts when we see actionState 24 (must roll) or 0 (main turn)
            if action_state in (0, 24):
                is_main_game = True

        current_player = current_state.get("currentTurnPlayerColor")
        if is_main_game and current_player is not None and last_player is not None and current_player != last_player:
            actions.append({
                "index": i,
                "type": "END_TURN",
                "player": last_player,
                "expected_resources": resources.snapshot(),
            })
        if current_player is not None:
            last_player = current_player
        resources.absorb(player_states)

        actions.extend(_roll_actions(i, state_change, current_state, last_dice, resources_before_event, resources))
        actions.extend(
            _building_actions(i, mapping_field(state_change, "mapState"), resources)
        )

        # Trade events
        trade_state = mapping_field(state_change, "tradeState")
        offer_updates = mapping_field(trade_state, "activeOffers")
        game_log_state = mapping_field(state_change, "gameLogState")
        game_log_types = {
            _log_text(log_entry).get("type")
            for log_entry in game_log_state.values()
        }
        if 115 in game_log_types or 116 in game_log_types:
            closure_reason = "transaction_closed"
        else:
            closure_reason = "cancelled"

        trade_actions, closed_trades = trades.decode(
            i, offer_updates, closure_reason, resources
        )
        actions.extend(trade_actions)
        actions.extend(decode_game_log(
            i,
            game_log_state,
            resources_before_event,
            bool(closed_trades),
            trades.counter,
            resources,
        ))
        actions.extend(_dev_card_purchases(i, state_change, resources))
        actions.extend(_robber_actions(i, state_change, tile_hex_states, last_player, resources))

    return actions


def _log_text(log_entry: object) -> Mapping[str, object]:
    """Read one game-log entry's text fragment."""
    return mapping_field(
        as_mapping(log_entry, "gameLogState[]"), "text", "gameLogState[].text"
    )


def _roll_actions(
    index: int,
    state_change: Mapping[str, object],
    current_state: Mapping[str, object],
    last_dice: list[int],
    resources_before_event: ColonistHands,
    resources: ResourceTracker,
) -> list[ActionHint]:
    """Emit the ROLL row, merging Colonist's delta-encoded dice values."""
    dice_state = mapping_field(state_change, "diceState")
    if not dice_state.get("diceThrown"):
        return []

    # Update only the dice that changed (delta encoding)
    dice1 = dice_state.get("dice1")
    if isinstance(dice1, int):
        last_dice[0] = dice1
    dice2 = dice_state.get("dice2")
    if isinstance(dice2, int):
        last_dice[1] = dice2

    roll_player = current_state.get("currentTurnPlayerColor")
    if roll_player is None:
        for log_entry in mapping_field(state_change, "gameLogState").values():
            entry = as_mapping(log_entry, "gameLogState[]")
            text = mapping_field(entry, "text", "gameLogState[].text")
            if text.get("type") == 10:
                roll_player = text.get("playerColor", entry.get("from"))
                break

    resources_after_event = resources.snapshot()
    resource_payouts, resource_payouts_complete = resources.infer_roll_resource_payouts(
        resources_before_event,
        resources_after_event,
        sum(last_dice),
    )
    return [{
        "index": index,
        "type": "ROLL",
        "player": roll_player,
        "dice": tuple(last_dice),
        "resource_payouts": resource_payouts,
        "resource_payouts_complete": resource_payouts_complete,
        "expected_resources": resources_after_event,
    }]


def _building_actions(
    index: int,
    map_state: Mapping[str, object],
    resources: ResourceTracker,
) -> list[ActionHint]:
    """Emit at most one settlement/city row and one road row for this event."""
    actions: list[ActionHint] = []

    # Building placement (settlement/city)
    corners = mapping_field(map_state, "tileCornerStates")
    for corner_id, corner in corners.items():
        data = as_mapping(corner, f"mapState.tileCornerStates.{corner_id}")
        btype = data.get("buildingType", 0)
        actions.append({
            "index": index,
            "type": "BUILD_SETTLEMENT" if btype == 1 else "BUILD_CITY",
            "player": data.get("owner"),
            "colonist_corner": int(corner_id),
            "expected_resources": resources.snapshot(),
        })
        break

    # Road placement
    edges = mapping_field(map_state, "tileEdgeStates")
    for edge_id, edge in edges.items():
        data = as_mapping(edge, f"mapState.tileEdgeStates.{edge_id}")
        actions.append({
            "index": index,
            "type": "BUILD_ROAD",
            "player": data.get("owner"),
            "colonist_edge": int(edge_id),
            "expected_resources": resources.snapshot(),
        })
        break

    return actions


def _dev_card_purchases(
    index: int,
    state_change: Mapping[str, object],
    resources: ResourceTracker,
) -> list[ActionHint]:
    """Emit a BUY_DEVELOPMENT_CARD row per player who drew this event."""
    actions: list[ActionHint] = []
    dev_state = mapping_field(state_change, "mechanicDevelopmentCardsState")
    for color_str, player_dev in mapping_field(dev_state, "players").items():
        player = as_mapping(
            player_dev, f"mechanicDevelopmentCardsState.players.{color_str}"
        )
        bought = player.get("developmentCardsBoughtThisTurn")
        bought_cards = optional_list_field(
            player, "developmentCardsBoughtThisTurn"
        )
        if bought_cards:
            card_enum = bought_cards[-1]
            card_type = (
                COLONIST_DEV_CARD.get(card_enum, "UNKNOWN")
                if isinstance(card_enum, int)
                else "UNKNOWN"
            )
            print(f"[Parse] BUY_DEVELOPMENT_CARD: player={color_str}, bought={bought}, card_enum={card_enum}, card_type={card_type}")
            actions.append({
                "index": index,
                "type": "BUY_DEVELOPMENT_CARD",
                "player": int(color_str),
                "card_type": card_type,
                "expected_resources": resources.snapshot(),
            })
    return actions


def _robber_actions(
    index: int,
    state_change: Mapping[str, object],
    tile_hex_states: Mapping[str, ColonistTileState],
    last_player: object,
    resources: ResourceTracker,
) -> list[ActionHint]:
    """Emit MOVE_ROBBER from mechanicRobberState, not from game log types."""
    robber_state = mapping_field(state_change, "mechanicRobberState")
    tile_index = robber_state.get("locationTileIndex")
    if tile_index is None:
        return []

    tile_hex = tile_hex_states.get(str(tile_index), {})
    tile_info: dict[str, object] = {
        "resourceType": tile_hex.get("type"),
        "diceNumber": tile_hex.get("diceNumber"),
        "x": tile_hex.get("x"),
        "y": tile_hex.get("y"),
    }
    current_state = mapping_field(state_change, "currentState")
    player = current_state.get("currentTurnPlayerColor", last_player)

    print(f"[Parse] MOVE_ROBBER: tile_index={tile_index}, tile_info={tile_info}")
    return [{
        "index": index,
        "type": "MOVE_ROBBER",
        "tile_info": tile_info,
        "tile_index": tile_index,
        "player": player,
        "expected_resources": resources.snapshot(),
    }]
