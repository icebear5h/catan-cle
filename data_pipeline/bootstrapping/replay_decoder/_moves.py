"""Enumerate the moves that are legal in the decoder's current state."""

from data_pipeline.bootstrapping.replay_decoder._enums import (
    ActionState,
    BuildingType,
    DevCardType,
    ResourceType,
)
from data_pipeline.bootstrapping.replay_decoder._models import GameState
from data_pipeline.bootstrapping.replay_decoder._placement import (
    can_afford_city,
    can_afford_dev_card,
    can_afford_road,
    can_afford_settlement,
    can_place_road,
    can_place_settlement,
)
from data_pipeline.bootstrapping.replay_decoder._types import Move


def valid_moves(state: GameState) -> list[Move]:
    """Get all valid moves for current game state"""
    moves: list[Move] = []
    player = state.players.get(state.current_player)
    if not player:
        return moves

    action_state = state.action_state

    # Setup phase - place settlement
    if action_state == ActionState.SETUP_PLACE_SETTLEMENT:
        for cid in state.corners:
            if can_place_settlement(state, cid, is_setup=True):
                moves.append({'type': 'build_settlement', 'corner': cid})

    # Setup phase - place road
    elif action_state == ActionState.SETUP_PLACE_ROAD:
        for eid in state.edges:
            if can_place_road(state, eid):
                moves.append({'type': 'build_road', 'edge': eid})

    # Must roll dice
    elif action_state == ActionState.MUST_ROLL_DICE:
        moves.append({'type': 'roll_dice'})
        # Can also play knight before rolling
        if DevCardType.KNIGHT in player.dev_cards:
            moves.append({'type': 'play_knight'})

    # Must move robber
    elif action_state == ActionState.MUST_MOVE_ROBBER:
        for tid in state.tiles:
            if tid != state.robber_tile:
                moves.append({'type': 'move_robber', 'tile': tid})

    # Main turn
    elif action_state == ActionState.MAIN_TURN:
        moves.append({'type': 'end_turn'})

        # Build settlement
        if can_afford_settlement(player):
            for cid in state.corners:
                if can_place_settlement(state, cid):
                    moves.append({'type': 'build_settlement', 'corner': cid})

        # Build city
        if can_afford_city(player):
            for cid, corner in state.corners.items():
                if corner.owner == player.color and corner.building_type == BuildingType.SETTLEMENT:
                    moves.append({'type': 'build_city', 'corner': cid})

        # Build road
        if can_afford_road(player):
            for eid in state.edges:
                if can_place_road(state, eid):
                    moves.append({'type': 'build_road', 'edge': eid})

        # Buy dev card
        if can_afford_dev_card(player) and len(state.bank_dev_cards) > 0:
            moves.append({'type': 'buy_dev_card'})

        # Play dev cards
        for card in set(player.dev_cards):
            if card not in player.dev_cards_used:
                moves.append({'type': 'play_dev_card', 'card': card})

        # Bank trades (simplified - just check if can trade 4:1 or better)
        for res in ResourceType:
            if res == ResourceType.ANY:
                continue
            ratio = player.bank_trade_ratios.get(res, 4)
            if player.resource_count(res) >= ratio:
                for target in ResourceType:
                    if target != res and target != ResourceType.ANY:
                        if state.bank_resources.get(target, 0) > 0:
                            moves.append({'type': 'bank_trade', 'give': res,
                                          'get': target, 'ratio': ratio})

    return moves


__all__ = ["valid_moves"]
