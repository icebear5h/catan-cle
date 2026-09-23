"""Resource shortage resolution during production."""
from typing import Any

import pytest

from cle.game_engine.models.decks import (
    RESOURCE_CARDS_PER_TYPE,
)
from cle.game_engine.models.enums import (
    CITY,
    RESOURCES,
    Action,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import (
    yield_resources,
)
from cle.game_engine.state_functions import get_player_freqdeck

from .support import assert_resource_conservation, make_game, transfer_from_bank_to_player


@pytest.mark.parametrize("supply", [0, 1, 2, 4])
@pytest.mark.parametrize("owners", [(Color.RED,), (Color.RED, Color.RED), (Color.RED, Color.BLUE)])
def test_resource_shortage_depends_on_distinct_entitled_players(supply: int, owners: tuple[Color, ...]) -> None:
    game = make_game()
    state: Any = game.state
    coordinate, tile = next(
        (coordinate, tile)
        for coordinate, tile in state.board.map.land_tiles.items()
        if coordinate != state.board.robber_coordinate
        and tile.resource is not None
        and tile.number is not None
    )
    nodes = tuple(tile.nodes.values())
    for owner, node_id in zip(owners, (nodes[0], nodes[3])):
        state.board.buildings[node_id] = (owner, CITY)
    resource_index = RESOURCES.index(tile.resource)
    cards = [0] * len(RESOURCES)
    cards[resource_index] = RESOURCE_CARDS_PER_TYPE - supply
    transfer_from_bank_to_player(game, Color.ORANGE, cards)
    demand = {
        owner: sum(
            2
            for node, (building_owner, _) in state.board.buildings.items()
            for adjacent in state.board.map.adjacent_tiles[node]
            if building_owner == owner
            and adjacent.resource == tile.resource
            and adjacent.number == tile.number
        )
        for owner in set(owners)
    }
    total_demand = sum(demand.values())
    expected = {
        owner: (
            count if supply >= total_demand else min(count, supply) if len(demand) == 1 else 0
        )
        for owner, count in demand.items()
    }
    before_bank = state.resource_freqdeck.copy()

    payout, depleted_resources = yield_resources(
        state.board,
        state.resource_freqdeck,
        tile.number,
    )

    assert coordinate != state.board.robber_coordinate
    assert (tile.resource in depleted_resources) == (supply < total_demand)
    assert {owner: payout[owner][resource_index] for owner in demand} == expected
    assert state.resource_freqdeck == before_bank

    first_die = max(1, tile.number - 6)
    dice = (first_die, tile.number - first_die)
    game.step(Action(Color.RED, ActionType.ROLL, dice), force=True)

    assert state.resource_freqdeck[resource_index] == supply - sum(expected.values())
    for owner, count in expected.items():
        assert get_player_freqdeck(state, owner)[resource_index] == count
    assert_resource_conservation(game)


def test_shortage_of_one_resource_does_not_suppress_other_resource_production() -> None:
    game = make_game()
    state: Any = game.state
    tiles = [tile for tile in state.board.map.land_tiles.values() if tile.resource is not None]
    first, second = next(
        (first, second)
        for first in tiles
        for second in tiles
        if first.number == second.number and first.resource != second.resource
    )
    first_node = next(node for node in first.nodes.values() if node not in second.nodes.values())
    second_node = next(node for node in second.nodes.values() if node not in first.nodes.values())
    state.board.buildings[first_node] = (Color.RED, CITY)
    state.board.buildings[second_node] = (Color.BLUE, CITY)
    full_payout, _ = yield_resources(state.board, state.resource_freqdeck, first.number)
    first_index = RESOURCES.index(first.resource)
    second_index = RESOURCES.index(second.resource)
    cards = [0] * len(RESOURCES)
    cards[first_index] = RESOURCE_CARDS_PER_TYPE
    transfer_from_bank_to_player(game, Color.ORANGE, cards)
    payout, depleted = yield_resources(state.board, state.resource_freqdeck, first.number)
    assert first.resource in depleted
    assert second.resource not in depleted
    assert full_payout[Color.BLUE][second_index] >= 2
    assert all(cards[first_index] == 0 for cards in payout.values())
    assert {
        color: cards[second_index] for color, cards in payout.items()
    } == {color: cards[second_index] for color, cards in full_payout.items()}
