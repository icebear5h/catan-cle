"""Production counting and contract fact validation."""
import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import JsonValue, snapshot_public_board
from evals.catan_board_bench.builder import CatanObservationSuite
from sft.board.spatial_tasks import (
    atlas_node_graph,
    dice_production,
    local_node_tiles,
)

from .support import ZERO


def test_local_tiles_match_both_real_engine_contract_emitters(game: GameEngine, contract: dict[str, JsonValue]) -> None:
    expanded = CatanObservationSuite().public_board_contract(game)
    for token in atlas_node_graph():
        expected = {
            f"<T{tile['id']:02d}>": {
                "resource": tile["resource"].lower() if tile["resource"] is not None else "desert",
                "number": tile["number"],
            }
            for tile in contract["tiles"]
            if int(token[2:-1]) in tile["nodes"]
        }
        actual = local_node_tiles(contract, token)
        assert actual == expected == local_node_tiles(expanded, token)
        assert list(actual) == sorted(actual)
    assert local_node_tiles(contract, "<N03>")["<T03>"] == {"resource": "desert", "number": None}


def test_production_counts_cities_settlements_robber_and_ignores_bank(game: GameEngine) -> None:
    board = game.state.board
    for color, node in ((Color.RED, 0), (Color.RED, 2), (Color.RED, 18), (Color.BLUE, 4)):
        board.build_settlement(color, node, initial_build_phase=True)
    before_city = snapshot_public_board(game.observe(Color.RED)).contract()
    assert dice_production(before_city, "RED", 8) == {**ZERO, "ore": 3}
    board.build_city(Color.RED, 0)
    contract = snapshot_public_board(game.observe(Color.RED)).contract()
    assert dice_production(contract, "RED", 8) == {**ZERO, "ore": 4}
    assert dice_production(contract, "BLUE", 8) == {**ZERO, "ore": 1}
    assert dice_production(contract, "RED", 11) == {**ZERO, "sheep": 2}
    assert dice_production(contract, "RED", 7) == ZERO
    assert dice_production(contract, "WHITE", 8) == ZERO
    assert dice_production(contract, "RED", 2) == ZERO
    for roll in range(2, 13):
        assert dice_production(contract, "RED", roll) == dice_production(
            CatanObservationSuite().public_board_contract(game),
            "RED",
            roll,
        )
    game.state.resource_freqdeck = [0] * 5
    empty_bank = snapshot_public_board(game.observe(Color.RED)).contract()
    assert dice_production(empty_bank, "RED", 8) == {**ZERO, "ore": 4}
    board.robber_coordinate = next(
        coord for coord, tile in board.map.land_tiles.items() if tile.id == 0
    )
    blocked = snapshot_public_board(game.observe(Color.RED)).contract()
    assert dice_production(blocked, "RED", 8) == {**ZERO, "ore": 1}
    assert dice_production(blocked, "BLUE", 8) == ZERO


@pytest.mark.parametrize(
    "color,roll",
    [
        ("RED", 1),
        ("RED", 13),
        ("RED", True),
        ("RED", 8.0),
        ("RED", "8"),
        ("red", 8),
        ("<RED>", 8),
        ("PURPLE", 8),
        ("GREEN", 8),
        (None, 8),
        ([], 8),
    ],
)
def test_production_rejects_invalid_targets_and_absent_players(
    contract: dict[str, JsonValue], color: object, roll: int | float | str
) -> None:
    with pytest.raises(ValueError):
        dice_production(contract, color, roll)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema",), "public_board_contract/v0"),
        (("tiles", 0, "resource"), "wood"),
        (("tiles", 0, "resource"), "GOLD"),
        (("tiles", 0, "number"), True),
        (("tiles", 0, "number"), 8.0),
        (("tiles", 0, "number"), 7),
        (("tiles", 0, "number"), None),
        (("tiles", 3, "number"), 8),
        (("tiles", 0, "id"), True),
        (("tiles", 0, "id"), 1),
        (("tiles", 0, "token"), "<T01>"),
        (("tiles", 0, "nodes"), [0, 1, 2, 3, 4, 4]),
        (("tiles", 0, "coord"), [1, 0, -1]),
        (("tiles", 0, "has_robber"), 1),
        (("tiles", 0, "has_robber"), True),
        (("robber", "tile_id"), 0),
        (("robber", "tile_id"), True),
        (("robber", "coord"), [0, 0, 0]),
        (("robber", "tile_token"), "<T00>"),
        (("tiles",), []),
        (("robber",), None),
    ],
)
def test_invalid_contract_tile_and_robber_facts_raise(
    contract: dict[str, JsonValue], path: tuple[str | int, ...], value: object
) -> None:
    container = contract
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    for call in (
        lambda: local_node_tiles(contract, "<N00>"),
        lambda: dice_production(contract, "RED", 8),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize(
    "path,value",
    [
        (("nodes", 0, "building"), "ROAD"),
        (("nodes", 0, "color"), "RED"),
        (("nodes", 0, "building"), "CITY"),
        (("nodes", 0, "color"), "PURPLE"),
        (("nodes", 0, "adjacent_tiles"), [0, 5, 5]),
        (("nodes", 0, "adjacent_tiles"), [False, 5, 6]),
        (("nodes", 0, "id"), 1),
        (("nodes", 0, "id"), 0.0),
        (("nodes", 0, "token"), "<N01>"),
        (("players", 0, "color"), "ORANGE"),
        (("players", 0, "color"), "red"),
        (("nodes",), []),
        (("players",), []),
    ],
)
def test_invalid_contract_piece_and_player_facts_raise_even_on_seven(
    contract: dict[str, JsonValue], path: tuple[str | int, ...], value: object
) -> None:
    container = contract
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    with pytest.raises(ValueError):
        dice_production(contract, "RED", 7)
