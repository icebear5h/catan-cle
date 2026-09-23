import random
from collections import Counter
from dataclasses import replace

import pytest

from cle.game_engine.models.enums import BRICK, ORE, SHEEP, WHEAT, WOOD
from cle.game_engine.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    TOURNAMENT_MAP,
    CatanMap,
    LandTile,
    initialize_tiles,
)

STANDARD_TERRAIN_COUNTS = {WOOD: 4, BRICK: 3, SHEEP: 4, WHEAT: 4, ORE: 3, None: 1}


def test_standard_template_has_the_19_hex_terrain_inventory() -> None:
    assert Counter(BASE_MAP_TEMPLATE.tile_resources) == STANDARD_TERRAIN_COUNTS
    assert sum(tile_type is LandTile for tile_type in BASE_MAP_TEMPLATE.topology.values()) == 19


def test_random_boards_preserve_terrain_counts_across_seeds() -> None:
    layouts = set()
    for seed in range(100):
        board = CatanMap.from_template(BASE_MAP_TEMPLATE, rng=random.Random(seed))
        tiles = list(board.land_tiles.values())
        assert Counter(tile.resource for tile in tiles) == STANDARD_TERRAIN_COUNTS
        assert all((tile.number is None) == (tile.resource is None) for tile in tiles)
        assert Counter(tile.number for tile in tiles if tile.resource is not None) == Counter(
            BASE_MAP_TEMPLATE.numbers
        )
        layouts.add(tuple(tile.resource for tile in tiles))
    assert len(layouts) > 1


def test_explicit_terrain_order_is_preserved_without_consuming_the_input() -> None:
    resources = list(reversed(BASE_MAP_TEMPLATE.tile_resources))
    before = resources.copy()

    tiles = initialize_tiles(
        BASE_MAP_TEMPLATE,
        shuffled_tile_resources_param=resources,
        rng=random.Random(7),
    )

    assert resources == before
    assert [tile.resource for tile in tiles.values() if isinstance(tile, LandTile)] == list(
        reversed(before)
    )


@pytest.mark.parametrize("invalid_kind", ["missing", "extra", "wrong_type", "no_desert", "empty"])
def test_explicit_terrain_order_rejects_invalid_inventory(invalid_kind: str) -> None:
    resources = BASE_MAP_TEMPLATE.tile_resources.copy()
    if invalid_kind == "missing":
        resources.pop(0)
    elif invalid_kind == "extra":
        resources.append(None)
    elif invalid_kind == "wrong_type":
        resources[resources.index(WOOD)] = ORE
    elif invalid_kind == "no_desert":
        resources[resources.index(None)] = WOOD
    else:
        resources = []

    with pytest.raises(ValueError, match="permutation.*terrain inventory"):
        initialize_tiles(
            BASE_MAP_TEMPLATE,
            shuffled_tile_resources_param=resources,
            rng=random.Random(7),
        )


@pytest.mark.parametrize("size_delta", [-1, 1])
def test_template_inventory_must_match_the_number_of_land_positions(size_delta: int) -> None:
    resources = BASE_MAP_TEMPLATE.tile_resources.copy()
    if size_delta < 0:
        resources.pop(0)
    else:
        resources.append(WOOD)
    template = replace(BASE_MAP_TEMPLATE, tile_resources=resources)

    with pytest.raises(ValueError, match="exactly one resource per land tile"):
        initialize_tiles(template, rng=random.Random(7))


def test_mini_map_keeps_its_own_inventory() -> None:
    board = CatanMap.from_template(MINI_MAP_TEMPLATE, rng=random.Random(7))
    assert len(board.land_tiles) == 7
    assert Counter(tile.resource for tile in board.land_tiles.values()) == Counter(
        MINI_MAP_TEMPLATE.tile_resources
    )


def test_tournament_map_preserves_its_layout_and_standard_inventory() -> None:
    resources = [tile.resource for tile in TOURNAMENT_MAP.land_tiles.values()]
    assert Counter(resources) == STANDARD_TERRAIN_COUNTS
    assert resources == [
        None,
        ORE,
        WOOD,
        ORE,
        BRICK,
        ORE,
        WHEAT,
        WHEAT,
        SHEEP,
        BRICK,
        SHEEP,
        BRICK,
        WHEAT,
        WOOD,
        WHEAT,
        WOOD,
        SHEEP,
        SHEEP,
        WOOD,
    ]
