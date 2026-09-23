"""Compatibility contracts captured before splitting the map module."""

import hashlib
import pickle
import random
from collections import Counter
from dataclasses import MISSING, FrozenInstanceError, fields

import pytest

from cle.game_engine.models import map_generation, map_templates, map_types
from cle.game_engine.models.coordinate_system import Direction
from cle.game_engine.models.enums import WOOD, FastResource
from cle.game_engine.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    TOURNAMENT_MAP,
    TOURNAMENT_MAP_TILES,
    CatanMap,
    LandTile,
    MapTemplate,
    Port,
    Water,
    build_map,
    initialize_tiles,
)

# Protocol-0 payload emitted by the original classes, including both frozen records.
LEGACY_RECORDS = (
    b'(ccopy_reg\n_reconstructor\np0\n(ccle.game_engine.models.map\nLandTile\np1\n'
    b'c__builtin__\nobject\np2\nNtp3\nRp4\n(dp5\nVid\np6\nI13\nsVresource\np7\n'
    b'VWOOD\np8\nsVnumber\np9\nI6\nsVnodes\np10\n(dp11\nsVedges\np12\n(dp13\nsb'
    b'g0\n(ccle.game_engine.models.map\nPort\np14\ng2\nNtp15\nRp16\n(dp17\ng6\nI4\n'
    b'sg7\nNsVdirection\np18\nccle.game_engine.models.coordinate_system\nDirection\np19\n'
    b'(VWEST\np20\ntp21\nRp22\nsg10\n(dp23\nsg12\n(dp24\nsbg0\n'
    b'(ccle.game_engine.models.map\nWater\np25\ng2\nNtp26\nRp27\n(dp28\ng10\n(dp29\n'
    b'sg12\n(dp30\nsbg0\n(ccle.game_engine.models.map\nMapTemplate\np31\ng2\nNtp32\n'
    b'Rp33\n(dp34\nVnumbers\np35\n(lp36\nsVport_resources\np37\n(lp38\n'
    b'sVtile_resources\np39\n(lp40\nsVtopology\np41\n(dp42\nsbtp43\n.'
)


def test_seeded_map_contents_and_rng_match_presplit_snapshot() -> None:
    source = random.Random(4711)
    base = CatanMap.from_template(BASE_MAP_TEMPLATE, source)
    base_rng_state = source.getstate()
    boards = (base, CatanMap.from_template(MINI_MAP_TEMPLATE, source), TOURNAMENT_MAP)

    # Repr captures insertion order, IDs, all caches, and fractional Counter values;
    # unlike pickle bytes, it is independent of string interning/memoization.
    snapshot = repr(tuple(vars(board) for board in boards)).encode()
    assert hashlib.sha256(snapshot).hexdigest() == (
        "a3e951c652ebc42b7ae5850ef9c011190dceb9541ac173fa380bf9cf2c195995"
    )
    assert hashlib.sha256(pickle.dumps(source.getstate(), protocol=4)).hexdigest() == (
        "97c7643da431a67ed788461401de236fdba4dd16ce63a1486b756eab521f2d0c"
    )
    assert all(isinstance(counter, Counter) for counter in base.node_production.values())
    assert any(0 < value < 1 for counter in base.node_production.values() for value in counter.values())

    global_state = random.getstate()
    try:
        random.seed(4711)
        assert CatanMap.from_template(BASE_MAP_TEMPLATE, random).tiles == base.tiles
        assert random.getstate() == base_rng_state
    finally:
        random.setstate(global_state)


def test_explicit_orders_keep_input_consumption_and_empty_fallbacks() -> None:
    numbers = BASE_MAP_TEMPLATE.numbers.copy()
    ports = BASE_MAP_TEMPLATE.port_resources.copy()
    terrain = BASE_MAP_TEMPLATE.tile_resources.copy()
    source = random.Random(17)
    state = source.getstate()

    tiles = initialize_tiles(BASE_MAP_TEMPLATE, numbers, ports, terrain, source)
    assert numbers == [] and ports == []
    assert terrain == BASE_MAP_TEMPLATE.tile_resources
    assert source.getstate() == state
    assert [tile.resource for tile in tiles.values() if isinstance(tile, LandTile)] == terrain[::-1]
    assert [tile.number for tile in tiles.values() if isinstance(tile, LandTile) and tile.resource] == (
        BASE_MAP_TEMPLATE.numbers[::-1]
    )
    assert [tile.resource for tile in tiles.values() if isinstance(tile, Port)] == (
        BASE_MAP_TEMPLATE.port_resources[::-1]
    )

    empty_numbers: list[int] = []
    empty_ports: list[FastResource | None] = []
    expected_source = random.Random(17)
    assert initialize_tiles(BASE_MAP_TEMPLATE, empty_numbers, empty_ports, rng=source) == (
        initialize_tiles(BASE_MAP_TEMPLATE, rng=expected_source)
    )
    assert source.getstate() == expected_source.getstate()
    assert empty_numbers == [] and empty_ports == []


def test_historical_pickles_exports_defaults_and_singletons() -> None:
    land = LandTile(13, WOOD, 6, {}, {})
    port = Port(4, None, Direction.WEST, {}, {})
    water = Water({}, {})
    template = MapTemplate([], [], [], {})
    records = (land, port, water, template)
    assert pickle.loads(LEGACY_RECORDS) == records
    assert pickle.loads(pickle.dumps(records)) == records
    assert LandTile is map_types.LandTile and Port is map_types.Port
    assert Water is map_types.Water and MapTemplate is map_types.MapTemplate
    assert BASE_MAP_TEMPLATE is map_templates.BASE_MAP_TEMPLATE
    assert MINI_MAP_TEMPLATE is map_templates.MINI_MAP_TEMPLATE
    assert initialize_tiles is map_generation.initialize_tiles
    assert build_map("TOURNAMENT") is TOURNAMENT_MAP
    assert TOURNAMENT_MAP.tiles is TOURNAMENT_MAP_TILES

    first, second = CatanMap(), CatanMap()
    assert all(vars(first)[name] is vars(second)[name] for name in vars(first))
    assert all(
        field.default is MISSING and field.default_factory is MISSING
        for record in records
        for field in fields(record)
    )
    land.number = 8
    port.resource = WOOD
    template.numbers.append(3)
    water.nodes.update(land.nodes)
    with pytest.raises(FrozenInstanceError):
        setattr(water, "nodes", {})
    with pytest.raises(FrozenInstanceError):
        setattr(template, "numbers", [])
