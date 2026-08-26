from catan_board_bench.tokens import (
    added_tokens,
    base_edges,
    color_token,
    edge_token,
    node_token,
    object_token,
    port_token,
    tile_token,
    token_manifest,
)
from game_engine.models.map import NUM_EDGES, NUM_NODES, NUM_TILES
from game_engine.models.player import Color


def test_atlas_token_counts_are_stable():
    manifest = token_manifest()

    assert manifest["counts"]["node"] == NUM_NODES
    assert manifest["counts"]["edge"] == NUM_EDGES
    assert manifest["counts"]["tile"] == NUM_TILES
    assert manifest["counts"]["port"] == 9
    assert manifest["counts"]["object"] == 1
    assert len(added_tokens()) == manifest["total"]
    assert len(added_tokens()) == len(set(added_tokens()))


def test_slot_token_formatting():
    assert node_token(0) == "<N00>"
    assert node_token(53) == "<N53>"
    assert tile_token(0) == "<T00>"
    assert tile_token(18) == "<T18>"
    assert port_token(0) == "<P00>"
    assert port_token(8) == "<P08>"
    assert edge_token((17, 3)) == "<E03_17>"
    assert object_token("ROBBER") == "<ROBBER>"
    assert color_token(Color.MYSTIC_BLUE) == "<MYSTIC_BLUE>"


def test_base_edges_are_canonical_and_complete():
    edges = base_edges()

    assert len(edges) == NUM_EDGES
    assert edges == sorted(edges)
    assert all(a < b for a, b in edges)
