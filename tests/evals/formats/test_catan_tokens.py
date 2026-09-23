from cle.game_engine.models.map import NUM_EDGES, NUM_NODES, NUM_TILES
from cle.game_engine.models.player import Color
from evals.catan_board_bench.tokens import (
    added_tokens,
    base_edges,
    color_token,
    edge_token,
    node_token,
    object_token,
    port_token,
    recognition_answer_token,
    recognition_answer_tokens,
    recognition_query_token,
    recognition_query_tokens,
    recognition_token_inventory,
    tile_token,
    token_manifest,
)


def test_atlas_token_counts_are_stable() -> None:
    manifest = token_manifest()

    assert manifest["counts"]["node"] == NUM_NODES
    assert manifest["counts"]["edge"] == NUM_EDGES
    assert manifest["counts"]["tile"] == NUM_TILES
    assert manifest["counts"]["port"] == 9
    assert manifest["counts"]["object"] == 1
    assert manifest["counts"]["recognition_query"] == 6
    assert manifest["counts"]["recognition_answer"] == 60
    assert len(added_tokens()) == manifest["total"]
    assert len(added_tokens()) == len(set(added_tokens()))


def test_slot_token_formatting() -> None:
    assert node_token(0) == "<N00>"
    assert node_token(53) == "<N53>"
    assert tile_token(0) == "<T00>"
    assert tile_token(18) == "<T18>"
    assert port_token(0) == "<P00>"
    assert port_token(8) == "<P08>"
    assert edge_token((17, 3)) == "<E03_17>"
    assert object_token("ROBBER") == "<ROBBER>"
    assert color_token(Color.MYSTIC_BLUE) == "<MYSTIC_BLUE>"


def test_recognition_tokens_are_atomic_head_specific_classes() -> None:
    inventory = recognition_token_inventory()

    assert recognition_query_token("node.occupancy") == "<Q_NODE_OCCUPANCY>"
    assert recognition_answer_token("node.occupancy", "WHITE_SETTLEMENT") == (
        "<A_NODE_OCCUPANCY_WHITE_SETTLEMENT>"
    )
    assert recognition_answer_token("edge.owner", "MYSTIC_BLUE") == ("<A_EDGE_OWNER_MYSTIC_BLUE>")
    assert len(recognition_query_tokens()) == 6
    assert len(recognition_answer_tokens()) == 60
    assert inventory["counts"] == {"atlas": 154, "query": 6, "answer": 60, "total": 220}
    assert len(inventory["tokens"]) == len(set(inventory["tokens"])) == 220
    assert not set(inventory["tokens"]) & {
        "<BUILD_CITY>",
        "<WOOD>",
        "<RED>",
        "<ROBBER>",
    }


def test_base_edges_are_canonical_and_complete() -> None:
    edges = base_edges()

    assert len(edges) == NUM_EDGES
    assert edges == sorted(edges)
    assert all(a < b for a, b in edges)
