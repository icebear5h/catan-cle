"""Canonical graph, BFS determinism, and node boundaries."""

import networkx as nx
import pytest

from cle.game_engine.public_board import JsonValue
from evals.catan_board_bench.tokens import atlas_metadata
from sft.board.spatial_tasks import (
    atlas_node_graph,
    local_node_tiles,
    node_tile_tokens,
    shortest_node_path,
)

from .support import PATH, PATH_METADATA


def test_canonical_graph_and_all_node_tile_boundaries() -> None:
    atlas = atlas_metadata()
    graph = atlas_node_graph()
    assert set(graph) == {node["token"] for node in atlas["nodes"]}
    assert len(graph) == 54 and sum(map(len, graph.values())) == 144
    assert {tuple(sorted((a, b))) for a in graph for b in graph[a]} == {
        tuple(f"<N{node:02d}>" for node in edge["id"]) for edge in atlas["edges"]
    }
    assert graph["<N00>"] == {"<N01>", "<N05>", "<N20>"}
    assert graph["<N53>"] == {"<N24>", "<N52>"}
    assert node_tile_tokens("<N00>") == ["<T00>", "<T05>", "<T06>"]
    assert node_tile_tokens("<N53>") == ["<T18>"]
    counts = set()
    for node in atlas["nodes"]:
        expected = sorted(
            tile["token"] for tile in atlas["tiles"] if node["id"] in tile["nodes"].values()
        )
        assert node_tile_tokens(node["token"]) == expected
        counts.add(len(expected))
    assert counts == {1, 2, 3}
    graph["<N00>"].clear()
    node_tile_tokens("<N00>").clear()
    assert atlas_node_graph()["<N00>"] == {"<N01>", "<N05>", "<N20>"}
    assert len(node_tile_tokens("<N00>")) == 3


def test_bfs_is_deterministic_and_minimal_for_every_node_pair() -> None:
    atlas = atlas_metadata()
    graph = nx.Graph(tuple(f"<N{node:02d}>" for node in edge["id"]) for edge in atlas["edges"])
    assert shortest_node_path(*PATH_METADATA["target"].values()) == PATH
    for start, distances in nx.all_pairs_shortest_path_length(graph):
        for end, distance in distances.items():
            path = shortest_node_path(start, end)
            assert path == shortest_node_path(start, end)
            assert path[0] == start and path[-1] == end
            assert len(path) == distance + 1
            assert all(graph.has_edge(a, b) for a, b in zip(path, path[1:]))
            if start == end:
                assert path == [start]


@pytest.mark.parametrize(
    "node",
    ["<N54>", "<N99>", "<N-1>", "N00", "<N0>", "<n00>", "<T00>", " <N00>", 0, True, None, []],
)
def test_invalid_nodes_raise(node: object, contract: dict[str, JsonValue]) -> None:
    for call in (
        lambda: node_tile_tokens(node),
        lambda: shortest_node_path(node, "<N00>"),
        lambda: shortest_node_path("<N00>", node),
        lambda: local_node_tiles(contract, node),
    ):
        with pytest.raises(ValueError):
            call()
