"""Topology invariants and fact table completeness."""
import networkx as nx  # type: ignore

from cle.game_engine.models.board import STATIC_GRAPH, base_map
from cle.game_engine.models.map import NUM_EDGES, NUM_NODES, NUM_TILES
from sft.board.board_atlas import (
    LAND_GRAPH,
    LAND_NODES,
    FactTable,
    assert_topology_invariants,
)


def test_topology_invariants_hold() -> None:
    assert_topology_invariants()
    assert len(LAND_NODES) == NUM_NODES
    assert LAND_GRAPH.number_of_edges() == NUM_EDGES
    assert len(base_map.tiles_by_id) == NUM_TILES


def test_fact_tables_are_complete(tables: dict[str, FactTable]) -> None:
    expected = {
        "node_neighbors": NUM_NODES,
        "node_step": NUM_NODES * 6,  # six compass directions, empty at the coast
        "node_edges": NUM_NODES,
        "edge_endpoints": NUM_EDGES,
        "node_tiles": NUM_NODES,
        "tile_nodes": NUM_TILES,
        "tile_neighbors": NUM_TILES,
        "port_nodes": len(base_map.ports_by_id),
        "node_port": NUM_NODES,
        "node_distance": NUM_NODES * (NUM_NODES - 1) // 2,
        "node_path": NUM_NODES * (NUM_NODES - 1) // 2,
    }
    assert {name: len(t) for name, t in tables.items()} == expected


def test_facts_match_the_engine(tables: dict[str, FactTable]) -> None:
    """Spot-check against values the board-fluency corpus already treats as gold."""
    assert tables["node_tiles"].facts["<N09>"] == ["<T01>", "<T02>", "<T08>"]
    assert tables["node_neighbors"].facts["<N19>"] == ["<N20>", "<N21>", "<N46>"]
    assert tables["edge_endpoints"].facts["<E19_21>"] == ["<N19>", "<N21>"]


def test_neighbors_agree_with_static_graph(tables: dict[str, FactTable]) -> None:
    for node in LAND_NODES:
        expected = [f"<N{m:02d}>" for m in sorted(STATIC_GRAPH.neighbors(node)) if m in set(LAND_NODES)]
        assert tables["node_neighbors"].facts[f"<N{node:02d}>"] == expected


def test_distances_agree_with_networkx(tables: dict[str, FactTable]) -> None:
    lengths = dict(nx.all_pairs_shortest_path_length(LAND_GRAPH))
    for key, value in tables["node_distance"].facts.items():
        a, b = (int(tok[2:4]) for tok in key.split())
        assert value == [str(lengths[a][b])]


def test_paths_are_shortest_and_well_formed(tables: dict[str, FactTable]) -> None:
    lengths = dict(nx.all_pairs_shortest_path_length(LAND_GRAPH))
    for key, path in tables["node_path"].facts.items():
        a, b = (int(tok[2:4]) for tok in key.split())
        assert len(path) - 1 == lengths[a][b]
        assert {path[0], path[-1]} == {f"<N{a:02d}>", f"<N{b:02d}>"}
        ids = [int(tok[2:4]) for tok in path]
        for x, y in zip(ids, ids[1:]):
            assert LAND_GRAPH.has_edge(x, y)


def test_node_port_and_port_nodes_are_inverses(tables: dict[str, FactTable]) -> None:
    forward = tables["port_nodes"].facts
    reverse = tables["node_port"].facts
    for port, nodes in forward.items():
        for node in nodes:
            assert port in reverse[node]
    attached = {n for nodes in forward.values() for n in nodes}
    for node, ports in reverse.items():
        assert bool(ports) == (node in attached)
