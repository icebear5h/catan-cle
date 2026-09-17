"""Tests for the static board-topology atlas.

The atlas is a closed world, so these assert exact facts against the engine
rather than sampling properties.
"""

import random

import networkx as nx  # type: ignore
import pytest

from cle.game_engine.models.board import STATIC_GRAPH, base_map
from cle.game_engine.models.map import NUM_EDGES, NUM_NODES, NUM_TILES
from sft.board_atlas import (
    LAND_GRAPH,
    LAND_NODES,
    NONE_ANSWER,
    assert_topology_invariants,
    build_fact_tables,
    coverage_report,
    generate_examples,
    generate_exhaustive,
    parse_answer,
    render_example,
    sample_cardinality,
    score_atlas,
)


@pytest.fixture(scope="module")
def tables():
    return build_fact_tables()


def test_topology_invariants_hold():
    assert_topology_invariants()
    assert len(LAND_NODES) == NUM_NODES
    assert LAND_GRAPH.number_of_edges() == NUM_EDGES
    assert len(base_map.tiles_by_id) == NUM_TILES


def test_fact_tables_are_complete(tables):
    expected = {
        "node_neighbors": NUM_NODES,
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


def test_facts_match_the_engine(tables):
    """Spot-check against values the board-fluency corpus already treats as gold."""
    assert tables["node_tiles"].facts["<N09>"] == ["<T01>", "<T02>", "<T08>"]
    assert tables["node_neighbors"].facts["<N19>"] == ["<N20>", "<N21>", "<N46>"]
    assert tables["edge_endpoints"].facts["<E19_21>"] == ["<N19>", "<N21>"]


def test_neighbors_agree_with_static_graph(tables):
    for node in LAND_NODES:
        expected = [f"<N{m:02d}>" for m in sorted(STATIC_GRAPH.neighbors(node)) if m in set(LAND_NODES)]
        assert tables["node_neighbors"].facts[f"<N{node:02d}>"] == expected


def test_distances_agree_with_networkx(tables):
    lengths = dict(nx.all_pairs_shortest_path_length(LAND_GRAPH))
    for key, value in tables["node_distance"].facts.items():
        a, b = (int(tok[2:4]) for tok in key.split())
        assert value == [str(lengths[a][b])]


def test_paths_are_shortest_and_well_formed(tables):
    lengths = dict(nx.all_pairs_shortest_path_length(LAND_GRAPH))
    for key, path in tables["node_path"].facts.items():
        a, b = (int(tok[2:4]) for tok in key.split())
        assert len(path) - 1 == lengths[a][b]
        assert {path[0], path[-1]} == {f"<N{a:02d}>", f"<N{b:02d}>"}
        ids = [int(tok[2:4]) for tok in path]
        for x, y in zip(ids, ids[1:]):
            assert LAND_GRAPH.has_edge(x, y)


def test_node_port_and_port_nodes_are_inverses(tables):
    forward = tables["port_nodes"].facts
    reverse = tables["node_port"].facts
    for port, nodes in forward.items():
        for node in nodes:
            assert port in reverse[node]
    attached = {n for nodes in forward.values() for n in nodes}
    for node, ports in reverse.items():
        assert bool(ports) == (node in attached)


def test_empty_answers_render_as_none(tables):
    table = tables["node_port"]
    empty = next(k for k, v in table.facts.items() if not v)
    _, answer = render_example(table, [empty])
    assert answer == f"{empty}: {NONE_ANSWER}"
    assert parse_answer(answer) == {empty: []}


def test_prompt_carries_the_queried_keys_in_order(tables):
    table = tables["node_neighbors"]
    keys = ["<N19>", "<N03>", "<N00>"]
    prompt, answer = render_example(table, keys)
    for key in keys:
        assert key in prompt
    assert [line.split(":")[0] for line in answer.splitlines()] == keys


def test_sample_cardinality_is_in_range_and_favours_small_k():
    rng = random.Random(0)
    draws = [sample_cardinality(rng, 54, 24) for _ in range(5000)]
    assert min(draws) >= 1 and max(draws) <= 24
    # Single-key queries are the form traversal consumes, so they must not be rare.
    assert draws.count(1) / len(draws) > 0.10


def test_sample_cardinality_respects_small_key_sets():
    rng = random.Random(1)
    assert all(sample_cardinality(rng, 1, 24) == 1 for _ in range(50))
    assert all(sample_cardinality(rng, 3, 24) <= 3 for _ in range(200))


def test_exhaustive_generation_covers_every_fact(tables):
    examples = list(generate_exhaustive(tables, rng=random.Random(7), max_k=24))
    report = coverage_report(tables, examples)
    assert report["fully_covered"]
    assert report["covered_facts"] == report["total_facts"]
    for name, entry in report["by_table"].items():
        assert entry["uncovered"] == [], name


def test_generated_examples_are_answerable_from_their_own_table(tables):
    rng = random.Random(3)
    for example in generate_examples(tables, n_examples=200, rng=rng, max_k=12):
        table = tables[example.table]
        score = score_atlas(table, example.keys, example.answer)
        assert score.exact
        assert score.entry_accuracy == 1.0


def test_scoring_is_per_entry_not_exact_match(tables):
    table = tables["node_neighbors"]
    keys = ["<N19>", "<N03>", "<N00>"]
    _, answer = render_example(table, keys)

    dropped = answer.replace(" <N46>", "", 1)
    score = score_atlas(table, keys, dropped)
    assert not score.exact
    assert score.entry_accuracy == pytest.approx(2 / 3)
    assert score.atom_precision == 1.0
    assert score.atom_recall < 1.0


def test_scoring_penalises_spurious_atoms(tables):
    table = tables["node_neighbors"]
    keys = ["<N19>"]
    _, answer = render_example(table, keys)
    score = score_atlas(table, keys, answer.replace("<N46>", "<N46> <N07>"))
    assert score.atom_recall == 1.0
    assert score.atom_precision < 1.0
    assert not score.exact


def test_set_answers_ignore_atom_order_but_sequences_do_not(tables):
    node_table = tables["node_neighbors"]
    key = "<N19>"
    atoms = node_table.facts[key]
    shuffled = f"{key}: {' '.join(reversed(atoms))}"
    assert score_atlas(node_table, [key], shuffled).exact

    path_table = tables["node_path"]
    pair = next(k for k, v in path_table.facts.items() if len(v) > 2)
    path = path_table.facts[pair]
    reversed_path = f"{pair}: {' '.join(reversed(path))}"
    assert not score_atlas(path_table, [pair], reversed_path).exact


def test_unasked_entries_break_exactness_without_inflating_accuracy(tables):
    table = tables["node_neighbors"]
    keys = ["<N19>"]
    _, answer = render_example(table, keys)
    score = score_atlas(table, keys, answer + "\n<N07>: <N06> <N08>")
    assert score.entry_accuracy == 1.0
    assert not score.exact
    assert score.atom_precision < 1.0


def test_unparseable_response_scores_zero(tables):
    table = tables["node_neighbors"]
    keys = ["<N19>", "<N03>"]
    score = score_atlas(table, keys, "the neighbours are hard to say")
    assert score.entry_accuracy == 0.0
    assert score.entries_missing == 2
    assert not score.exact
