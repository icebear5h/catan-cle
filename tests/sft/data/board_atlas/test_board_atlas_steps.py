"""Directional node steps and compass symmetry."""
from cle.game_engine.models.map import NUM_NODES
from sft.board.board_atlas import (
    LAND_GRAPH,
    LAND_NODES,
    FactTable,
)

from .support import COMPASS


def test_node_step_is_total_over_nodes_and_directions(tables: dict[str, FactTable]) -> None:
    steps = tables["node_step"].facts
    assert len(steps) == NUM_NODES * len(COMPASS)
    for node in LAND_NODES:
        for direction in COMPASS:
            assert f"<N{node:02d}> {direction}" in steps


def test_node_step_has_at_most_one_neighbour_per_direction(tables: dict[str, FactTable]) -> None:
    for value in tables["node_step"].facts.values():
        assert len(value) <= 1


def test_node_step_union_reproduces_node_neighbors(tables: dict[str, FactTable]) -> None:
    steps, neighbours = tables["node_step"].facts, tables["node_neighbors"].facts
    for node in LAND_NODES:
        token = f"<N{node:02d}>"
        union = sorted(
            {v[0] for d in COMPASS for v in [steps[f"{token} {d}"]] if v}
        )
        assert union == neighbours[token]


def test_node_step_filled_slots_match_edge_count(tables: dict[str, FactTable]) -> None:
    filled = sum(1 for v in tables["node_step"].facts.values() if v)
    assert filled == 2 * LAND_GRAPH.number_of_edges()


def test_node_step_is_symmetric_under_opposite_directions(tables: dict[str, FactTable]) -> None:
    steps = tables["node_step"].facts
    opposite = dict(zip(COMPASS, COMPASS[3:] + COMPASS[:3]))
    for key, value in steps.items():
        if not value:
            continue
        token, direction = key.rsplit(" ", 1)
        back = steps[f"{value[0]} {opposite[direction]}"]
        assert back == [token], f"{key} -> {value[0]} does not step back"
