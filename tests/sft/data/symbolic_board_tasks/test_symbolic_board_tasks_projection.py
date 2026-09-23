"""State shape, relevance projection, and case audits."""

import copy

import pytest

from evals.catan_board_bench.tokens import atlas_metadata
from sft.board.symbolic_board_tasks import (
    PhysicalStateError,
    decode_state,
    symbolic_prompt,
)
from sft.scripts.builders.build_symbolic_board_dataset import (
    graph_case_coverage,
    transfer_projection,
)

from .support import answer, reduced_state, target


def test_state_shape_physical_invariants_and_prompt_no_derived_input() -> None:
    state = reduced_state([(0, 1)], {3: ("BLUE", "city")})
    for bad in ({**state, "lengths": {"RED": 10}}, {**state, "colors": ["RED"] * 4},
                {**state, "board": state["board"] + "; <N00> empty"},
                {**state, "board": state["board"].replace("<N01> empty", "<N00> empty")},
                {**state, "board": state["board"].replace("red road", "green road")}):
        with pytest.raises(ValueError):
            decode_state(bad)
    with pytest.raises(PhysicalStateError, match="distance"):
        decode_state(reduced_state(buildings={0: ("RED", "settlement"), 1: ("BLUE", "city")}))
    with pytest.raises(PhysicalStateError, match="supply"):
        decode_state(reduced_state([e["id"] for e in atlas_metadata()["edges"][:16]]))
    with pytest.raises(ValueError):
        answer("owned_roads", state, color="GREEN")
    t = target(state, color="RED", phase="normal", near=None)
    prompt = symbolic_prompt("symbolic_settlement_locations", t)
    assert prompt.count(state["board"]) == 1
    for forbidden in ("coord", "node_tokens", "adjacent_tiles", "road_lengths", "has_longest_road", "legal_nodes", "<image>"):
        assert forbidden not in prompt
    # Only state and authored query/rules can enter the prompt.
    assert "Ignore hand, piece supply, and whose turn it is" in prompt
    assert t == target(state, color="RED", phase="normal", near=None)


def test_relevant_projection_removes_irrelevant_facts_but_keeps_rule_inputs() -> None:
    state = reduced_state([(0, 1)], {3: ("BLUE", "settlement")})
    setup = dict(color="RED", phase="setup", near=None)
    other_color = dict(setup, color="BLUE")
    assert transfer_projection("symbolic_settlement_locations", state, setup) == transfer_projection(
        "symbolic_settlement_locations", state, other_color)
    normal = dict(setup, phase="normal")
    assert transfer_projection("symbolic_settlement_locations", state, normal) != transfer_projection(
        "symbolic_settlement_locations", state, dict(normal, color="BLUE"))
    changed = copy.deepcopy(state)
    changed["colors"] = list(reversed(state["colors"]))
    changed["board"] = changed["board"].rsplit("robber ", 1)[0] + "robber <T01>"
    for task, query in (("symbolic_settlement_locations", normal), ("symbolic_longest_lengths", {})):
        assert transfer_projection(task, state, query) == transfer_projection(task, changed, query)
    # A city versus settlement is the same blocker; the blocker itself is essential.
    blocked = reduced_state([(0, 1), (1, 2)], {1: ("BLUE", "settlement")})
    city = reduced_state([(0, 1), (1, 2)], {1: ("BLUE", "city")})
    clear = reduced_state([(0, 1), (1, 2)])
    assert transfer_projection("symbolic_longest_lengths", blocked, {}) == transfer_projection("symbolic_longest_lengths", city, {})
    assert transfer_projection("symbolic_longest_lengths", blocked, {}) != transfer_projection("symbolic_longest_lengths", clear, {})


def test_graph_case_audit_detects_positive_cases_not_just_zero_real_pool_counts() -> None:
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    states = [reduced_state(ring + [(0, 20)], {0: ("BLUE", "settlement")}),
              reduced_state({**dict.fromkeys(ring[:-1], "RED"),
                             **dict.fromkeys([(24, 25), (25, 26), (26, 27), (27, 28), (28, 29)], "BLUE")})]
    records = [{"state": state, "provenance": {"state_id": f"reduced_fixture_{i}",
                                               "board_fact_sha256": "unit_test_only"}}
               for i, state in enumerate(states)]
    coverage = graph_case_coverage(records)
    assert coverage["counts"]["cycle"] == 1
    assert coverage["counts"]["effective_blocker"] == 1
    assert coverage["counts"]["branch"] == 1
    assert coverage["counts"]["tied_max_ge5"] == 1
    assert coverage["missing_coverage"] == []
