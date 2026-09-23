"""Tests for connected-road readouts.

The readout must agree with the operations it is meant to make redundant, so the
consistency tests below drive the real `Facts` implementation rather than a
fixture of hand-written partitions.
"""
import json
from pathlib import Path

import pytest

from sft.board.board_readouts import (
    NONE_ANSWER,
    connected_roads,
    implied_component_count,
    implied_component_roads,
    parse_answer,
    render_answer,
    score_readout,
)
from sft.board.symbolic_board_tasks import _atlas, decode_state
from sft.scripts.builders import build_board_fluency_review as rev


@pytest.fixture
def partition() -> list[list[str]]:
    return [
        ["<E00_01>", "<E01_02>", "<E02_03>"],
        ["<E10_11>", "<E11_12>"],
        ["<E39_41>"],
    ]


def test_render_and_parse_round_trip(partition: list[list[str]]) -> None:
    assert parse_answer(render_answer(partition)) == partition


def test_empty_partition_renders_as_none() -> None:
    assert render_answer([]) == NONE_ANSWER
    assert parse_answer(NONE_ANSWER) == []


def test_perfect_answer_scores_exact(partition: list[list[str]]) -> None:
    score = score_readout(partition, render_answer(partition))
    assert score.exact
    assert score.component_recall == 1.0
    assert score.edge_recall == 1.0
    assert score.pair_agreement == 1.0


def test_component_order_and_edge_order_do_not_matter(partition: list[list[str]]) -> None:
    shuffled = [list(reversed(partition[2])), list(reversed(partition[0])), partition[1]]
    assert score_readout(partition, render_answer(shuffled)).exact


def test_merging_all_components_keeps_edges_but_loses_structure(partition: list[list[str]]) -> None:
    merged = render_answer([[e for part in partition for e in part]])
    score = score_readout(partition, merged)
    assert not score.exact
    assert score.edge_recall == 1.0  # every edge is present ...
    assert score.edges_missing == 0
    assert score.component_recall == 0.0  # ... but no component survived
    assert score.pair_agreement < 1.0


def test_splitting_a_component_is_penalised_by_pair_agreement(partition: list[list[str]]) -> None:
    split = [partition[0][:1], partition[0][1:], partition[1], partition[2]]
    score = score_readout(partition, render_answer(split))
    assert not score.exact
    assert score.edge_recall == 1.0
    assert score.components_predicted == 4
    assert score.pair_agreement < 1.0


def test_dropped_edge_shows_as_missing(partition: list[list[str]]) -> None:
    dropped = [partition[0][1:], partition[1], partition[2]]
    score = score_readout(partition, render_answer(dropped))
    assert score.edges_missing == 1
    assert score.edges_spurious == 0
    assert score.edge_recall < 1.0


def test_spurious_edge_shows_as_spurious(partition: list[list[str]]) -> None:
    extra = [partition[0] + ["<E52_53>"], partition[1], partition[2]]
    score = score_readout(partition, render_answer(extra))
    assert score.edges_spurious == 1
    assert score.edges_missing == 0


def test_none_answer_against_real_partition_scores_zero_recall(partition: list[list[str]]) -> None:
    score = score_readout(partition, NONE_ANSWER)
    assert not score.exact
    assert score.edge_recall == 0.0
    assert score.component_recall == 0.0


def test_pair_agreement_degrades_smoothly(partition: list[list[str]]) -> None:
    """One misplaced edge must score better than total collapse."""
    moved = [partition[0][:2], partition[1] + [partition[0][2]], partition[2]]
    merged = [[e for part in partition for e in part]]
    near = score_readout(partition, render_answer(moved)).pair_agreement
    far = score_readout(partition, render_answer(merged)).pair_agreement
    assert 0.0 < far < near < 1.0


def test_implied_helpers_agree_with_the_partition(partition: list[list[str]]) -> None:
    assert implied_component_count(partition) == 3
    assert implied_component_roads(partition, "<E11_12>") == partition[1]
    with pytest.raises(KeyError):
        implied_component_roads(partition, "<E52_53>")


# --- consistency against the live corpus ----------------------------------


def _review_cases() -> list[dict[str, object]]:
    """Decode the board-fluency review rows, if the artifact is present."""
    path = (
        Path(__file__).resolve().parents[3]
        / "artifacts/generated/sft/symbolic_board_fluency_review_v1/review.jsonl"
    )
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("metadata", {}).get("operation") in
            {"component_count", "component_roads"}]


@pytest.mark.skipif(not _review_cases(), reason="review artifact not built locally")
def test_readout_implies_existing_component_golds() -> None:
    atlas = _atlas()
    for row in _review_cases():
        meta = row["metadata"]
        query = meta["target"]["query"]
        facts = rev.Facts(decode_state(meta["target"]["state"]), atlas)
        parts = connected_roads(facts, query["color"])
        if meta["operation"] == "component_count":
            assert str(implied_component_count(parts)) == meta["answer"]
        else:
            expected = meta["answer"].split()
            assert sorted(implied_component_roads(parts, query["edge"])) == sorted(expected)


def test_every_owned_road_appears_exactly_once() -> None:
    """A partition, not a cover: no edge may repeat across components."""
    atlas = _atlas()
    data = {
        "colors": ("RED", "BLUE"),
        "buildings": {"<N01>": ("BLUE", "settlement")},
        "roads": {"<E00_01>": "RED", "<E01_02>": "RED", "<E10_11>": "RED"},
        # Facts derives resource coverage eagerly, so every tile must be present.
        "tiles": {f"<T{i:02d}>": ("brick", 9) for i in range(19)},
        "robber": "<T00>",
    }
    facts = rev.Facts(data, atlas)
    parts = connected_roads(facts, "RED")
    flat = [edge for part in parts for edge in part]
    assert sorted(flat) == ["<E00_01>", "<E01_02>", "<E10_11>"]
    assert len(flat) == len(set(flat))
    # An enemy settlement at <N01> blocks the junction, so the two touching
    # roads must not be merged.
    assert {frozenset(p) for p in parts} == {
        frozenset({"<E00_01>"}),
        frozenset({"<E01_02>"}),
        frozenset({"<E10_11>"}),
    }
