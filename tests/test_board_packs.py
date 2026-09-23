"""Tests for randomized multi-operation board packs.

Packs are driven by the live task machinery, so these exercise real decoded
boards rather than fixtures wherever an artifact is available.
"""
import json
import random
from pathlib import Path
from typing import Any

import pytest

from sft.board.board_packs import (
    Pack,
    PackEntry,
    build_pack,
    compose_pack,
    entry_correct,
    generate_packs,
    parse_pack_answer,
    render_pack,
    render_query,
    sample_entry_count,
    score_pack,
)
from sft.board.symbolic_board_tasks import _atlas, decode_state
from sft.board.symbolic_board_tasks._geometry import Atlas
from sft.scripts.builders import build_board_fluency_review as rev

REVIEW = (
    Path(__file__).resolve().parents[1]
    / "artifacts/generated/sft/symbolic_board_fluency_review_v1/review.jsonl"
)


def _rows() -> list[dict[str, Any]]:
    if not REVIEW.exists():
        return []
    return [
        json.loads(line) for line in REVIEW.read_text().splitlines() if line.strip()
    ]


needs_review = pytest.mark.skipif(not _rows(), reason="review artifact not built locally")


@pytest.fixture(scope="module")
def atlas() -> Atlas:
    return _atlas()


@pytest.fixture(scope="module")
def donors(atlas: Atlas) -> list[rev.Donor]:
    states: dict[str, Any] = {}
    for row in _rows():
        meta: Any = row["metadata"]
        states.setdefault(meta["state_id"], meta["target"]["state"])
    return [
        rev.Donor(
            line=0, row={}, state=state, data=decode_state(state),
            provenance={}, state_hash="", line_sha256="",
        )
        for state in list(states.values())[:12]
    ]


# --- rendering -------------------------------------------------------------


def test_render_query_is_generic_over_the_query_dict() -> None:
    assert render_query("node_pip_sum", {"operation": "node_pip_sum", "node": "<N23>"}) == (
        "node_pip_sum node=<N23>"
    )
    rendered = render_query(
        "shortest_distance",
        {"operation": "shortest_distance", "color": "RED", "start": "<N02>", "end": "<N46>"},
    )
    assert rendered.startswith("shortest_distance ")
    for token in ("color=RED", "start=<N02>", "end=<N46>"):
        assert token in rendered


def test_parse_pack_answer_reads_numbered_lines() -> None:
    assert parse_pack_answer("1: NONE\n2: <N04> <N05>\n") == {1: "NONE", 2: "<N04> <N05>"}


def test_parse_pack_answer_ignores_unnumbered_noise() -> None:
    assert parse_pack_answer("here you go:\n1: 7\nthanks") == {1: "7"}


def test_shared_rubrics_shortens_the_prompt() -> None:
    pack = Pack(
        board="<T00> brick 9;",
        entries=[
            PackEntry(1, "node_pip_sum", {"operation": "node_pip_sum", "node": "<N23>"}, "9"),
            PackEntry(2, "node_pip_sum", {"operation": "node_pip_sum", "node": "<N44>"}, "7"),
            PackEntry(3, "node_pip_sum", {"operation": "node_pip_sum", "node": "<N12>"}, "4"),
        ],
    )
    shared, _ = render_pack(pack, shared_rubrics=True)
    repeated, _ = render_pack(pack, shared_rubrics=False)
    assert len(shared) < len(repeated)
    # The rubric appears once, not once per entry.
    assert shared.count("Pips are the number of two-dice outcomes") == 1


def test_target_is_one_numbered_line_per_entry() -> None:
    pack = Pack(
        board="board",
        entries=[
            PackEntry(1, "node_pip_sum", {"operation": "node_pip_sum", "node": "<N23>"}, "9"),
            PackEntry(2, "port_access", {"operation": "port_access", "color": "RED"}, "NONE"),
        ],
    )
    _, target = render_pack(pack)
    assert target.splitlines() == ["1: 9", "2: NONE"]


# --- composition -----------------------------------------------------------


def test_sample_entry_count_stays_in_range_and_favours_small_packs() -> None:
    rng = random.Random(0)
    draws = [sample_entry_count(rng, 240, 48) for _ in range(4000)]
    assert min(draws) >= 1 and max(draws) <= 48
    assert sum(1 for d in draws if d <= 4) / len(draws) > 0.15


def test_sample_entry_count_respects_available_candidates() -> None:
    rng = random.Random(1)
    assert all(sample_entry_count(rng, 3, 48) <= 3 for _ in range(200))


def test_compose_pack_on_empty_candidates_is_empty() -> None:
    assert compose_pack([], rng=random.Random(0)) == []


@needs_review
def test_compose_pack_never_repeats_a_query(atlas: Atlas, donors: list[rev.Donor]) -> None:
    rng = random.Random(4)
    for donor in donors[:6]:
        chosen = compose_pack(
            rev.candidates_for(donor, atlas, rng), rng=rng, max_entries=48
        )
        keys = [(c.operation, tuple(sorted(c.query.items()))) for c in chosen]
        assert len(keys) == len(set(keys))


@needs_review
def test_max_per_operation_is_respected(atlas: Atlas, donors: list[rev.Donor]) -> None:
    rng = random.Random(4)
    chosen = compose_pack(
        rev.candidates_for(donors[0], atlas, rng),
        rng=rng, max_entries=48, max_per_operation=2,
    )
    counts = {}
    for candidate in chosen:
        counts[candidate.operation] = counts.get(candidate.operation, 0) + 1
    assert counts and max(counts.values()) <= 2


@needs_review
def test_share_rate_drives_argument_reuse(atlas: Atlas, donors: list[rev.Donor]) -> None:
    """The retrieval-then-composition pairing depends on this, so it is asserted."""
    def shared_fraction(rate: float) -> float:
        rng: Any = random.Random(11)
        totals: list[float] = []
        for donor in donors:
            cands: Any = rev.candidates_for(donor, atlas, rng)
            chosen: Any = compose_pack(cands, rng=rng, max_entries=48, share_rate=rate)
            if len(chosen) < 8:
                continue
            facts: Any = rev.Facts(donor.data, atlas)
            pack: Any = build_pack(donor.state["board"], facts, chosen)
            totals.append(pack.shared_argument_entries() / len(pack.entries))
        return sum(totals) / len(totals)

    assert shared_fraction(0.9) > shared_fraction(0.0)


# --- scoring ---------------------------------------------------------------


@needs_review
def test_entry_correct_matches_the_recorded_review_scores() -> None:
    """The pack scorer must agree with the scorer that graded r04."""
    rows: Any = _rows()
    assert rows
    for row in rows:
        meta: Any = row["metadata"]
        expected: Any = meta["answer"]
        assert entry_correct(meta["operation"], expected, expected)


def test_entry_correct_rejects_unknown_operations() -> None:
    with pytest.raises(KeyError):
        entry_correct("not_an_operation", "1", "1")


VALID_VECTOR = '{"brick":0,"ore":0,"sheep":0,"wheat":1,"wood":0}'


def test_entry_correct_is_false_on_malformed_predictions() -> None:
    assert not entry_correct("node_pip_sum", "9", "not a number")
    assert not entry_correct("roll_production", VALID_VECTOR, "{{{")
    assert not entry_correct("roll_production", VALID_VECTOR, '{"brick":0}')


def test_entry_correct_raises_on_a_malformed_gold() -> None:
    """Bad golds are a data bug and must not be silently scored as wrong."""
    with pytest.raises(ValueError):
        entry_correct("roll_production", '{"brick":0}', VALID_VECTOR)


def test_entry_correct_ignores_trailing_transport_tokens() -> None:
    assert entry_correct("node_pip_sum", "9", "9<|im_end|><|endoftext|>")


@needs_review
def test_gold_target_round_trips_to_full_accuracy(atlas: Atlas, donors: list[rev.Donor]) -> None:
    rng = random.Random(9)
    donor = donors[0]
    facts = rev.Facts(donor.data, atlas)
    chosen = compose_pack(rev.candidates_for(donor, atlas, rng), rng=rng, max_entries=48)
    pack = build_pack(donor.state["board"], facts, chosen)
    _, target = render_pack(pack)
    score = score_pack(pack, target)
    assert score.entries_correct == score.entries_total
    assert score.entries_missing == 0
    assert score.accuracy == 1.0


@needs_review
def test_scoring_is_per_entry_not_all_or_nothing(atlas: Atlas, donors: list[rev.Donor]) -> None:
    rng = random.Random(9)
    donor: Any = donors[0]
    facts: Any = rev.Facts(donor.data, atlas)
    chosen: Any = compose_pack(rev.candidates_for(donor, atlas, rng), rng=rng, max_entries=48)
    pack = build_pack(donor.state["board"], facts, chosen)
    _, target = render_pack(pack)

    lines = target.splitlines()
    lines[0] = lines[0].split(":")[0] + ": GARBAGE"
    score = score_pack(pack, "\n".join(lines))
    assert score.entries_correct == score.entries_total - 1
    assert 0.0 < score.accuracy < 1.0


@needs_review
def test_truncated_response_counts_entries_as_missing(atlas: Atlas, donors: list[rev.Donor]) -> None:
    rng = random.Random(9)
    donor: Any = donors[0]
    facts: Any = rev.Facts(donor.data, atlas)
    chosen: Any = compose_pack(rev.candidates_for(donor, atlas, rng), rng=rng, max_entries=48)
    pack = build_pack(donor.state["board"], facts, chosen)
    _, target = render_pack(pack)
    half = target.splitlines()[: len(pack.entries) // 2]
    score = score_pack(pack, "\n".join(half))
    assert score.entries_missing == len(pack.entries) - len(half)


@needs_review
def test_score_attributes_entries_to_operations(atlas: Atlas, donors: list[rev.Donor]) -> None:
    rng = random.Random(9)
    donor = donors[0]
    facts = rev.Facts(donor.data, atlas)
    chosen = compose_pack(rev.candidates_for(donor, atlas, rng), rng=rng, max_entries=48)
    pack = build_pack(donor.state["board"], facts, chosen)
    _, target = render_pack(pack)
    score = score_pack(pack, target)
    assert set(score.by_operation) == set(pack.operations)
    assert sum(t for _, t in score.by_operation.values()) == len(pack.entries)


@needs_review
def test_generated_rows_carry_board_and_every_query(atlas: Atlas, donors: list[rev.Donor]) -> None:
    rng = random.Random(2)
    rows: Any = list(generate_packs(donors[:4], atlas, rng=rng, packs_per_board=2))
    assert rows
    for row in rows:
        meta: Any = row["metadata"]
        prompt: Any = row["messages"][0]["content"]
        assert prompt.startswith(meta["queries"][0]["query"].get("board", "") or "<T00>")
        assert meta["entries"] == len(meta["queries"])
        assert len(parse_pack_answer(row["messages"][1]["content"])) == meta["entries"]
