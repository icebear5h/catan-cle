"""Per-entry scoring and exactness rules."""
import pytest

from sft.board.board_atlas import (
    FactTable,
    render_example,
    score_atlas,
)


def test_scoring_is_per_entry_not_exact_match(tables: dict[str, FactTable]) -> None:
    table = tables["node_neighbors"]
    keys = ["<N19>", "<N03>", "<N00>"]
    _, answer = render_example(table, keys)

    dropped = answer.replace(" <N46>", "", 1)
    score = score_atlas(table, keys, dropped)
    assert not score.exact
    assert score.entry_accuracy == pytest.approx(2 / 3)
    assert score.atom_precision == 1.0
    assert score.atom_recall < 1.0


def test_scoring_penalises_spurious_atoms(tables: dict[str, FactTable]) -> None:
    table = tables["node_neighbors"]
    keys = ["<N19>"]
    _, answer = render_example(table, keys)
    score = score_atlas(table, keys, answer.replace("<N46>", "<N46> <N07>"))
    assert score.atom_recall == 1.0
    assert score.atom_precision < 1.0
    assert not score.exact


def test_set_answers_ignore_atom_order_but_sequences_do_not(tables: dict[str, FactTable]) -> None:
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


def test_unasked_entries_break_exactness_without_inflating_accuracy(tables: dict[str, FactTable]) -> None:
    table = tables["node_neighbors"]
    keys = ["<N19>"]
    _, answer = render_example(table, keys)
    score = score_atlas(table, keys, answer + "\n<N07>: <N06> <N08>")
    assert score.entry_accuracy == 1.0
    assert not score.exact
    assert score.atom_precision < 1.0


def test_unparseable_response_scores_zero(tables: dict[str, FactTable]) -> None:
    table = tables["node_neighbors"]
    keys = ["<N19>", "<N03>"]
    score = score_atlas(table, keys, "the neighbours are hard to say")
    assert score.entry_accuracy == 0.0
    assert score.entries_missing == 2
    assert not score.exact
