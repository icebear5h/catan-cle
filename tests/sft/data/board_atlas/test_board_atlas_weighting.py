"""Weighted sampling and empty-fact down-weighting."""
import random

from sft.board.board_atlas import (
    FactTable,
    answer_shape_report,
    coverage_report,
    generate_examples,
    generate_exhaustive,
    is_empty_fact,
    weighted_sample,
)


def test_empty_facts_are_a_large_share_of_some_tables(tables: dict[str, FactTable]) -> None:
    """The reason down-weighting exists: uniform sampling would be mostly NONE."""
    step = tables["node_step"]
    empty = sum(1 for k in step.facts if is_empty_fact(step, k))
    assert empty / len(step.facts) > 0.5


def test_weighted_sample_respects_k_and_never_repeats(tables: dict[str, FactTable]) -> None:
    rng = random.Random(0)
    table = tables["node_step"]
    keys = list(table.facts)
    weights = [0.2 if is_empty_fact(table, k) else 1.0 for k in keys]
    for k in (1, 5, 24):
        drawn = weighted_sample(keys, k, weights, rng)
        assert len(drawn) == k
        assert len(set(drawn)) == k
        assert set(drawn) <= set(keys)


def test_weighted_sample_returns_everything_when_k_exceeds_pool(tables: dict[str, FactTable]) -> None:
    rng = random.Random(0)
    keys = list(tables["port_nodes"].facts)
    drawn = weighted_sample(keys, 999, [1.0] * len(keys), rng)
    assert sorted(drawn) == sorted(keys)


def test_lower_none_weight_monotonically_reduces_none_share(tables: dict[str, FactTable]) -> None:
    shares = []
    for weight in (1.0, 0.5, 0.2, 0.05):
        rng = random.Random(3)
        examples = list(
            generate_examples(tables, n_examples=1500, rng=rng, max_k=24, none_weight=weight)
        )
        shares.append(answer_shape_report(tables, examples)["none_share"])
    assert shares == sorted(shares, reverse=True), shares
    assert shares[0] > shares[-1] * 2


def test_down_weighting_does_not_break_coverage(tables: dict[str, FactTable]) -> None:
    """Exhaustive pass still emits every empty fact; only repetition is reduced."""
    rng = random.Random(3)
    examples = list(generate_exhaustive(tables, rng=rng, max_k=24)) + list(
        generate_examples(tables, n_examples=2000, rng=rng, max_k=24, none_weight=0.05)
    )
    report = coverage_report(tables, examples)
    assert report["fully_covered"]
    assert report["covered_facts"] == report["total_facts"]


def test_answer_shape_report_counts_add_up(tables: dict[str, FactTable]) -> None:
    rng = random.Random(5)
    examples = list(generate_examples(tables, n_examples=200, rng=rng, max_k=12))
    report = answer_shape_report(tables, examples)
    assert report["entries"] == sum(len(e.keys) for e in examples)
    assert report["empty"] <= report["entries"]
    assert 0.0 <= report["none_share"] <= 1.0
    for name, entry in report["by_table"].items():
        assert entry["empty"] <= entry["entries"]


def test_tables_with_no_empty_facts_are_unaffected(tables: dict[str, FactTable]) -> None:
    rng = random.Random(7)
    examples = [
        e
        for e in generate_examples(tables, n_examples=800, rng=rng, max_k=24, none_weight=0.01)
        if e.table == "node_neighbors"
    ]
    if examples:
        assert answer_shape_report(tables, examples)["by_table"]["node_neighbors"]["none_share"] == 0.0
