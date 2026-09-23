"""Prompt rendering, sampling cardinality, and coverage."""
import random

from sft.board.board_atlas import (
    NONE_ANSWER,
    FactTable,
    coverage_report,
    generate_examples,
    generate_exhaustive,
    parse_answer,
    render_example,
    sample_cardinality,
    score_atlas,
)


def test_empty_answers_render_as_none(tables: dict[str, FactTable]) -> None:
    table = tables["node_port"]
    empty = next(k for k, v in table.facts.items() if not v)
    _, answer = render_example(table, [empty])
    assert answer == f"{empty}: {NONE_ANSWER}"
    assert parse_answer(answer) == {empty: []}


def test_prompt_carries_the_queried_keys_in_order(tables: dict[str, FactTable]) -> None:
    table = tables["node_neighbors"]
    keys = ["<N19>", "<N03>", "<N00>"]
    prompt, answer = render_example(table, keys)
    for key in keys:
        assert key in prompt
    assert [line.split(":")[0] for line in answer.splitlines()] == keys


def test_sample_cardinality_is_in_range_and_favours_small_k() -> None:
    rng = random.Random(0)
    draws = [sample_cardinality(rng, 54, 24) for _ in range(5000)]
    assert min(draws) >= 1 and max(draws) <= 24
    # Single-key queries are the form traversal consumes, so they must not be rare.
    assert draws.count(1) / len(draws) > 0.10


def test_sample_cardinality_respects_small_key_sets() -> None:
    rng = random.Random(1)
    assert all(sample_cardinality(rng, 1, 24) == 1 for _ in range(50))
    assert all(sample_cardinality(rng, 3, 24) <= 3 for _ in range(200))


def test_exhaustive_generation_covers_every_fact(tables: dict[str, FactTable]) -> None:
    examples = list(generate_exhaustive(tables, rng=random.Random(7), max_k=24))
    report = coverage_report(tables, examples)
    assert report["fully_covered"]
    assert report["covered_facts"] == report["total_facts"]
    for name, entry in report["by_table"].items():
        assert entry["uncovered"] == [], name


def test_generated_examples_are_answerable_from_their_own_table(tables: dict[str, FactTable]) -> None:
    rng = random.Random(3)
    for example in generate_examples(tables, n_examples=200, rng=rng, max_k=12):
        table = tables[example.table]
        score = score_atlas(table, example.keys, example.answer)
        assert score.exact
        assert score.entry_accuracy == 1.0


def test_multi_token_keys_are_segmentable_in_the_prompt(tables: dict[str, FactTable]) -> None:
    """Keys like "<N19> NORTH" must not be space-joined into an ambiguous list."""
    for name in ("node_step", "node_distance", "node_path"):
        table = tables[name]
        keys = list(table.facts)[:3]
        prompt, _ = render_example(table, keys)
        query_line = prompt.splitlines()[0]
        assert query_line.count(";") == len(keys) - 1
        for key in keys:
            assert key in query_line
