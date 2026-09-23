import json
from pathlib import Path
from typing import Any

from scripts.board_bench.run.summarize_catan_board_bench_runs import (
    build_comparison,
    exact_two_sided_binomial_p,
)


def write_run(root: Path, outcomes: list[bool], image_tokens: int) -> Path:
    root.mkdir()
    rows = []
    for index, correct in enumerate(outcomes):
        rows.append(
            {
                "question_id": f"q{index}",
                "category": "node_occupancy",
                "expected": "EMPTY",
                "model_id": "model",
                "served_model": "model",
                "provider": "provider",
                "error": None,
                "latency_ms": 100 + index,
                "score": {
                    "correct": correct,
                    "component_correct": int(correct),
                    "component_total": 1,
                },
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                    "prompt_tokens_details": {"image_tokens": image_tokens},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
    (root / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def test_build_comparison_aggregates_usage_and_paired_flips(tmp_path: Path) -> None:
    first = write_run(tmp_path / "first", [True, False, False, True], 2)
    second = write_run(tmp_path / "second", [True, True, False, False], 4)

    comparison: Any = build_comparison([("first", first), ("second", second)], [("first", "second")])

    assert comparison["runs"]["first"]["exact_correct"] == 2
    assert comparison["runs"]["second"]["usage"]["image_tokens"] == 16
    assert comparison["runs"]["second"]["component_accuracy"] == 0.5
    assert comparison["paired_exact"]["first_vs_second"] == {
        "first": "first",
        "second": "second",
        "pairs": 4,
        "both_correct": 1,
        "first_only": 1,
        "second_only": 1,
        "both_wrong": 1,
        "first_exact": 2,
        "second_exact": 2,
        "second_minus_first_percentage_points": 0.0,
        "exact_mcnemar_binomial_p": 1.0,
    }


def test_exact_two_sided_binomial_p_handles_no_discordance() -> None:
    assert exact_two_sided_binomial_p(0, 0) is None
    assert exact_two_sided_binomial_p(0, 5) == 0.0625
