"""Markdown rendering for the leakage ledger and the baseline table."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.load import pct
from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.paths import (
    QWEN_KEY,
    ROOT,
    TRAINING_CANDIDATES_JSON,
)
from scripts.board_bench.shapes import JsonDict, markdown_table, obj, text, values

__all__ = ["render_baseline_md", "render_leakage_md"]


def render_leakage_md(
    benchmark_games: dict[str, JsonDict],
    metadata: JsonDict,
    raw_replay_ids: list[str],
    candidate_count: int,
    source_counts: dict[str, int],
) -> str:
    rows: list[list[object]] = []
    for game in benchmark_games.values():
        replay_files = [str(item) for item in values(game["replay_files"], "replay_files")]
        replay_steps = [str(step) for step in values(game["replay_steps"], "replay_steps")]
        rows.append(
            [
                game["game_id"],
                game["sample_count"],
                ", ".join(replay_files) or "-",
                ", ".join(replay_steps[:8]) or "-",
            ]
        )

    index_rows: list[list[object]] = [
        [name, count] for name, count in sorted(source_counts.items())
    ]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return (
        "\n\n".join(
            [
                "# CatanBoardBench Leakage Ledger",
                f"Generated: {now}",
                (
                    "Rule: these game IDs are evaluation-only. Do not use their replay JSON, "
                    "rendered board images, contracts, QA rows, or derived text in SFT, CPT, "
                    "validation, prompt tuning, or data selection."
                ),
                (
                    f"Benchmark: {metadata.get('name', 'CatanBoardBench-100')} with "
                    f"{metadata.get('sample_count')} samples and {metadata.get('qa_count')} QA rows, "
                    f"generated at {metadata.get('generated_at')}."
                ),
                "## Held-Out Benchmark Games",
                markdown_table(["Game ID", "Samples", "Replay file", "Replay steps"], rows),
                "## Local Replay Inventory",
                (
                    "Raw replay IDs currently present locally: "
                    + (", ".join(raw_replay_ids) if raw_replay_ids else "none")
                ),
                (
                    "Current status: every local raw replay ID overlaps the benchmark held-out set. "
                    "Training data must come from newly pulled candidate games or another non-overlapping source."
                ),
                "## Candidate Index Inputs",
                markdown_table(["Index file", "Rows"], index_rows),
                (
                    f"Wrote {candidate_count} unique training candidate games to "
                    f"`{TRAINING_CANDIDATES_JSON.relative_to(ROOT)}` after excluding benchmark IDs."
                ),
            ]
        )
        + "\n"
    )


def _overall_rows(summaries: list[JsonDict]) -> list[list[object]]:
    rows: list[list[object]] = []
    for summary in summaries:
        metrics = obj(summary["metrics"], "summary metrics")
        categories = [str(item) for item in values(summary["categories"], "categories")]
        rows.append(
            [
                summary["run_id"],
                summary["model_key"],
                metrics.get("requests", "-"),
                metrics.get("attempted", "-"),
                pct(metrics.get("exact_accuracy")),
                pct(metrics.get("component_accuracy")),
                ", ".join(categories),
            ]
        )
    return rows


def _qwen_rows(summaries: list[JsonDict]) -> list[list[object]]:
    rows: list[list[object]] = []
    for summary in summaries:
        if summary["model_key"] != QWEN_KEY:
            continue
        metrics = obj(summary["metrics"], "summary metrics")
        by_category = obj(metrics.get("categories", {}), "metric categories")
        for category, category_metrics in sorted(by_category.items()):
            entry = obj(category_metrics, "category metrics")
            rows.append(
                [
                    summary["run_id"],
                    category,
                    entry.get("attempted", "-"),
                    pct(entry.get("exact_accuracy")),
                    pct(entry.get("component_accuracy")),
                    entry.get("errors", 0),
                ]
            )
    return rows


def _failure_rows(qwen_failures: list[JsonDict]) -> list[list[object]]:
    rows: list[list[object]] = []
    failures_by_category: dict[str, list[JsonDict]] = defaultdict(list)
    for failure in qwen_failures:
        failures_by_category[text(failure["category"], "failure category")].append(failure)
    for category in sorted(failures_by_category):
        bucket = failures_by_category[category]
        examples = bucket[:3]
        expected_seen = Counter(str(item["expected"]) for item in bucket)
        response_seen = Counter(str(item["response"]) for item in bucket)
        rows.append(
            [
                category,
                len(bucket),
                expected_seen.most_common(1)[0][0],
                response_seen.most_common(1)[0][0],
                "; ".join(text(item["question_id"], "question_id") for item in examples),
            ]
        )
    return rows


def render_baseline_md(summaries: list[JsonDict], qwen_failures: list[JsonDict]) -> str:
    return (
        "\n\n".join(
            [
                "# CatanBoardBench-100 OpenRouter Baseline",
                (
                    "Scope: stored OpenRouter eval runs under "
                    "`artifacts/runs/catan_board_bench/catan_board_bench_100/openrouter`. "
                    "These are the frozen baseline numbers before any Catan-specific tuning."
                ),
                "## Overall Runs",
                markdown_table(
                    ["Run", "Model", "Requests", "Attempted", "Exact", "Component", "Categories"],
                    _overall_rows(summaries),
                ),
                "## Qwen3-VL-8B Category Breakdown",
                markdown_table(
                    ["Run", "Category", "Attempted", "Exact", "Component", "Errors"],
                    _qwen_rows(summaries),
                ),
                "## Qwen3-VL-8B Failure Buckets",
                markdown_table(
                    [
                        "Category",
                        "Misses",
                        "Common expected",
                        "Common response",
                        "Example question IDs",
                    ],
                    _failure_rows(qwen_failures),
                ),
                (
                    "Interpretation target for SFT: exact accuracy should rise first on atlas-bound local "
                    "grounding tasks (`tile_resource_number`, `robber_tile`, `edge_road_owner`, "
                    "`port_type_nodes`) without only improving sentinel-heavy categories such as "
                    "`longest_road_holder` or empty-node answers."
                ),
            ]
        )
        + "\n"
    )
