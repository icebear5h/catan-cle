"""Build leakage-control and pre-SFT baseline artifacts for CatanBoardBench."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "evals/catan_board_bench/datasets/catan_board_bench_100"
MANIFEST_PATH = BENCH_DIR / "manifest.jsonl"
METADATA_PATH = BENCH_DIR / "metadata.json"
OPENROUTER_EVAL_DIR = (
    ROOT / "artifacts" / "runs" / "catan_board_bench" / "catan_board_bench_100" / "openrouter"
)
RAW_REPLAY_DIR = ROOT / "artifacts" / "raw" / "colonist" / "replays"
INDEX_DIR = ROOT / "artifacts" / "raw" / "colonist" / "indexes"

LEAKAGE_DIR = BENCH_DIR / "leakage"
RESULTS_DIR = ROOT / "reports" / "catan_board_bench"

BENCHMARK_IDS_JSON = LEAKAGE_DIR / "benchmark_game_ids.json"
BENCHMARK_IDS_MD = LEAKAGE_DIR / "benchmark_game_ids.md"
TRAINING_CANDIDATES_JSON = INDEX_DIR / "4p_games_training_candidates.json"
BASELINE_TABLE_MD = RESULTS_DIR / "catan_board_bench_100_openrouter_baseline.md"

INDEX_FILES = [
    INDEX_DIR / "4p_games_top100.json",
    INDEX_DIR / "4p_games_current.json",
    INDEX_DIR / "4p_games_me_all.json",
]

QWEN_KEY = "qwen3-vl-8b"


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def norm_game_id(value: Any) -> str:
    text = str(value)
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else text


def pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def load_benchmark_games() -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    samples = read_jsonl(MANIFEST_PATH)
    sample_counts: Counter[str] = Counter()
    steps_by_game: dict[str, list[int]] = defaultdict(list)
    replay_files_by_game: dict[str, set[str]] = defaultdict(set)

    for sample in samples:
        source = sample.get("source", {})
        game_id = norm_game_id(source.get("game_id") or source.get("replay_id") or "")
        if not game_id:
            continue
        sample_counts[game_id] += 1
        replay_step = source.get("replay_step")
        if replay_step is not None:
            steps_by_game[game_id].append(int(replay_step))
        replay_file = source.get("replay_file") or source.get("path")
        if replay_file:
            replay_files_by_game[game_id].add(str(replay_file))

    metadata = read_json(METADATA_PATH)
    for key, count in metadata.get("counts", {}).items():
        if not key.startswith("replay:"):
            continue
        replay_file = key.removeprefix("replay:")
        game_id = norm_game_id(replay_file)
        sample_counts.setdefault(game_id, int(count))
        replay_files_by_game[game_id].add(replay_file)

    games: dict[str, dict[str, Any]] = {}
    for game_id in sorted(sample_counts, key=int):
        games[game_id] = {
            "game_id": game_id,
            "sample_count": sample_counts[game_id],
            "replay_steps": sorted(set(steps_by_game.get(game_id, []))),
            "replay_files": sorted(replay_files_by_game.get(game_id, [])),
        }
    return games, sample_counts


def load_raw_replay_ids() -> list[str]:
    if not RAW_REPLAY_DIR.exists():
        return []
    return sorted({norm_game_id(path.name) for path in RAW_REPLAY_DIR.glob("*.json")}, key=int)


def build_training_candidates(
    benchmark_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    for index_path in INDEX_FILES:
        if not index_path.exists():
            continue
        index_rows = read_json(index_path)
        source_counts[index_path.name] = len(index_rows)
        for row in index_rows:
            game_id = norm_game_id(row.get("game_id") or row.get("id") or "")
            if not game_id or game_id in benchmark_ids or game_id in seen:
                continue
            seen.add(game_id)
            candidate = dict(row)
            candidate["game_id"] = game_id
            candidate["source_index"] = index_path.name
            candidates.append(candidate)

    return candidates, source_counts


def load_eval_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(OPENROUTER_EVAL_DIR.glob("*/summary.json")):
        run_dir = summary_path.parent
        plan_path = run_dir / "plan.json"
        summary = read_json(summary_path)
        plan = read_json(plan_path) if plan_path.exists() else {}
        for model_key, metrics in sorted(summary.get("models", {}).items()):
            summaries.append(
                {
                    "run_id": run_dir.name,
                    "model_key": model_key,
                    "model_id": plan.get("models", {}).get(model_key, model_key),
                    "question_count": plan.get("question_count", metrics.get("requests")),
                    "categories": plan.get("categories") or sorted(metrics.get("categories", {})),
                    "metrics": metrics,
                }
            )
    return summaries


def load_qwen_failures() -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for response_path in sorted(OPENROUTER_EVAL_DIR.glob("*/responses.jsonl")):
        run_id = response_path.parent.name
        for row in read_jsonl(response_path):
            if row.get("model_key") != QWEN_KEY:
                continue
            score = row.get("score") or {}
            if score.get("correct") is True:
                continue
            failures.append(
                {
                    "run_id": run_id,
                    "category": row.get("category"),
                    "question_id": row.get("question_id"),
                    "expected": row.get("expected"),
                    "response": row.get("response"),
                    "component_accuracy": score.get("component_accuracy"),
                }
            )
    return failures


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_leakage_md(
    benchmark_games: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    raw_replay_ids: list[str],
    candidate_count: int,
    source_counts: dict[str, int],
) -> str:
    rows = []
    for game in benchmark_games.values():
        rows.append(
            [
                game["game_id"],
                game["sample_count"],
                ", ".join(game["replay_files"]) or "-",
                ", ".join(str(step) for step in game["replay_steps"][:8]) or "-",
            ]
        )

    index_rows = [[name, count] for name, count in sorted(source_counts.items())]
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
                table(["Game ID", "Samples", "Replay file", "Replay steps"], rows),
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
                table(["Index file", "Rows"], index_rows),
                (
                    f"Wrote {candidate_count} unique training candidate games to "
                    f"`{TRAINING_CANDIDATES_JSON.relative_to(ROOT)}` after excluding benchmark IDs."
                ),
            ]
        )
        + "\n"
    )


def render_baseline_md(summaries: list[dict[str, Any]], qwen_failures: list[dict[str, Any]]) -> str:
    overall_rows = []
    for summary in summaries:
        metrics = summary["metrics"]
        overall_rows.append(
            [
                summary["run_id"],
                summary["model_key"],
                metrics.get("requests", "-"),
                metrics.get("attempted", "-"),
                pct(metrics.get("exact_accuracy")),
                pct(metrics.get("component_accuracy")),
                ", ".join(summary["categories"]),
            ]
        )

    qwen_rows = []
    for summary in summaries:
        if summary["model_key"] != QWEN_KEY:
            continue
        for category, metrics in sorted(summary["metrics"].get("categories", {}).items()):
            qwen_rows.append(
                [
                    summary["run_id"],
                    category,
                    metrics.get("attempted", "-"),
                    pct(metrics.get("exact_accuracy")),
                    pct(metrics.get("component_accuracy")),
                    metrics.get("errors", 0),
                ]
            )

    failure_rows = []
    failures_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for failure in qwen_failures:
        failures_by_category[failure["category"]].append(failure)
    for category in sorted(failures_by_category):
        examples = failures_by_category[category][:3]
        expected_seen = Counter(str(item["expected"]) for item in failures_by_category[category])
        response_seen = Counter(str(item["response"]) for item in failures_by_category[category])
        failure_rows.append(
            [
                category,
                len(failures_by_category[category]),
                expected_seen.most_common(1)[0][0],
                response_seen.most_common(1)[0][0],
                "; ".join(item["question_id"] for item in examples),
            ]
        )

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
                table(
                    ["Run", "Model", "Requests", "Attempted", "Exact", "Component", "Categories"],
                    overall_rows,
                ),
                "## Qwen3-VL-8B Category Breakdown",
                table(["Run", "Category", "Attempted", "Exact", "Component", "Errors"], qwen_rows),
                "## Qwen3-VL-8B Failure Buckets",
                table(
                    [
                        "Category",
                        "Misses",
                        "Common expected",
                        "Common response",
                        "Example question IDs",
                    ],
                    failure_rows,
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


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    metadata = read_json(METADATA_PATH)
    benchmark_games, _ = load_benchmark_games()
    benchmark_ids = set(benchmark_games)
    raw_replay_ids = load_raw_replay_ids()
    training_candidates, source_counts = build_training_candidates(benchmark_ids)
    summaries = load_eval_summaries()
    qwen_failures = load_qwen_failures()

    write_json(
        BENCHMARK_IDS_JSON,
        {
            "schema": "catan_board_bench/leakage/v0",
            "benchmark": metadata.get("name", "CatanBoardBench-100"),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "benchmark_generated_at": metadata.get("generated_at"),
            "benchmark_game_ids": sorted(benchmark_ids, key=int),
            "games": list(benchmark_games.values()),
            "raw_replay_ids": raw_replay_ids,
            "training_candidate_file": str(TRAINING_CANDIDATES_JSON.relative_to(ROOT)),
        },
    )
    write_json(TRAINING_CANDIDATES_JSON, training_candidates)
    write_text(
        BENCHMARK_IDS_MD,
        render_leakage_md(
            benchmark_games=benchmark_games,
            metadata=metadata,
            raw_replay_ids=raw_replay_ids,
            candidate_count=len(training_candidates),
            source_counts=source_counts,
        ),
    )
    write_text(BASELINE_TABLE_MD, render_baseline_md(summaries, qwen_failures))

    print(f"benchmark_game_ids={len(benchmark_ids)}")
    print(f"training_candidates={len(training_candidates)}")
    print(f"eval_model_rows={len(summaries)}")
    print(BENCHMARK_IDS_MD.relative_to(ROOT))
    print(TRAINING_CANDIDATES_JSON.relative_to(ROOT))
    print(BASELINE_TABLE_MD.relative_to(ROOT))


if __name__ == "__main__":
    main()
