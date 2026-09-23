"""Command line entry point for the leakage ledger and baseline build."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.load import (
    build_training_candidates,
    load_benchmark_games,
    load_eval_summaries,
    load_qwen_failures,
    load_raw_replay_ids,
)
from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.paths import (
    BASELINE_TABLE_MD,
    BENCHMARK_IDS_JSON,
    BENCHMARK_IDS_MD,
    METADATA_PATH,
    ROOT,
    TRAINING_CANDIDATES_JSON,
)
from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.render import (
    render_baseline_md,
    render_leakage_md,
)
from scripts.board_bench.shapes import JsonDict, read_json_object, write_json, write_text

# The pre-split module path stays the advertised program name and description.
PROG = "build_catan_board_bench_leakage_and_presft.py"
DESCRIPTION = "Build leakage-control and pre-SFT baseline artifacts for CatanBoardBench."

__all__ = ["DESCRIPTION", "PROG", "main"]


def main() -> None:
    argparse.ArgumentParser(prog=PROG, description=DESCRIPTION).parse_args()

    metadata = read_json_object(METADATA_PATH)
    benchmark_games, _ = load_benchmark_games()
    benchmark_ids = set(benchmark_games)
    ordered_ids: list[str] = sorted(benchmark_ids, key=int)
    raw_replay_ids = load_raw_replay_ids()
    training_candidates, source_counts = build_training_candidates(benchmark_ids)
    summaries = load_eval_summaries()
    qwen_failures = load_qwen_failures()

    ledger: JsonDict = {
        "schema": "catan_board_bench/leakage/v0",
        "benchmark": metadata.get("name", "CatanBoardBench-100"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_generated_at": metadata.get("generated_at"),
        "benchmark_game_ids": list(ordered_ids),
        "games": list(benchmark_games.values()),
        "raw_replay_ids": list(raw_replay_ids),
        "training_candidate_file": str(TRAINING_CANDIDATES_JSON.relative_to(ROOT)),
    }
    write_json(BENCHMARK_IDS_JSON, ledger)
    write_json(TRAINING_CANDIDATES_JSON, list(training_candidates))
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
