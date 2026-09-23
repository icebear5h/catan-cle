"""Build leakage-control and pre-SFT baseline artifacts for CatanBoardBench.

Paths, loading, markdown rendering, and the CLI live in sibling modules.
Every pre-split name stays importable at this path."""

from __future__ import annotations

from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.cli import main
from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.load import (
    build_training_candidates,
    load_benchmark_games,
    load_eval_summaries,
    load_qwen_failures,
    load_raw_replay_ids,
    norm_game_id,
    pct,
)
from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.paths import (
    BASELINE_TABLE_MD,
    BENCH_DIR,
    BENCHMARK_IDS_JSON,
    BENCHMARK_IDS_MD,
    INDEX_DIR,
    INDEX_FILES,
    LEAKAGE_DIR,
    MANIFEST_PATH,
    METADATA_PATH,
    OPENROUTER_EVAL_DIR,
    QWEN_KEY,
    RAW_REPLAY_DIR,
    RESULTS_DIR,
    ROOT,
    TRAINING_CANDIDATES_JSON,
)
from scripts.board_bench.builders.build_catan_board_bench_leakage_and_presft.render import (
    render_baseline_md,
    render_leakage_md,
)
from scripts.board_bench.shapes import read_json, read_jsonl, write_json, write_text

__all__ = [
    "BASELINE_TABLE_MD",
    "BENCHMARK_IDS_JSON",
    "BENCHMARK_IDS_MD",
    "BENCH_DIR",
    "INDEX_DIR",
    "INDEX_FILES",
    "LEAKAGE_DIR",
    "MANIFEST_PATH",
    "METADATA_PATH",
    "OPENROUTER_EVAL_DIR",
    "QWEN_KEY",
    "RAW_REPLAY_DIR",
    "RESULTS_DIR",
    "ROOT",
    "TRAINING_CANDIDATES_JSON",
    "build_training_candidates",
    "load_benchmark_games",
    "load_eval_summaries",
    "load_qwen_failures",
    "load_raw_replay_ids",
    "main",
    "norm_game_id",
    "pct",
    "read_json",
    "read_jsonl",
    "render_baseline_md",
    "render_leakage_md",
    "write_json",
    "write_text",
]
