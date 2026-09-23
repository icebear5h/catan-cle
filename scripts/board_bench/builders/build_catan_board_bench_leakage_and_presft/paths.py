"""Benchmark, replay, and report paths for the leakage ledger build."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
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
]
