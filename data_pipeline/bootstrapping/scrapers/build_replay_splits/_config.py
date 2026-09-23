"""Paths, colour names, and step plans for the Colonist replay split builder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EXCLUDE_IDS = (
    ROOT / "evals/catan_board_bench/datasets/catan_board_bench_100/leakage/benchmark_game_ids.json"
)
DEFAULT_RAW_REPLAY_DIR = ROOT / "artifacts" / "raw" / "colonist" / "replays"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "manifests" / "colonist" / "splits"

COLONIST_COLOR_NAMES = {
    1: "RED",
    2: "BLUE",
    3: "ORANGE",
    4: "GREEN",
    5: "BLACK",
    6: "BRONZE",
    7: "SILVER",
    8: "GOLD",
    9: "WHITE",
    10: "PINK",
    11: "MYSTIC_BLUE",
}

EARLY_MID_STEP_PLAN = {
    "early": [0.08, 0.18, 0.28],
    "mid": [0.38, 0.50],
}
LATE_HARD_STEP_PLAN = {
    "late_hard": [0.62, 0.74, 0.86],
}

__all__ = [
    "COLONIST_COLOR_NAMES",
    "DEFAULT_EXCLUDE_IDS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RAW_REPLAY_DIR",
    "EARLY_MID_STEP_PLAN",
    "LATE_HARD_STEP_PLAN",
    "ROOT",
]
