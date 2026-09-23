"""Default locations, Colonist colour maps, and the build result record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from cle.game_engine.models.player import Color
from evals.catan_board_bench.paths import DATASETS_DIR, PROJECT_ROOT
from evals.json_types import JsonDict as JsonDict

DEFAULT_REPLAY_DIR = PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "replays"
DEFAULT_OUTPUT_DIR = DATASETS_DIR / "catan_board_bench_100"
DEFAULT_QUESTION_DIR = DEFAULT_OUTPUT_DIR / "questions"


COLONIST_COLOR_NAMES = {
    1: "red",
    2: "blue",
    3: "orange",
    4: "green",
    5: "black",
    6: "bronze",
    7: "silver",
    8: "gold",
    9: "white",
    10: "pink",
    11: "mystic_blue",
}

COLONIST_TO_ENGINE_COLOR = {
    1: Color.RED,
    2: Color.BLUE,
    3: Color.ORANGE,
    4: Color.GREEN,
    5: Color.BLACK,
    6: Color.BRONZE,
    7: Color.SILVER,
    8: Color.GOLD,
    9: Color.WHITE,
    10: Color.PINK,
    11: Color.MYSTIC_BLUE,
}

FALLBACK_ENGINE_COLORS = [
    Color.ORANGE,
    Color.BRONZE,
    Color.SILVER,
    Color.GOLD,
    Color.PINK,
    Color.MYSTIC_BLUE,
]


EdgeId = Tuple[int, int]


@dataclass(frozen=True)
class BenchmarkBuildResult:
    """Summary of a completed benchmark build."""

    output_dir: Path
    sample_count: int
    qa_count: int
    rendered_images: int
    replay_count: int
    metadata_path: Path
    manifest_path: Path
    questions_path: Path
    answer_key_path: Path

