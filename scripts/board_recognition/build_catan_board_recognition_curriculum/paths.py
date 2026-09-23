"""Project paths and repository-relative rendering."""

from __future__ import annotations

from pathlib import Path

from evals.catan_board_bench.paths import DATASETS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC_PATH = PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "curriculum.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "generated" / "board_recognition" / "curriculum"
DEFAULT_LEAKAGE_LEDGER = (
    DATASETS_DIR / "catan_board_bench_100" / "leakage" / "benchmark_game_ids.json"
)
DEFAULT_STYLE_PATH = PROJECT_ROOT / "configs" / "sft" / "renderer_style.json"

__all__ = [
    "DEFAULT_LEAKAGE_LEDGER",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SPEC_PATH",
    "DEFAULT_STYLE_PATH",
    "PROJECT_ROOT",
    "repository_relative",
    "resolve_project_path",
]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)
