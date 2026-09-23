"""Canonical filesystem paths for CatanBoardBench."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELATIVE_DATASETS_DIR = Path("evals/catan_board_bench/datasets")
DATASETS_DIR = PROJECT_ROOT / RELATIVE_DATASETS_DIR
LEGACY_RELATIVE_DATASETS_DIR = Path("data_pipeline/catan_board_bench/datasets")


def canonical_benchmark_reference(value: str | Path) -> Path:
    """Return the canonical repository-relative form of a benchmark path."""

    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            return path
    try:
        relative = path.relative_to(LEGACY_RELATIVE_DATASETS_DIR)
    except ValueError:
        return path
    return RELATIVE_DATASETS_DIR / relative


def resolve_benchmark_reference(value: str | Path) -> Path:
    """Resolve current paths and immutable pre-consolidation dataset references."""

    path = canonical_benchmark_reference(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
