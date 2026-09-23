"""The trace contract: the schema name and the curated runs behind it."""

from pathlib import Path
from typing import TypedDict

__all__ = [
    "MODEL_TRACE_SCHEMA",
    "PROJECT_ROOT",
    "CuratedTraceRun",
    "ModelTraceArtifactError",
    "curated_trace_run",
]

# One level deeper than the old module, so the repo root is four parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_TRACE_SCHEMA = "paired-replay-model-trace-v1"


class CuratedTraceRun(TypedDict):
    """One curated action-diff run and every ledger it is validated against."""

    artifact_dir: Path
    model_id: str
    model_label: str
    target_player_id: int
    target_engine_color: str
    archived_player_perspective: int
    response_overrides_path: Path
    response_attempts_path: Path
    quality_path: Path
    rationale_repairs_path: Path
    rationale_repair_attempts_path: Path


_CURATED_TRACE_RUNS: dict[str, CuratedTraceRun] = {
    "242781000": {
        "artifact_dir": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817",
        "model_id": "qwen/qwen3.8-27b",
        "model_label": "Qwen 3.8 27B",
        "target_player_id": 2,
        "target_engine_color": "BLUE",
        "archived_player_perspective": 5,
        "response_overrides_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_strategy_overrides.jsonl",
        "response_attempts_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_strategy_attempts.jsonl",
        "quality_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_strategy_quality.json",
        "rationale_repairs_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_rationale_repairs.jsonl",
        "rationale_repair_attempts_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_rationale_repair_attempts.jsonl",
    }
}


class ModelTraceArtifactError(ValueError):
    """Raised when a curated model-trace artifact violates its contract."""


def curated_trace_run(game_id: str) -> CuratedTraceRun | None:
    """Return the curated model-trace run for a replay, if one is curated."""
    return _CURATED_TRACE_RUNS.get(str(game_id))
