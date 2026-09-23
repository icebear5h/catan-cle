"""Archive schema, default run locations, and the bundle record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inspect_ai import Task
from inspect_ai.model import Model

from evals.catan_board_bench.paths import PROJECT_ROOT
from evals.json_types import JsonDict as JsonDict

ARCHIVE_IMPORT_SCHEMA = "catan-inspect-archive/v1"
DEFAULT_INSPECT_LOG_DIR = PROJECT_ROOT / "artifacts/runs/inspect/catan_archive_v1"
DEFAULT_STRICT_VISION_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/catan_board_bench/strict_60_unified_20260825/image/novita"
)
DEFAULT_STRICT_VISION_RUNS: dict[str, Path] = {
    "deepseek_v4_flash_vision_exp": DEFAULT_STRICT_VISION_ROOT
    / "deepseek_v4_flash_vision_exp",
    "gemma4_31b": DEFAULT_STRICT_VISION_ROOT / "gemma4_31b",
    "glm4_6v": DEFAULT_STRICT_VISION_ROOT / "glm4_6v",
    "qwen3_8_max": DEFAULT_STRICT_VISION_ROOT / "qwen3_8_max",
}
DEFAULT_POLICY_RUN_ID = "qwen3_8_27b_blue_242781000"
MODEL_SELECTION_REPORT = (
    "reports/model_selection/2026-08-31-local-27b-80b-models-for-catan.md"
)


@dataclass(frozen=True)
class InspectArchiveBundle:
    """One validated archived run ready for provider-free Inspect evaluation."""

    archive_id: str
    task: Task
    model: Model
    model_id: str
    source_dir: Path
    input_mode: str
    source_records: int
    imported_records: int
    expected_metrics: JsonDict

