"""Curated run registry, artifact locations, and the artifact error type."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECISION_EVAL_SCHEMA = "decision-spot-check-run-v1"
INDEX_SCHEMA = "decision-bucket-index-v1"


@dataclass(frozen=True)
class DecisionEvalRunConfig:
    id: str
    title: str
    description: str
    artifact_dir: Path
    model_id: str
    bucket_index_path: Path
    response_override_paths: tuple[Path, ...] = ()
    rationale_repair_paths: tuple[Path, ...] = ()


CURATED_DECISION_RUNS: dict[str, DecisionEvalRunConfig] = {
    "qwen3_8_27b_blue_242781000": DecisionEvalRunConfig(
        id="qwen3_8_27b_blue_242781000",
        title="Qwen 3.8 27B · BLUE · 242781000",
        description="Full causal model-versus-human action trace for the BLUE seat.",
        artifact_dir=PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817",
        model_id="qwen/qwen3.8-27b",
        bucket_index_path=PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "decision_buckets_v1.jsonl",
        response_override_paths=(
            PROJECT_ROOT
            / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
            / "action_selection_diff/qwen3_8_27b_blue_20260817"
            / "setup_strategy_overrides.jsonl",
        ),
        rationale_repair_paths=(
            PROJECT_ROOT
            / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
            / "action_selection_diff/qwen3_8_27b_blue_20260817"
            / "setup_rationale_repairs.jsonl",
        ),
    ),
}


class DecisionEvalArtifactError(ValueError):
    """Raised when a decision-eval artifact cannot be safely joined."""

