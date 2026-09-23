"""The run contract: schema names, valid kinds, and the curated runs."""

from pathlib import Path
from typing import TypedDict

__all__ = [
    "ANCHOR_KINDS",
    "PARAGRAPH_KINDS",
    "PROJECT_ROOT",
    "RESULT_SCHEMA",
    "RUN_SCHEMA",
    "WINDOW_SCHEMA",
    "CuratedRun",
    "NarratorReasoningArtifactError",
    "curated_run",
]

# One level deeper than the old module, so the repo root is four parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUN_SCHEMA = "narrator-observation-assembly-run-v1"
RESULT_SCHEMA = "narrator-observation-assembly-result-v1"
WINDOW_SCHEMA = "paired-narrator-reasoning-v2"
PARAGRAPH_KINDS = frozenset(
    {
        "decision_reasoning",
        "board_observation",
        "opponent_assessment",
        "reaction",
        "reflection",
    }
)
ANCHOR_KINDS = frozenset({"decision", "observation", "complete"})


class CuratedRun(TypedDict):
    """One persisted narrator-reasoning run and the identity it must declare."""

    artifact_dir: Path
    model_id: str
    model_label: str
    generator_version: str


_CURATED_REASONING_RUNS: dict[str, CuratedRun] = {
    "242781000": {
        "artifact_dir": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "narrator_reasoning/gpt_5_6_sol_observation_v3_20260820",
        "model_id": "openai/gpt-5.6-sol",
        "model_label": "GPT-5.6",
        "generator_version": "narrator-observation-assembly-v1",
    }
}


class NarratorReasoningArtifactError(ValueError):
    """Raised when a persisted narrator-reasoning run violates its contract."""


def curated_run(game_id: str) -> CuratedRun | None:
    """Return the curated reasoning run for a replay, or None when there is none."""
    return _CURATED_REASONING_RUNS.get(str(game_id))
