"""Loading the curated run for a replay, failing soft so replays still open."""

from collections.abc import Mapping

from ..transcript import paired_transcript_fingerprint
from .artifact import load_narrator_reasoning_artifact
from .schema import (
    RUN_SCHEMA,
    WINDOW_SCHEMA,
    CuratedRun,
    NarratorReasoningArtifactError,
    curated_run,
)

__all__ = ["load_paired_narrator_reasoning"]


def _artifact_error_collection(
    game_id: str,
    run: CuratedRun,
    paired_transcript: Mapping[str, object],
    error: Exception,
) -> dict[str, object]:
    return {
        "schema": WINDOW_SCHEMA,
        "artifact_schema": RUN_SCHEMA,
        "game_id": game_id,
        "model_id": run["model_id"],
        "model_label": run["model_label"],
        "generator_version": run["generator_version"],
        "transcript_sha256": paired_transcript_fingerprint(paired_transcript),
        "narrator": paired_transcript.get("narrator") or {},
        "artifact_error": str(error),
        "complete": False,
        "expected_window_count": 0,
        "loaded_window_count": 0,
        "ready_window_count": 0,
        "empty_window_count": 0,
        "error_window_count": 0,
        "paragraph_count": 0,
        "decision_count": 0,
        "anchor_count": 0,
        "anchors_by_replay_index": {},
        "expected_job_id_by_replay_index": {},
        "results_by_replay_index": {},
    }


def load_paired_narrator_reasoning(
    game_id: str,
    paired_transcript: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Load optional narrator reasoning, failing soft so replay loading still works."""
    run = curated_run(game_id)
    if run is None or paired_transcript is None:
        return None
    try:
        return load_narrator_reasoning_artifact(
            run["artifact_dir"],
            expected_game_id=str(game_id),
            expected_model_id=run["model_id"],
            expected_model_label=run["model_label"],
            expected_generator_version=run["generator_version"],
            paired_transcript=paired_transcript,
        )
    except NarratorReasoningArtifactError as exc:
        return _artifact_error_collection(str(game_id), run, paired_transcript, exc)
    except (AttributeError, KeyError, TypeError) as exc:
        schema_error = NarratorReasoningArtifactError(
            f"Malformed narrator-reasoning artifact schema: {exc}"
        )
        return _artifact_error_collection(
            str(game_id), run, paired_transcript, schema_error
        )
