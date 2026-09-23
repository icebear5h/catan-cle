"""Loading the curated run for a replay, failing soft for optional UI data."""

from collections.abc import Mapping

from .artifact import load_model_trace_artifact
from .readers import _require_equal
from .schema import (
    MODEL_TRACE_SCHEMA,
    CuratedTraceRun,
    ModelTraceArtifactError,
    curated_trace_run,
)

__all__ = ["load_paired_model_traces"]


def _artifact_error_collection(
    game_id: str,
    run: CuratedTraceRun,
    narrator: Mapping[str, object] | None,
    error: ModelTraceArtifactError,
) -> dict[str, object]:
    return {
        "schema": MODEL_TRACE_SCHEMA,
        "game_id": str(game_id),
        "model_id": run["model_id"],
        "model_label": run["model_label"],
        "player": {
            "username": (narrator or {}).get("username"),
            "colonist_color": run["target_player_id"],
            "engine_color": run["target_engine_color"],
        },
        "policy": {
            "context_version": None,
            "stateless_goals": None,
            "allow_lookahead": None,
            "execute_model_actions": None,
        },
        "state_provenance": {
            "archived_player_perspective": run["archived_player_perspective"],
            "target_matches_archive_perspective": (
                run["archived_player_perspective"] == run["target_player_id"]
            ),
            "private_state_status": "reconstructed_non_capture_view",
        },
        "artifact_partial": False,
        "artifact_error": str(error),
        "expected_trace_count": 0,
        "loaded_trace_count": 0,
        "ready_trace_count": 0,
        "error_trace_count": 0,
        "override_trace_count": 0,
        "rationale_repair_count": 0,
        "complete": False,
        "decision_ids_by_available_replay_index": {},
        "traces_by_available_replay_index": {},
    }


def load_paired_model_traces(
    game_id: str,
    narrator: Mapping[str, object] | None = None,
    archived_player_perspective: int | None = None,
) -> dict[str, object] | None:
    """Load the narrator-seat model run, failing soft for optional UI data."""
    run = curated_trace_run(game_id)
    if run is None:
        return None
    try:
        if archived_player_perspective is not None:
            _require_equal(
                "archived player perspective",
                archived_player_perspective,
                run["archived_player_perspective"],
            )
        return load_model_trace_artifact(
            run["artifact_dir"],
            expected_game_id=str(game_id),
            expected_model_id=run["model_id"],
            expected_player_id=run["target_player_id"],
            expected_engine_color=run["target_engine_color"],
            model_label=run["model_label"],
            narrator=narrator,
            archived_player_perspective=run["archived_player_perspective"],
            response_overrides_path=run.get("response_overrides_path"),
            response_attempts_path=run.get("response_attempts_path"),
            quality_path=run.get("quality_path"),
            rationale_repairs_path=run.get("rationale_repairs_path"),
            rationale_repair_attempts_path=run.get(
                "rationale_repair_attempts_path"
            ),
        )
    except ModelTraceArtifactError as exc:
        return _artifact_error_collection(str(game_id), run, narrator, exc)
    except (AttributeError, KeyError, TypeError) as exc:
        schema_error = ModelTraceArtifactError(f"Malformed model-trace artifact schema: {exc}")
        return _artifact_error_collection(str(game_id), run, narrator, schema_error)
