"""Loading one curated action-diff artifact, phase by validated phase."""

import json
from collections.abc import Mapping
from pathlib import Path

from .indexing import _index_comparisons, _index_manifest
from .ledgers import read_side_ledgers, validate_plan_identity
from .overrides import build_setup_stages, validate_setup_overrides
from .readers import _read_json, _read_jsonl, _require_equal
from .repairs import validate_rationale_repairs
from .schema import MODEL_TRACE_SCHEMA
from .traces import build_traces

__all__ = ["load_model_trace_artifact"]


def load_model_trace_artifact(
    artifact_dir: Path,
    *,
    expected_game_id: str,
    expected_model_id: str,
    expected_player_id: int,
    expected_engine_color: str,
    model_label: str,
    narrator: Mapping[str, object] | None = None,
    archived_player_perspective: int | None = None,
    response_overrides_path: Path | None = None,
    response_attempts_path: Path | None = None,
    quality_path: Path | None = None,
    rationale_repairs_path: Path | None = None,
    rationale_repair_attempts_path: Path | None = None,
) -> dict[str, object]:
    """Load one action-diff artifact without exposing its human labels."""
    artifact_dir = Path(artifact_dir)
    plan = _read_json(artifact_dir / "plan.json")
    manifest, _ = _read_jsonl(artifact_dir / "decision_manifest.jsonl")
    responses, responses_partial = _read_jsonl(
        artifact_dir / "responses.jsonl",
        required=False,
        tolerate_incomplete_final_line=True,
    )
    comparisons, comparisons_partial = _read_jsonl(
        artifact_dir / "comparisons.jsonl",
        required=False,
        tolerate_incomplete_final_line=True,
    )
    ledgers = read_side_ledgers(
        response_overrides_path=response_overrides_path,
        response_attempts_path=response_attempts_path,
        quality_path=quality_path,
        rationale_repairs_path=rationale_repairs_path,
        rationale_repair_attempts_path=rationale_repair_attempts_path,
    )
    settings = validate_plan_identity(
        plan,
        expected_game_id=expected_game_id,
        expected_model_id=expected_model_id,
        expected_player_id=expected_player_id,
        expected_engine_color=expected_engine_color,
        narrator=narrator,
    )
    exact_records = _index_manifest(
        manifest,
        expected_player_id=expected_player_id,
        expected_engine_color=expected_engine_color,
    )
    planned_trace_count = plan.get("exact_decision_count")
    if planned_trace_count is not None:
        _require_equal("exact decision count", planned_trace_count, len(exact_records))

    comparisons_by_decision = _index_comparisons(comparisons, exact_records)
    base_latest_responses: dict[object, Mapping[str, object]] = {}
    for row in responses:
        if row.get("model_id") == expected_model_id:
            base_latest_responses[row.get("decision_id")] = row
    latest_responses = dict(base_latest_responses)

    attempt_rows = {
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in ledgers.response_attempts
    }
    setup_stages = build_setup_stages(exact_records)
    overrides = validate_setup_overrides(
        response_overrides=ledgers.response_overrides,
        exact_records=exact_records,
        attempt_rows=attempt_rows,
        expected_model_id=expected_model_id,
        setup_stages=setup_stages,
        base_latest_responses=base_latest_responses,
        latest_responses=latest_responses,
        decision_quality_warnings=ledgers.decision_quality_warnings,
    )
    rationale_repairs_by_decision = validate_rationale_repairs(
        rationale_repairs=ledgers.rationale_repairs,
        rationale_repair_attempts=ledgers.rationale_repair_attempts,
        override_rows_by_decision=overrides.rows_by_decision,
        expected_model_id=expected_model_id,
    )
    index = build_traces(
        plan=plan,
        exact_records=exact_records,
        latest_responses=latest_responses,
        comparisons_by_decision=comparisons_by_decision,
        override_decision_ids=overrides.decision_ids,
        rationale_repairs_by_decision=rationale_repairs_by_decision,
        decision_quality_warnings=ledgers.decision_quality_warnings,
        settings=settings,
        narrator=narrator,
        expected_game_id=expected_game_id,
        expected_model_id=expected_model_id,
        expected_player_id=expected_player_id,
        expected_engine_color=expected_engine_color,
        model_label=model_label,
    )

    expected_trace_count = len(exact_records)
    return {
        "schema": MODEL_TRACE_SCHEMA,
        "game_id": str(expected_game_id),
        "model_id": expected_model_id,
        "model_label": model_label,
        "player": {
            "username": (narrator or {}).get("username"),
            "colonist_color": expected_player_id,
            "engine_color": expected_engine_color,
        },
        "policy": {
            "context_version": settings.get("context_version"),
            "stateless_goals": settings["stateless_goals"],
            "allow_lookahead": settings["allow_lookahead"],
            "execute_model_actions": settings["execute_model_actions"],
        },
        "state_provenance": {
            "archived_player_perspective": archived_player_perspective,
            "target_matches_archive_perspective": (
                archived_player_perspective == expected_player_id
            ),
            "private_state_status": (
                "captured_player_view"
                if archived_player_perspective == expected_player_id
                else "reconstructed_non_capture_view"
            ),
        },
        "artifact_partial": (
            responses_partial
            or comparisons_partial
            or ledgers.overrides_partial
            or ledgers.attempts_partial
            or ledgers.repairs_partial
            or ledgers.repair_attempts_partial
        ),
        "artifact_error": None,
        "expected_trace_count": expected_trace_count,
        "loaded_trace_count": sum(
            len(traces)
            for traces in index.traces_by_available_replay_index.values()
        ),
        "ready_trace_count": index.ready_trace_count,
        "error_trace_count": index.error_trace_count,
        "override_trace_count": len(overrides.decision_ids),
        "rationale_repair_count": len(rationale_repairs_by_decision),
        "complete": index.ready_trace_count == expected_trace_count
        and index.error_trace_count == 0
        and not responses_partial
        and not comparisons_partial
        and not ledgers.overrides_partial
        and not ledgers.attempts_partial
        and not ledgers.repairs_partial
        and not ledgers.repair_attempts_partial,
        "decision_ids_by_available_replay_index": (
            index.decision_ids_by_available_replay_index
        ),
        "traces_by_available_replay_index": index.traces_by_available_replay_index,
    }
