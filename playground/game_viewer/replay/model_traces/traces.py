"""Turning validated records and responses into the viewer's trace rows."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .indexing import _canonical_availability
from .readers import _require_equal
from .schema import MODEL_TRACE_SCHEMA, ModelTraceArtifactError

__all__ = ["TraceIndex", "build_traces"]


@dataclass(frozen=True)
class TraceIndex:
    """Traces grouped by the first viewer cursor that may safely show them."""

    decision_ids_by_available_replay_index: dict[int, list[str]]
    traces_by_available_replay_index: dict[int, list[dict[str, object]]]
    ready_trace_count: int
    error_trace_count: int


def build_traces(
    *,
    plan: Mapping[str, object],
    exact_records: Mapping[str, Mapping[str, object]],
    latest_responses: Mapping[object, Mapping[str, object]],
    comparisons_by_decision: Mapping[object, Mapping[str, object]],
    override_decision_ids: set[object],
    rationale_repairs_by_decision: Mapping[object, Mapping[str, object]],
    decision_quality_warnings: Mapping[str, list[str]],
    settings: Mapping[str, object],
    narrator: Mapping[str, object] | None,
    expected_game_id: str,
    expected_model_id: str,
    expected_player_id: int,
    expected_engine_color: str,
    model_label: str,
) -> TraceIndex:
    """Build one trace per exact decision, grouped by its safe viewer cursor."""
    reordered_availability = _canonical_availability(plan)
    decision_ids_by_available_replay_index: dict[int, list[str]] = {}
    traces_by_available_replay_index: dict[int, list[dict[str, object]]] = {}
    seen_source_indices: set[int] = set()
    ready_trace_count = 0
    error_trace_count = 0

    for decision_id, record in exact_records.items():
        canonical_replay_index = record.get("replay_index")
        source_replay_index = record.get("source_replay_index")
        if not isinstance(canonical_replay_index, int) or isinstance(canonical_replay_index, bool):
            raise ModelTraceArtifactError(f"Decision {decision_id} has invalid replay_index")
        if not isinstance(source_replay_index, int) or isinstance(source_replay_index, bool):
            raise ModelTraceArtifactError(f"Decision {decision_id} has invalid source_replay_index")
        if source_replay_index in seen_source_indices:
            raise ModelTraceArtifactError(
                f"Duplicate source replay row {source_replay_index} in model traces"
            )
        seen_source_indices.add(source_replay_index)

        available_replay_index, alignment = reordered_availability.get(
            canonical_replay_index,
            (source_replay_index, "pre_action"),
        )
        if alignment == "pre_action_reordered_event":
            _require_equal(
                f"reordered prerequisite type for {decision_id}",
                record.get("effective_action_type"),
                "MOVE_ROBBER",
            )
        elif alignment == "post_event_reveal":
            _require_equal(
                f"reordered dependent type for {decision_id}",
                record.get("effective_action_type"),
                "STEAL",
            )
        decision_ids_by_available_replay_index.setdefault(available_replay_index, []).append(
            decision_id
        )

        response = latest_responses.get(decision_id)
        if response is None:
            continue
        _require_equal("response game", str(response.get("game_id")), str(expected_game_id))
        _require_equal(
            "response replay row",
            response.get("replay_index"),
            canonical_replay_index,
        )

        result = response.get("result")
        response_error = response.get("error")
        if result is not None and not isinstance(result, dict):
            raise ModelTraceArtifactError(f"Decision {decision_id} has a non-object result")
        if isinstance(result, dict):
            _require_equal(
                "response player color",
                result.get("player_color"),
                expected_engine_color,
            )

        comparison = comparisons_by_decision.get(decision_id, {})
        comparison_models = comparison.get("models", {})
        if not isinstance(comparison_models, dict):
            raise ModelTraceArtifactError(
                f"Decision {decision_id} comparison models must be an object"
            )
        model_comparison = comparison_models.get(expected_model_id, {})
        if not isinstance(model_comparison, dict):
            raise ModelTraceArtifactError(
                f"Decision {decision_id} model comparison must be an object"
            )
        selected_index = model_comparison.get("action_index")
        selected_action = model_comparison.get("action")
        selected_description = model_comparison.get("description")
        parse_warning = model_comparison.get("parse_error")
        if (
            decision_id in override_decision_ids
            or not model_comparison
            or not model_comparison.get("response_present", False)
        ) and isinstance(result, dict):
            selected_index = result.get("action_index")
            selected_action = result.get("action")
            selected_description = result.get("action_description")
            parse_warning = result.get("parse_error")

        trace_status = "error" if response_error or result is None else "ready"
        if trace_status == "ready":
            ready_trace_count += 1
        else:
            error_trace_count += 1

        result = result or {}
        rationale_repair = rationale_repairs_by_decision.get(decision_id)
        displayed_goals = (
            rationale_repair["goals"] if rationale_repair else result.get("goals") or ""
        )
        displayed_reasoning = (
            rationale_repair["reasoning"]
            if rationale_repair
            else result.get("reasoning") or ""
        )
        displayed_message = (
            rationale_repair.get("message") or ""
            if rationale_repair
            else result.get("message") or ""
        )
        trace: dict[str, object] = {
            "schema": MODEL_TRACE_SCHEMA,
            "status": trace_status,
            "decision_id": decision_id,
            "game_id": str(expected_game_id),
            "replay_index": available_replay_index,
            "available_replay_index": available_replay_index,
            "source_replay_index": source_replay_index,
            "canonical_replay_index": canonical_replay_index,
            "alignment": alignment,
            "trace_source": (
                "setup_strategy_override"
                if decision_id in override_decision_ids
                else "base_action_diff"
            ),
            "model_id": expected_model_id,
            "model": result.get("model") or expected_model_id,
            "model_label": model_label,
            "player": {
                "username": (narrator or {}).get("username"),
                "colonist_color": expected_player_id,
                "engine_color": expected_engine_color,
            },
            "context_version": result.get("context_version") or settings.get("context_version"),
            "setup_strategy_version": result.get("setup_strategy_version"),
            "setup_stage": result.get("setup_stage"),
            "generation_max_tokens": result.get("generation_max_tokens"),
            "phase": record.get("phase"),
            "forced": bool(record.get("forced", False)),
            "legal_action_count": len(
                cast(Sequence[object], record["available_actions"])
            ),
            "selection": {
                "index": selected_index,
                "action": selected_action,
                "description": selected_description,
            },
            "goals": displayed_goals,
            "reasoning": displayed_reasoning,
            "message": displayed_message,
            "reasoning_source": (
                "qwen_self_review" if rationale_repair else "qwen_action_response"
            ),
            "reasoning_recorded_at": (
                rationale_repair.get("recorded_at") if rationale_repair else None
            ),
            "draft_goals": result.get("goals") if rationale_repair else None,
            "draft_reasoning": result.get("reasoning") if rationale_repair else None,
            "parse_warning": parse_warning,
            "quality_warnings": decision_quality_warnings.get(decision_id, []),
            "response_truncated": bool(result.get("response_truncated", False)),
            "latency_ms": result.get("latency_ms"),
            "recorded_at": response.get("recorded_at"),
            "error": response_error,
        }
        traces_by_available_replay_index.setdefault(available_replay_index, []).append(trace)
    return TraceIndex(
        decision_ids_by_available_replay_index=decision_ids_by_available_replay_index,
        traces_by_available_replay_index=traces_by_available_replay_index,
        ready_trace_count=ready_trace_count,
        error_trace_count=error_trace_count,
    )
