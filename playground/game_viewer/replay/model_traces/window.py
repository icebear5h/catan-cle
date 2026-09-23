"""Exposing only the traces whose source context is safe at this cursor."""

from collections.abc import Mapping, Sequence
from typing import cast

from .schema import MODEL_TRACE_SCHEMA

__all__ = ["build_paired_model_trace_window"]


def build_paired_model_trace_window(
    replay_data: Mapping[str, object], replay_index: int
) -> dict[str, object] | None:
    """Expose only traces whose source context is safe at this cursor."""
    collection = cast(
        Mapping[str, object] | None, replay_data.get("paired_model_traces")
    )
    if not collection:
        return None

    base = {
        "schema": collection.get("schema", MODEL_TRACE_SCHEMA),
        "game_id": collection.get("game_id"),
        "replay_index": replay_index,
        "model_id": collection.get("model_id"),
        "model_label": collection.get("model_label"),
        "player": collection.get("player", {}),
        "policy": collection.get("policy", {}),
        "state_provenance": collection.get("state_provenance", {}),
        "artifact_partial": bool(collection.get("artifact_partial", False)),
        "artifact_error": collection.get("artifact_error"),
        "complete": bool(collection.get("complete", False)),
        "expected_trace_count": collection.get("expected_trace_count", 0),
        "ready_trace_count": collection.get("ready_trace_count", 0),
    }

    if collection.get("artifact_error"):
        return {
            **base,
            "status": "artifact_error",
            "pending_trace_count": 0,
            "traces": [],
        }
    if replay_index < 0:
        return {
            **base,
            "status": "unavailable",
            "pending_trace_count": 0,
            "traces": [],
        }

    expected_ids = cast(
        Mapping[int, Sequence[object]],
        collection.get("decision_ids_by_available_replay_index", {}),
    ).get(replay_index, [])
    traces = list(
        cast(
            Mapping[int, Sequence[Mapping[str, object]]],
            collection.get("traces_by_available_replay_index", {}),
        ).get(replay_index, [])
    )
    present_ids = {trace.get("decision_id") for trace in traces}
    pending_trace_count = sum(decision_id not in present_ids for decision_id in expected_ids)

    if traces:
        status = "error" if all(trace.get("status") == "error" for trace in traces) else "ready"
        return {
            **base,
            "status": status,
            "pending_trace_count": pending_trace_count,
            "traces": traces,
        }
    if pending_trace_count:
        return {
            **base,
            "status": "pending",
            "pending_trace_count": pending_trace_count,
            "traces": [],
        }

    total_events = replay_data.get("total_events", 0)
    if isinstance(total_events, int) and replay_index >= total_events:
        status = "complete"
    else:
        status = "no_decision"
    return {
        **base,
        "status": status,
        "pending_trace_count": 0,
        "traces": [],
    }
