"""Exposing assembled reasoning only at its strict causal cursor."""

from collections.abc import Iterable, Mapping
from typing import cast

from .readers import _listed
from .schema import WINDOW_SCHEMA

__all__ = ["build_paired_narrator_reasoning_window"]


def build_paired_narrator_reasoning_window(
    replay_data: Mapping[str, object], replay_index: int
) -> dict[str, object] | None:
    """Expose subject-grouped assembly only at its strict causal cursor."""
    collection = cast(
        Mapping[str, object] | None, replay_data.get("paired_narrator_reasoning")
    )
    if not collection:
        return None

    anchors = cast(
        Mapping[int, Mapping[str, object]],
        collection.get("anchors_by_replay_index", {}),
    )
    current_anchor: Mapping[str, object] = anchors.get(replay_index) or {}
    base = {
        "schema": collection.get("schema", WINDOW_SCHEMA),
        "game_id": collection.get("game_id"),
        "replay_index": replay_index,
        "model_id": collection.get("model_id"),
        "model_label": collection.get("model_label"),
        "generator_version": collection.get("generator_version"),
        "narrator": collection.get("narrator") or {},
        "artifact_error": collection.get("artifact_error"),
        "complete": bool(collection.get("complete", False)),
        "strict_causal": True,
        "decision_count": collection.get("decision_count", 0),
        "anchor_count": collection.get("anchor_count", 0),
        "expected_window_count": collection.get("expected_window_count", 0),
        "loaded_window_count": collection.get("loaded_window_count", 0),
        "anchor_kind": current_anchor.get("anchor_kind"),
        "decision_ids": _listed(current_anchor.get("decision_ids") or []),
    }
    empty_payload: dict[str, object] = {
        "error": None,
        "paragraphs": [],
        "history_paragraphs": [],
        "history_groups": [],
    }
    if collection.get("artifact_error"):
        return {**base, **empty_payload, "status": "artifact_error"}
    if replay_index < 0:
        return {**base, **empty_payload, "status": "unavailable"}

    results_by_replay_index = cast(
        Mapping[int, Mapping[str, object]],
        collection.get("results_by_replay_index", {}),
    )
    available_results = [
        (available_index, result)
        for available_index, result in sorted(results_by_replay_index.items())
        if available_index <= replay_index
    ]
    history_groups = [
        {
            "subject_replay_index": available_index,
            "available_replay_index": available_index,
            "anchor_kind": result.get("anchor_kind"),
            "decision_ids": _listed(result.get("decision_ids") or []),
            "paragraphs": _listed(result.get("paragraphs") or []),
        }
        for available_index, result in available_results
        if result.get("paragraphs")
    ]
    history_paragraphs = [
        paragraph
        for group in history_groups
        for paragraph in cast(Iterable[Mapping[str, object]], group["paragraphs"])
    ]
    history_paragraphs.sort(
        key=lambda paragraph: (
            cast(int, paragraph["available_replay_index"]),
            cast(float, paragraph["start_s"]),
            cast(float, paragraph["end_s"]),
            cast(str, paragraph["paragraph_id"]),
        )
    )
    history_payload = {
        "history_paragraphs": history_paragraphs,
        "history_groups": history_groups,
    }
    result = results_by_replay_index.get(replay_index)
    if result is not None:
        return {
            **base,
            **history_payload,
            "status": result["status"],
            "recorded_at": result.get("recorded_at"),
            "error": result.get("error"),
            "anchor_kind": result.get("anchor_kind"),
            "decision_ids": _listed(result.get("decision_ids") or []),
            "paragraphs": _listed(result.get("paragraphs") or []),
        }

    expected = cast(
        Mapping[int, str], collection.get("expected_job_id_by_replay_index", {})
    )
    if replay_index in expected:
        return {
            **base,
            **history_payload,
            "status": "pending",
            "error": None,
            "paragraphs": [],
        }

    total_events = replay_data.get("total_events", 0)
    status = (
        "complete"
        if isinstance(total_events, int) and replay_index >= total_events
        else "no_commentary"
    )
    return {
        **base,
        **history_payload,
        "status": status,
        "error": None,
        "paragraphs": [],
    }
