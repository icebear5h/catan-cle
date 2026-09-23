"""Indexing the manifest, the comparisons, and the reordered-row availability."""

from collections.abc import Mapping, Sequence
from typing import cast

from .readers import _require_equal
from .schema import ModelTraceArtifactError

__all__ = ["_canonical_availability", "_index_comparisons", "_index_manifest"]


def _index_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_player_id: int,
    expected_engine_color: str,
) -> dict[str, Mapping[str, object]]:
    exact_records: dict[str, Mapping[str, object]] = {}
    seen_decision_ids: set[str] = set()
    explicit_engine_indices: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        decision_id = row.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            raise ModelTraceArtifactError(
                f"decision_manifest.jsonl:{row_number} is missing decision_id"
            )
        if decision_id in seen_decision_ids:
            raise ModelTraceArtifactError(
                f"Duplicate decision_id {decision_id!r} in decision manifest"
            )
        seen_decision_ids.add(decision_id)
        if row.get("classification") != "exact":
            continue

        actor = row.get("actor")
        if not isinstance(actor, dict):
            raise ModelTraceArtifactError(
                f"Exact decision {decision_id} is missing actor attribution"
            )
        _require_equal(
            f"actor engine color for {decision_id}",
            actor.get("engine_color"),
            expected_engine_color,
        )
        engine_index = actor.get("engine_index")
        if not isinstance(engine_index, int) or isinstance(engine_index, bool):
            raise ModelTraceArtifactError(
                f"Exact decision {decision_id} has invalid actor engine_index"
            )
        available_actions = row.get("available_actions")
        if not isinstance(available_actions, list):
            raise ModelTraceArtifactError(
                f"Exact decision {decision_id} has invalid available_actions"
            )

        actor_player = actor.get("colonist_player")
        if actor_player is not None:
            _require_equal(
                f"actor player for {decision_id}",
                actor_player,
                expected_player_id,
            )
            explicit_engine_indices.add(engine_index)
        exact_records[decision_id] = row

    if exact_records and len(explicit_engine_indices) != 1:
        raise ModelTraceArtifactError(
            "Exact decisions do not resolve to one explicit target engine seat"
        )
    target_engine_index = next(iter(explicit_engine_indices), None)
    for decision_id, row in exact_records.items():
        actor = cast(Mapping[str, object], row["actor"])
        if actor.get("colonist_player") is not None:
            continue
        is_valid_owner_inference = (
            row.get("effective_action_type") == "BUILD_CITY"
            and actor.get("source") == "pre_action_building_owner"
            and actor.get("engine_index") == target_engine_index
        )
        if not is_valid_owner_inference:
            raise ModelTraceArtifactError(
                f"Exact decision {decision_id} has no validated Colonist actor"
            )
    return exact_records


def _index_comparisons(
    rows: Sequence[Mapping[str, object]],
    exact_records: Mapping[str, Mapping[str, object]],
) -> dict[object, Mapping[str, object]]:
    indexed: dict[object, Mapping[str, object]] = {}
    for row in rows:
        decision_id = row.get("decision_id")
        if decision_id not in exact_records:
            continue
        if decision_id in indexed:
            raise ModelTraceArtifactError(f"Duplicate decision_id {decision_id!r} in comparisons")
        models = row.get("models")
        if not isinstance(models, dict) or not all(
            isinstance(model_id, str) and isinstance(result, dict)
            for model_id, result in models.items()
        ):
            raise ModelTraceArtifactError(
                f"Decision {decision_id} comparison models must be an object"
            )
        indexed[decision_id] = row
    return indexed


def _canonical_availability(
    plan: Mapping[str, object],
) -> dict[int, tuple[int, str]]:
    """Map reordered canonical rows to their first safe viewer cursor.

    Colonist may emit STEAL before MOVE_ROBBER inside one raw event. The policy
    run swaps those rows so MOVE is queried first. Its pre-event trace is safe
    at the first viewer row. The dependent STEAL trace includes the recorded
    move in context and is therefore withheld until both original rows have
    been revealed.
    """
    availability: dict[int, tuple[int, str]] = {}
    canonicalizations = plan.get("canonicalizations", [])
    if not isinstance(canonicalizations, list):
        raise ModelTraceArtifactError("Model-trace plan canonicalizations must be a list")
    for change in canonicalizations:
        if not isinstance(change, dict):
            raise ModelTraceArtifactError("Model-trace plan canonicalization must be an object")
        canonical_indices = change.get("canonical_indices")
        source_indices = change.get("source_replay_indices")
        if (
            not isinstance(canonical_indices, list)
            or len(canonical_indices) != 2
            or not all(
                isinstance(index, int) and not isinstance(index, bool)
                for index in canonical_indices
            )
            or canonical_indices[1] != canonical_indices[0] + 1
            or not isinstance(source_indices, list)
            or len(source_indices) != 2
            or not all(
                isinstance(index, int) and not isinstance(index, bool) for index in source_indices
            )
            or sorted(source_indices) != canonical_indices
        ):
            raise ModelTraceArtifactError("Invalid same-event canonicalization in model-trace plan")
        first, second = canonical_indices
        if first in availability or second in availability:
            raise ModelTraceArtifactError(
                "Overlapping same-event canonicalizations in model-trace plan"
            )
        availability[first] = (first, "pre_action_reordered_event")
        availability[second] = (second + 1, "post_event_reveal")
    return availability
