"""Validating the setup-strategy overrides against their attempts ledger."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .readers import _require_equal
from .schema import ModelTraceArtifactError

__all__ = ["OverrideIndex", "build_setup_stages", "validate_setup_overrides"]


@dataclass(frozen=True)
class OverrideIndex:
    """Which decisions a setup override replaced, and with which row."""

    decision_ids: set[object]
    rows_by_decision: dict[object, Mapping[str, object]]


def build_setup_stages(
    exact_records: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    """Name each exact placement decision by its turn order in the setup phase."""
    setup_stage_counts = {"BUILD_SETTLEMENT": 0, "BUILD_ROAD": 0}
    expected_setup_stages: dict[str, str] = {}
    for decision_id, record in sorted(
        exact_records.items(),
        key=lambda item: cast(int, item[1].get("replay_index", -1)),
    ):
        if record.get("phase") != "initial_placement":
            continue
        action_type = record.get("effective_action_type")
        if action_type not in setup_stage_counts:
            raise ModelTraceArtifactError(
                f"Unexpected exact setup action type {action_type!r}"
            )
        ordinal = setup_stage_counts[action_type]
        setup_stage_counts[action_type] += 1
        stage_prefix = "first" if ordinal == 0 else "second"
        stage_suffix = "settlement" if action_type == "BUILD_SETTLEMENT" else "road"
        expected_setup_stages[decision_id] = f"{stage_prefix}_{stage_suffix}"
    return expected_setup_stages


def validate_setup_overrides(
    *,
    response_overrides: Sequence[Mapping[str, object]],
    exact_records: Mapping[str, Mapping[str, object]],
    attempt_rows: set[str],
    expected_model_id: str,
    setup_stages: Mapping[str, str],
    base_latest_responses: Mapping[object, Mapping[str, object]],
    latest_responses: dict[object, Mapping[str, object]],
    decision_quality_warnings: Mapping[str, list[str]],
) -> OverrideIndex:
    """Accept only overrides the immutable attempts ledger already recorded."""
    override_decision_ids: set[object] = set()
    override_rows_by_decision: dict[object, Mapping[str, object]] = {}
    for row in response_overrides:
        decision_id = row.get("decision_id")
        record = exact_records.get(cast(str, decision_id))
        if record is None or record.get("phase") != "initial_placement":
            raise ModelTraceArtifactError(
                f"Setup override {decision_id!r} is not an exact setup decision"
            )
        if decision_id in override_decision_ids:
            raise ModelTraceArtifactError(
                f"Duplicate setup override for decision {decision_id}"
            )
        serialized_override = json.dumps(
            row, sort_keys=True, separators=(",", ":")
        )
        if serialized_override not in attempt_rows:
            raise ModelTraceArtifactError(
                f"Setup override {decision_id} is not in the attempts ledger"
            )
        _require_equal(
            f"setup override model for {decision_id}",
            row.get("model_id"),
            expected_model_id,
        )
        _require_equal(
            f"setup override replay row for {decision_id}",
            row.get("replay_index"),
            record.get("replay_index"),
        )
        result = row.get("result")
        if row.get("error") or not isinstance(result, dict):
            raise ModelTraceArtifactError(
                f"Setup override {decision_id} is not a successful response"
            )
        if not isinstance(result.get("setup_strategy_version"), str):
            raise ModelTraceArtifactError(
                f"Setup override {decision_id} lacks strategy provenance"
            )
        _require_equal(
            f"setup override result replay row for {decision_id}",
            result.get("replay_index"),
            record.get("replay_index"),
        )
        _require_equal(
            f"setup override stage for {decision_id}",
            result.get("setup_stage"),
            setup_stages.get(cast(str, decision_id)),
        )
        activity_window = result.get("activity_window")
        recent_activity = result.get("recent_activity")
        if not isinstance(activity_window, dict) or not isinstance(recent_activity, list):
            raise ModelTraceArtifactError(
                f"Setup override {decision_id} has invalid activity provenance"
            )
        if (
            activity_window.get("end_replay_index") != record.get("replay_index")
            or activity_window.get("row_count") != len(recent_activity)
            or not isinstance(activity_window.get("start_replay_index"), int)
            or activity_window["start_replay_index"] > activity_window["end_replay_index"]
        ):
            raise ModelTraceArtifactError(
                f"Setup override {decision_id} has an invalid activity cutoff"
            )
        base_result = (base_latest_responses.get(decision_id) or {}).get("result")
        if not isinstance(base_result, dict):
            raise ModelTraceArtifactError(
                f"Setup override {decision_id} has no base causal response"
            )
        _require_equal(
            f"setup override activity window for {decision_id}",
            activity_window,
            base_result.get("activity_window"),
        )
        _require_equal(
            f"setup override recent activity for {decision_id}",
            recent_activity,
            base_result.get("recent_activity"),
        )
        action_index = result.get("action_index")
        available_actions = result.get("available_actions")
        if (
            not isinstance(action_index, int)
            or not isinstance(available_actions, list)
            or not 0 <= action_index < len(available_actions)
        ):
            raise ModelTraceArtifactError(
                f"Setup override {decision_id} lacks a valid selected action"
            )
        if not all(
            isinstance(action, dict) and action.get("index") == index
            for index, action in enumerate(available_actions)
        ):
            raise ModelTraceArtifactError(
                f"Setup override {decision_id} has invalid menu ordering"
            )
        manifest_actions = [
            (action.get("index"), action.get("action"))
            for action in cast(
                Sequence[Mapping[str, object]], record.get("available_actions", [])
            )
        ]
        override_actions = [
            (action.get("index"), action.get("action"))
            for action in available_actions
        ]
        if override_actions != manifest_actions:
            raise ModelTraceArtifactError(
                f"Setup override {decision_id} menu order or identity changed"
            )
        _require_equal(
            f"setup override top-level selected index for {decision_id}",
            row.get("model_action_index"),
            action_index,
        )
        _require_equal(
            f"setup override human index for {decision_id}",
            row.get("human_action_index"),
            cast(Mapping[str, object], record.get("human", {})).get("action_index"),
        )
        selected_action = available_actions[action_index]
        _require_equal(
            f"setup override selected engine action for {decision_id}",
            result.get("action"),
            selected_action.get("action"),
        )
        _require_equal(
            f"setup override selected description for {decision_id}",
            result.get("action_description"),
            selected_action.get("description"),
        )
        override_decision_ids.add(decision_id)
        override_rows_by_decision[decision_id] = row
        latest_responses[decision_id] = row

    if not set(decision_quality_warnings).issubset(override_decision_ids):
        raise ModelTraceArtifactError(
            "Setup quality warnings reference a non-override decision"
        )
    return OverrideIndex(
        decision_ids=override_decision_ids,
        rows_by_decision=override_rows_by_decision,
    )
