"""Curated, cursor-safe model traces for paired replay transcripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_TRACE_SCHEMA = "paired-replay-model-trace-v1"

_CURATED_TRACE_RUNS = {
    "242781000": {
        "artifact_dir": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817",
        "model_id": "qwen/qwen3.8-27b",
        "model_label": "Qwen 3.8 27B",
        "target_player_id": 2,
        "target_engine_color": "BLUE",
        "archived_player_perspective": 5,
        "response_overrides_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_strategy_overrides.jsonl",
        "response_attempts_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_strategy_attempts.jsonl",
        "quality_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_strategy_quality.json",
        "rationale_repairs_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_rationale_repairs.jsonl",
        "rationale_repair_attempts_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "setup_rationale_repair_attempts.jsonl",
    }
}


class ModelTraceArtifactError(ValueError):
    """Raised when a curated model-trace artifact violates its contract."""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ModelTraceArtifactError(f"Missing model-trace artifact: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelTraceArtifactError(
            f"Could not read model-trace artifact {path.name}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ModelTraceArtifactError(f"Model-trace artifact {path.name} is not an object")
    return data


def _read_jsonl(
    path: Path,
    *,
    required: bool = True,
    tolerate_incomplete_final_line: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Read JSONL, optionally ignoring only a concurrently appended tail."""
    if not path.exists():
        if required:
            raise ModelTraceArtifactError(f"Missing model-trace artifact: {path.name}")
        return [], False

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ModelTraceArtifactError(
            f"Could not read model-trace artifact {path.name}: {exc}"
        ) from exc

    rows = []
    lines = text.splitlines(keepends=True)
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            is_incomplete_tail = (
                tolerate_incomplete_final_line
                and line_index == len(lines) - 1
                and not line.endswith(("\n", "\r"))
            )
            if is_incomplete_tail:
                return rows, True
            raise ModelTraceArtifactError(
                f"Could not read model-trace artifact {path.name}: "
                f"invalid JSON on line {line_index + 1}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ModelTraceArtifactError(f"{path.name}:{line_index + 1} is not an object")
        rows.append(row)
    return rows, False


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ModelTraceArtifactError(
            f"Model-trace {label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _index_manifest(
    rows: List[Dict[str, Any]],
    *,
    expected_player_id: int,
    expected_engine_color: str,
) -> Dict[str, Dict[str, Any]]:
    exact_records = {}
    seen_decision_ids = set()
    explicit_engine_indices = set()
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
        actor = row["actor"]
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
    rows: List[Dict[str, Any]], exact_records: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    indexed = {}
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
    plan: Dict[str, Any],
) -> Dict[int, Tuple[int, str]]:
    """Map reordered canonical rows to their first safe viewer cursor.

    Colonist may emit STEAL before MOVE_ROBBER inside one raw event. The policy
    run swaps those rows so MOVE is queried first. Its pre-event trace is safe
    at the first viewer row. The dependent STEAL trace includes the recorded
    move in context and is therefore withheld until both original rows have
    been revealed.
    """
    availability = {}
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


def load_model_trace_artifact(
    artifact_dir: Path,
    *,
    expected_game_id: str,
    expected_model_id: str,
    expected_player_id: int,
    expected_engine_color: str,
    model_label: str,
    narrator: Optional[Dict[str, Any]] = None,
    archived_player_perspective: Optional[int] = None,
    response_overrides_path: Optional[Path] = None,
    response_attempts_path: Optional[Path] = None,
    quality_path: Optional[Path] = None,
    rationale_repairs_path: Optional[Path] = None,
    rationale_repair_attempts_path: Optional[Path] = None,
) -> Dict[str, Any]:
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
    response_overrides = []
    overrides_partial = False
    rationale_repairs = []
    repairs_partial = False
    rationale_repair_attempts = []
    repair_attempts_partial = False
    decision_quality_warnings: Dict[str, List[str]] = {}
    response_attempts = []
    attempts_partial = False
    if response_overrides_path is not None:
        response_overrides, overrides_partial = _read_jsonl(
            Path(response_overrides_path),
            tolerate_incomplete_final_line=False,
        )
        if response_attempts_path is None:
            raise ModelTraceArtifactError(
                "Setup overrides require an immutable attempts ledger"
            )
        response_attempts, attempts_partial = _read_jsonl(
            Path(response_attempts_path),
            tolerate_incomplete_final_line=False,
        )
        if quality_path is not None:
            quality = _read_json(Path(quality_path))
            _require_equal(
                "setup quality schema",
                quality.get("schema"),
                "setup-strategy-quality-v1",
            )
            warnings = quality.get("decision_warnings")
            if not isinstance(warnings, dict) or not all(
                isinstance(decision_id, str)
                and isinstance(items, list)
                and items
                and all(isinstance(item, str) and item for item in items)
                for decision_id, items in warnings.items()
            ):
                raise ModelTraceArtifactError(
                    "Setup strategy quality warnings are malformed"
                )
            decision_quality_warnings = warnings
        if rationale_repairs_path is not None:
            if rationale_repair_attempts_path is None:
                raise ModelTraceArtifactError(
                    "Selected rationale repairs require an attempts ledger"
                )
            rationale_repairs, repairs_partial = _read_jsonl(
                Path(rationale_repairs_path),
                tolerate_incomplete_final_line=False,
            )
            rationale_repair_attempts, repair_attempts_partial = _read_jsonl(
                Path(rationale_repair_attempts_path),
                tolerate_incomplete_final_line=False,
            )

    _require_equal("game", str(plan.get("game_id")), str(expected_game_id))
    _require_equal("models", plan.get("models"), [expected_model_id])
    _require_equal("target player", plan.get("target_player_id"), expected_player_id)
    _require_equal(
        "target engine color",
        plan.get("target_engine_color"),
        expected_engine_color,
    )
    if narrator:
        _require_equal(
            "narrator player",
            narrator.get("colonist_color"),
            expected_player_id,
        )

    settings = plan.get("settings")
    if not isinstance(settings, dict):
        raise ModelTraceArtifactError("Model-trace plan is missing settings")
    _require_equal("stateless goals policy", settings.get("stateless_goals"), True)
    _require_equal("lookahead policy", settings.get("allow_lookahead"), False)
    _require_equal(
        "model-action execution policy",
        settings.get("execute_model_actions"),
        False,
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
    base_latest_responses = {}
    for row in responses:
        if row.get("model_id") == expected_model_id:
            base_latest_responses[row.get("decision_id")] = row
    latest_responses = dict(base_latest_responses)

    attempt_rows = {
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in response_attempts
    }
    setup_stage_counts = {"BUILD_SETTLEMENT": 0, "BUILD_ROAD": 0}
    expected_setup_stages = {}
    for decision_id, record in sorted(
        exact_records.items(), key=lambda item: item[1].get("replay_index", -1)
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

    override_decision_ids = set()
    override_rows_by_decision = {}
    for row in response_overrides:
        decision_id = row.get("decision_id")
        record = exact_records.get(decision_id)
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
            expected_setup_stages.get(decision_id),
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
            for action in record.get("available_actions", [])
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
            record.get("human", {}).get("action_index"),
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

    repair_attempt_rows = {
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rationale_repair_attempts
    }
    rationale_repairs_by_decision = {}
    for repair in rationale_repairs:
        decision_id = repair.get("decision_id")
        source = override_rows_by_decision.get(decision_id)
        if source is None:
            raise ModelTraceArtifactError(
                f"Rationale repair {decision_id!r} has no setup override"
            )
        if decision_id in rationale_repairs_by_decision:
            raise ModelTraceArtifactError(
                f"Duplicate rationale repair for decision {decision_id}"
            )
        serialized_repair = json.dumps(
            repair, sort_keys=True, separators=(",", ":")
        )
        if serialized_repair not in repair_attempt_rows:
            raise ModelTraceArtifactError(
                f"Rationale repair {decision_id} is not in its attempts ledger"
            )
        source_result = source["result"]
        source_sha256 = hashlib.sha256(
            json.dumps(
                source, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        context_sha256 = hashlib.sha256(
            source_result["context_prompt"].encode()
        ).hexdigest()
        _require_equal(
            f"rationale repair source timestamp for {decision_id}",
            repair.get("source_recorded_at"),
            source.get("recorded_at"),
        )
        _require_equal(
            f"rationale repair source hash for {decision_id}",
            repair.get("source_response_sha256"),
            source_sha256,
        )
        _require_equal(
            f"rationale repair context hash for {decision_id}",
            repair.get("context_prompt_sha256"),
            context_sha256,
        )
        _require_equal(
            f"rationale repair model for {decision_id}",
            repair.get("model_id"),
            expected_model_id,
        )
        _require_equal(
            f"rationale repair selected index for {decision_id}",
            repair.get("selected_action_index"),
            source_result.get("action_index"),
        )
        _require_equal(
            f"rationale repair selected action for {decision_id}",
            repair.get("selected_action"),
            source_result.get("action"),
        )
        _require_equal(
            f"rationale repair selected description for {decision_id}",
            repair.get("selected_action_description"),
            source_result.get("action_description"),
        )
        if (
            repair.get("parse_error")
            or not isinstance(repair.get("goals"), str)
            or not repair.get("goals")
            or not isinstance(repair.get("reasoning"), str)
            or not repair.get("reasoning")
        ):
            raise ModelTraceArtifactError(
                f"Rationale repair {decision_id} is not a valid completed repair"
            )
        rationale_repairs_by_decision[decision_id] = repair

    reordered_availability = _canonical_availability(plan)
    decision_ids_by_available_replay_index: Dict[int, List[str]] = {}
    traces_by_available_replay_index: Dict[int, List[Dict[str, Any]]] = {}
    seen_source_indices = set()
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
        trace = {
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
            "legal_action_count": len(record["available_actions"]),
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
            or overrides_partial
            or attempts_partial
            or repairs_partial
            or repair_attempts_partial
        ),
        "artifact_error": None,
        "expected_trace_count": expected_trace_count,
        "loaded_trace_count": sum(
            len(traces) for traces in traces_by_available_replay_index.values()
        ),
        "ready_trace_count": ready_trace_count,
        "error_trace_count": error_trace_count,
        "override_trace_count": len(override_decision_ids),
        "rationale_repair_count": len(rationale_repairs_by_decision),
        "complete": ready_trace_count == expected_trace_count
        and error_trace_count == 0
        and not responses_partial
        and not comparisons_partial
        and not overrides_partial
        and not attempts_partial
        and not repairs_partial
        and not repair_attempts_partial,
        "decision_ids_by_available_replay_index": (decision_ids_by_available_replay_index),
        "traces_by_available_replay_index": traces_by_available_replay_index,
    }


def _artifact_error_collection(
    game_id: str,
    run: Dict[str, Any],
    narrator: Optional[Dict[str, Any]],
    error: ModelTraceArtifactError,
) -> Dict[str, Any]:
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
    narrator: Optional[Dict[str, Any]] = None,
    archived_player_perspective: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Load the narrator-seat model run, failing soft for optional UI data."""
    run = _CURATED_TRACE_RUNS.get(str(game_id))
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


def build_paired_model_trace_window(
    replay_data: Dict[str, Any], replay_index: int
) -> Optional[Dict[str, Any]]:
    """Expose only traces whose source context is safe at this cursor."""
    collection = replay_data.get("paired_model_traces")
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

    expected_ids = collection.get("decision_ids_by_available_replay_index", {}).get(
        replay_index, []
    )
    traces = list(collection.get("traces_by_available_replay_index", {}).get(replay_index, []))
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
