"""Filtering, compaction, and per-decision detail views."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from evals.decision_spot_checks.config import DecisionEvalArtifactError
from evals.decision_spot_checks.shapes import decision_model, dict_or_empty, list_or_empty
from evals.json_types import JsonDict, JsonValue, as_dict, as_list


def filter_decisions(
    run: Mapping[str, JsonValue],
    *,
    bucket_id: str | None = None,
    stage: str | None = None,
    agreement: str | None = None,
    classification: str | None = None,
) -> list[JsonDict]:
    decisions = [
        as_dict(row, "run decision")
        for row in list_or_empty(run.get("decisions"), "run decisions")
    ]
    if bucket_id and bucket_id != "all":
        decisions = [
            row
            for row in decisions
            if bucket_id in as_list(row.get("bucket_ids", []), "bucket_ids")
        ]
    if stage and stage != "all":
        decisions = [row for row in decisions if row.get("stage") == stage]
    if classification and classification != "all":
        decisions = [row for row in decisions if row.get("classification") == classification]
    if agreement and agreement != "all":
        if agreement == "agree":
            decisions = [row for row in decisions if decision_model(row).get("agreement") is True]
        elif agreement == "disagree":
            decisions = [row for row in decisions if decision_model(row).get("agreement") is False]
        elif agreement == "unscored":
            decisions = [row for row in decisions if not decision_model(row).get("response_present")]
        else:
            raise ValueError("agreement must be all, agree, disagree, or unscored")
    return decisions


def compact_decision(decision: Mapping[str, JsonValue]) -> JsonDict:
    return {
        "decision_id": decision["decision_id"],
        "game_id": decision["game_id"],
        "replay_index": decision["replay_index"],
        "source_event_index": decision.get("source_event_index"),
        "action_type": decision["action_type"],
        "classification": decision["classification"],
        "forced": decision["forced"],
        "stage": decision["stage"],
        "critical": decision["critical"],
        "bucket_ids": decision["bucket_ids"],
        "episode_ids": decision["episode_ids"],
        "actor": decision["actor"],
        "human": decision["human"],
        "model": {
            key: decision_model(decision).get(key)
            for key in (
                "model_id",
                "response_present",
                "action_index",
                "action",
                "description",
                "agreement",
                "error",
                "parse_error",
            )
        },
    }


def decision_detail(run: Mapping[str, JsonValue], decision_id: str) -> JsonDict | None:
    for item in as_list(run.get("decisions", []), "run decisions"):
        decision = as_dict(item, "run decision")
        if decision["decision_id"] == decision_id:
            return decision
    return None


def _joined_decision(
    manifest: Mapping[str, JsonValue],
    bucket_row: Mapping[str, JsonValue],
    response: Mapping[str, JsonValue],
    comparison: Mapping[str, JsonValue] | None,
    model_id: str,
) -> JsonDict:
    assignment = dict(as_dict(bucket_row["bucket_assignment"], "bucket_assignment"))
    model_comparison: JsonDict = {}
    if isinstance(comparison, Mapping):
        models = comparison.get("models")
        model_entry = models.get(model_id) if isinstance(models, Mapping) else None
        if isinstance(model_entry, Mapping):
            model_comparison = dict(model_entry)
    raw_result = response.get("result")
    result: JsonDict = raw_result if isinstance(raw_result, Mapping) else {}
    response_present = bool(response)
    model_index = result.get("action_index", response.get("model_action_index"))
    model_action = result.get("action", model_comparison.get("action"))
    model_description = result.get(
        "action_description",
        model_comparison.get("description"),
    )
    agreement = response.get("agreement") if response_present else None
    if agreement is None and model_comparison:
        agreement = model_comparison.get("agreement")

    return {
        "decision_id": str(manifest["decision_id"]),
        "game_id": str(manifest.get("game_id") or bucket_row.get("game_id") or ""),
        "replay_index": manifest.get("replay_index"),
        "source_replay_index": manifest.get("source_replay_index"),
        "source_event_index": manifest.get("source_event_index"),
        "action_type": str(
            manifest.get("effective_action_type")
            or manifest.get("source_action_type")
            or ""
        ),
        "classification": str(manifest.get("classification") or "unknown"),
        "reason": manifest.get("reason"),
        "phase": manifest.get("phase"),
        "forced": bool(manifest.get("forced", False)),
        "stage": assignment.get("stage", "unknown"),
        "critical": bool(assignment.get("critical", False)),
        "bucket_ids": list(list_or_empty(assignment.get("bucket_ids"), "bucket_ids")),
        "episode_ids": dict(dict_or_empty(assignment.get("episode_ids"), "episode_ids")),
        "bucket_evidence": dict(dict_or_empty(assignment.get("evidence"), "evidence")),
        "state_features": bucket_row.get("state_features"),
        "actor": dict(dict_or_empty(manifest.get("actor"), "actor")),
        "human": deepcopy(manifest.get("human")),
        "available_actions": deepcopy(manifest.get("available_actions") or []),
        "model": {
            "model_id": model_id,
            "response_present": response_present,
            "response_source": response.get("response_source"),
            "recorded_at": response.get("recorded_at"),
            "action_index": model_index,
            "action": model_action,
            "description": model_description,
            "agreement": agreement,
            "error": deepcopy(response.get("error")),
            "parse_error": result.get("parse_error", model_comparison.get("parse_error")),
            "game_plan": result.get("game_plan") or result.get("goals") or "",
            "rationale": result.get("rationale") or result.get("reasoning") or "",
            "rationale_source": (
                "rationale"
                if result.get("rationale") is not None
                else "legacy_reasoning_field"
                if result.get("reasoning") is not None
                else None
            ),
            "native_reasoning": result.get("native_reasoning") or "",
            "native_reasoning_details": deepcopy(result.get("native_reasoning_details") or []),
            "reasoning_request": deepcopy(result.get("reasoning_request")),
            "reasoning_tokens": result.get("reasoning_tokens"),
            "native_reasoning_returned": result.get("native_reasoning_returned"),
            "usage": deepcopy(result.get("usage") or {}),
            "finish_reason": result.get("finish_reason"),
            "provider_native_finish_reason": result.get(
                "provider_native_finish_reason"
            ),
            "provider_response_id": result.get("provider_response_id"),
            "provider_request_id": result.get("provider_request_id"),
            "latency_ms": result.get("latency_ms"),
            "context_version": result.get("context_version"),
            "context_prompt": result.get("context_prompt") or result.get("prompt") or "",
            "system_prompt": result.get("system_prompt") or "",
            "raw_response": result.get("raw_response") or "",
        },
    }


def _normalize_response(
    response: Mapping[str, JsonValue] | None,
    repair: Mapping[str, JsonValue] | None,
    model_id: str,
) -> JsonDict:
    if response is None:
        return {}
    normalized = deepcopy(dict(response))
    normalized["response_source"] = "action_diff"
    result = normalized.get("result")
    if isinstance(result, dict):
        result = deepcopy(result)
        normalized["result"] = result
    if repair:
        if not isinstance(result, dict):
            raise DecisionEvalArtifactError(
                f"Rationale repair {repair.get('decision_id')} has no base response"
            )
        if repair.get("model_id") != model_id:
            raise DecisionEvalArtifactError("Rationale repair model does not match the run")
        result["game_plan"] = repair.get("game_plan") or repair.get("goals") or ""
        result["rationale"] = repair.get("rationale") or repair.get("reasoning") or ""
        result["raw_response"] = repair.get("raw_response") or result.get("raw_response")
        normalized["response_source"] = "selected_rationale_repair"
    return normalized

