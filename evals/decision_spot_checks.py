"""Read-only adapters for bucketed replay decision-evaluation artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evals.decision_buckets import DecisionBucketSuite, load_decision_bucket_suite


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISION_EVAL_SCHEMA = "decision-spot-check-run-v1"
INDEX_SCHEMA = "decision-bucket-index-v1"


@dataclass(frozen=True)
class DecisionEvalRunConfig:
    id: str
    title: str
    description: str
    artifact_dir: Path
    model_id: str
    bucket_index_path: Path
    response_override_paths: tuple[Path, ...] = ()
    rationale_repair_paths: tuple[Path, ...] = ()


CURATED_DECISION_RUNS: dict[str, DecisionEvalRunConfig] = {
    "qwen3_8_27b_blue_242781000": DecisionEvalRunConfig(
        id="qwen3_8_27b_blue_242781000",
        title="Qwen 3.8 27B · BLUE · 242781000",
        description="Full causal model-versus-human action trace for the BLUE seat.",
        artifact_dir=PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817",
        model_id="qwen/qwen3.8-27b",
        bucket_index_path=PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "action_selection_diff/qwen3_8_27b_blue_20260817"
        / "decision_buckets_v1.jsonl",
        response_override_paths=(
            PROJECT_ROOT
            / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
            / "action_selection_diff/qwen3_8_27b_blue_20260817"
            / "setup_strategy_overrides.jsonl",
        ),
        rationale_repair_paths=(
            PROJECT_ROOT
            / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
            / "action_selection_diff/qwen3_8_27b_blue_20260817"
            / "setup_rationale_repairs.jsonl",
        ),
    ),
}


class DecisionEvalArtifactError(ValueError):
    """Raised when a decision-eval artifact cannot be safely joined."""


def decision_bucket_catalog() -> dict[str, Any]:
    suite = load_decision_bucket_suite()
    return suite.model_dump(mode="json", by_alias=True)


def list_decision_eval_runs(
    configs: Mapping[str, DecisionEvalRunConfig] | None = None,
) -> list[dict[str, Any]]:
    run_configs = configs or CURATED_DECISION_RUNS
    runs = []
    for config in run_configs.values():
        available = _run_files_available(config)
        metadata = _read_optional_json(config.artifact_dir / "plan.json") or {}
        runs.append(
            {
                "id": config.id,
                "title": config.title,
                "description": config.description,
                "model_id": config.model_id,
                "game_id": str(metadata.get("game_id") or ""),
                "target_player_id": metadata.get("target_player_id"),
                "target_engine_color": metadata.get("target_engine_color"),
                "available": available,
            }
        )
    return runs


def load_decision_eval_run(
    config: DecisionEvalRunConfig,
    suite: DecisionBucketSuite | None = None,
) -> dict[str, Any]:
    bucket_suite = suite or load_decision_bucket_suite()
    plan = _read_json(config.artifact_dir / "plan.json")
    manifest = _read_jsonl(config.artifact_dir / "decision_manifest.jsonl")
    comparisons = _read_jsonl(config.artifact_dir / "comparisons.jsonl", required=False)
    bucket_rows = _read_jsonl(config.bucket_index_path)
    response_rows = _read_jsonl(config.artifact_dir / "responses.jsonl", required=False)
    for path in config.response_override_paths:
        response_rows.extend(_read_jsonl(path, required=False))
    repair_rows: list[dict[str, Any]] = []
    for path in config.rationale_repair_paths:
        repair_rows.extend(_read_jsonl(path, required=False))

    game_id = str(plan.get("game_id") or "")
    if not game_id:
        raise DecisionEvalArtifactError("Decision-eval plan is missing game_id")
    if config.model_id not in plan.get("models", []):
        raise DecisionEvalArtifactError(
            f"Decision-eval plan does not contain model {config.model_id}"
        )

    manifest_by_id = _unique_index(manifest, "decision manifest")
    comparison_by_id = _unique_index(comparisons, "comparison")
    bucket_by_id = _unique_index(bucket_rows, "bucket index")
    latest_responses = _latest_responses(response_rows, config.model_id)
    repairs = _unique_index(repair_rows, "rationale repair")

    if set(bucket_by_id) != set(manifest_by_id):
        missing = sorted(set(manifest_by_id) - set(bucket_by_id))
        extra = sorted(set(bucket_by_id) - set(manifest_by_id))
        raise DecisionEvalArtifactError(
            "Bucket index does not cover the decision manifest exactly: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    decisions = []
    for manifest_row in sorted(
        manifest,
        key=lambda row: (int(row.get("replay_index", -1)), str(row.get("decision_id"))),
    ):
        decision_id = str(manifest_row["decision_id"])
        bucket_row = bucket_by_id[decision_id]
        assignment = bucket_row.get("bucket_assignment")
        if not isinstance(assignment, dict):
            raise DecisionEvalArtifactError(
                f"Bucket index row {decision_id} is missing bucket_assignment"
            )
        if assignment.get("suite_id") != bucket_suite.id or assignment.get(
            "suite_version"
        ) != bucket_suite.version:
            raise DecisionEvalArtifactError(
                f"Bucket suite mismatch for {decision_id}: "
                f"{assignment.get('suite_id')}@{assignment.get('suite_version')}"
            )
        if bucket_row.get("schema") != INDEX_SCHEMA:
            raise DecisionEvalArtifactError(
                f"Bucket index row {decision_id} has unsupported schema"
            )

        response = latest_responses.get(decision_id)
        repair = repairs.get(decision_id)
        normalized_response = _normalize_response(response, repair, config.model_id)
        comparison = comparison_by_id.get(decision_id)
        decisions.append(
            _joined_decision(
                manifest_row,
                bucket_row,
                normalized_response,
                comparison,
                config.model_id,
            )
        )

    bucket_counts = _bucket_counts(decisions, bucket_suite)
    stage_counts = Counter(decision["stage"] for decision in decisions)
    response_count = sum(decision["model"]["response_present"] for decision in decisions)
    disagreement_count = sum(
        decision["model"]["response_present"]
        and decision["model"]["agreement"] is False
        for decision in decisions
    )
    run = {
        "schema": DECISION_EVAL_SCHEMA,
        "id": config.id,
        "title": config.title,
        "description": config.description,
        "game_id": game_id,
        "model_id": config.model_id,
        "target_player_id": plan.get("target_player_id"),
        "target_engine_color": plan.get("target_engine_color"),
        "bucket_suite": {
            "id": bucket_suite.id,
            "version": bucket_suite.version,
            "schema": bucket_suite.schema_id,
        },
        "decision_count": len(decisions),
        "bucketed_decision_count": sum(bool(decision["bucket_ids"]) for decision in decisions),
        "response_count": response_count,
        "disagreement_count": disagreement_count,
        "stage_counts": dict(sorted(stage_counts.items())),
        "bucket_counts": bucket_counts,
        "decisions": decisions,
    }
    return run


def filter_decisions(
    run: Mapping[str, Any],
    *,
    bucket_id: str | None = None,
    stage: str | None = None,
    agreement: str | None = None,
    classification: str | None = None,
) -> list[dict[str, Any]]:
    decisions = list(run.get("decisions") or [])
    if bucket_id and bucket_id != "all":
        decisions = [row for row in decisions if bucket_id in row.get("bucket_ids", [])]
    if stage and stage != "all":
        decisions = [row for row in decisions if row.get("stage") == stage]
    if classification and classification != "all":
        decisions = [row for row in decisions if row.get("classification") == classification]
    if agreement and agreement != "all":
        if agreement == "agree":
            decisions = [row for row in decisions if row["model"].get("agreement") is True]
        elif agreement == "disagree":
            decisions = [row for row in decisions if row["model"].get("agreement") is False]
        elif agreement == "unscored":
            decisions = [row for row in decisions if not row["model"].get("response_present")]
        else:
            raise ValueError("agreement must be all, agree, disagree, or unscored")
    return decisions


def compact_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
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
            key: decision["model"].get(key)
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


def decision_detail(run: Mapping[str, Any], decision_id: str) -> dict[str, Any] | None:
    return next(
        (decision for decision in run.get("decisions", []) if decision["decision_id"] == decision_id),
        None,
    )


def _joined_decision(
    manifest: Mapping[str, Any],
    bucket_row: Mapping[str, Any],
    response: Mapping[str, Any],
    comparison: Mapping[str, Any] | None,
    model_id: str,
) -> dict[str, Any]:
    assignment = dict(bucket_row["bucket_assignment"])
    model_comparison = {}
    if isinstance(comparison, Mapping):
        models = comparison.get("models")
        if isinstance(models, Mapping) and isinstance(models.get(model_id), Mapping):
            model_comparison = dict(models[model_id])
    result = response.get("result") if isinstance(response.get("result"), Mapping) else {}
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
        "bucket_ids": list(assignment.get("bucket_ids") or []),
        "episode_ids": dict(assignment.get("episode_ids") or {}),
        "bucket_evidence": dict(assignment.get("evidence") or {}),
        "state_features": bucket_row.get("state_features"),
        "actor": dict(manifest.get("actor") or {}),
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
    response: Mapping[str, Any] | None,
    repair: Mapping[str, Any] | None,
    model_id: str,
) -> dict[str, Any]:
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


def _bucket_counts(
    decisions: Sequence[Mapping[str, Any]],
    suite: DecisionBucketSuite,
) -> list[dict[str, Any]]:
    decision_counts: Counter[str] = Counter()
    response_counts: Counter[str] = Counter()
    disagreement_counts: Counter[str] = Counter()
    episodes: defaultdict[str, set[str]] = defaultdict(set)
    for decision in decisions:
        for bucket_id in decision.get("bucket_ids", []):
            decision_counts[bucket_id] += 1
            episode_id = (decision.get("episode_ids") or {}).get(bucket_id)
            if episode_id:
                episodes[bucket_id].add(str(episode_id))
            model = decision.get("model") or {}
            if model.get("response_present"):
                response_counts[bucket_id] += 1
                if model.get("agreement") is False:
                    disagreement_counts[bucket_id] += 1
    return [
        {
            "bucket_id": bucket.id,
            "decision_count": decision_counts[bucket.id],
            "episode_count": len(episodes[bucket.id]),
            "response_count": response_counts[bucket.id],
            "disagreement_count": disagreement_counts[bucket.id],
        }
        for bucket in suite.buckets
    ]


def _latest_responses(
    rows: Iterable[Mapping[str, Any]],
    model_id: str,
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("model_id") != model_id:
            continue
        decision_id = row.get("decision_id")
        if isinstance(decision_id, str) and decision_id:
            latest[decision_id] = dict(row)
    return latest


def _unique_index(
    rows: Iterable[Mapping[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        decision_id = row.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            raise DecisionEvalArtifactError(f"{label} row is missing decision_id")
        if decision_id in indexed:
            raise DecisionEvalArtifactError(f"Duplicate {label} decision_id: {decision_id}")
        indexed[decision_id] = dict(row)
    return indexed


def _run_files_available(config: DecisionEvalRunConfig) -> bool:
    required = (
        config.artifact_dir / "plan.json",
        config.artifact_dir / "decision_manifest.jsonl",
        config.bucket_index_path,
    )
    return all(path.exists() for path in required)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DecisionEvalArtifactError(f"Missing decision-eval artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionEvalArtifactError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise DecisionEvalArtifactError(f"{path.name} must contain a JSON object")
    return value


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise DecisionEvalArtifactError(f"Missing decision-eval artifact: {path}")
        return []
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DecisionEvalArtifactError(f"Could not read {path.name}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DecisionEvalArtifactError(
                f"Could not read {path.name}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise DecisionEvalArtifactError(
                f"{path.name}:{line_number} must contain a JSON object"
            )
        rows.append(row)
    return rows
