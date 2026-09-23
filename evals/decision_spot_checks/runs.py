"""Bucket catalog, run listing, and assembly of one joined evaluation run."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

from evals.decision_buckets import DecisionBucketSuite, load_decision_bucket_suite
from evals.decision_spot_checks.aggregates import (
    _bucket_counts,
    _latest_responses,
    _unique_index,
)
from evals.decision_spot_checks.artifacts import (
    _read_json,
    _read_jsonl,
    _read_optional_json,
    _run_files_available,
)
from evals.decision_spot_checks.config import (
    CURATED_DECISION_RUNS,
    DECISION_EVAL_SCHEMA,
    INDEX_SCHEMA,
    DecisionEvalArtifactError,
    DecisionEvalRunConfig,
)
from evals.decision_spot_checks.decisions import (
    _joined_decision,
    _normalize_response,
)
from evals.decision_spot_checks.shapes import decision_model
from evals.json_types import JsonDict, JsonValue, as_list, as_str


def decision_bucket_catalog() -> JsonDict:
    suite = load_decision_bucket_suite()
    return suite.model_dump(mode="json", by_alias=True)


def list_decision_eval_runs(
    configs: Mapping[str, DecisionEvalRunConfig] | None = None,
) -> list[JsonDict]:
    run_configs = configs or CURATED_DECISION_RUNS
    runs: list[JsonDict] = []
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
) -> JsonDict:
    bucket_suite = suite or load_decision_bucket_suite()
    plan = _read_json(config.artifact_dir / "plan.json")
    manifest = _read_jsonl(config.artifact_dir / "decision_manifest.jsonl")
    comparisons = _read_jsonl(config.artifact_dir / "comparisons.jsonl", required=False)
    bucket_rows = _read_jsonl(config.bucket_index_path)
    response_rows = _read_jsonl(config.artifact_dir / "responses.jsonl", required=False)
    for path in config.response_override_paths:
        response_rows.extend(_read_jsonl(path, required=False))
    repair_rows: list[JsonDict] = []
    for path in config.rationale_repair_paths:
        repair_rows.extend(_read_jsonl(path, required=False))

    game_id = str(plan.get("game_id") or "")
    if not game_id:
        raise DecisionEvalArtifactError("Decision-eval plan is missing game_id")
    if config.model_id not in as_list(plan.get("models", []), "plan models"):
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

    decisions: list[JsonDict] = []
    for manifest_row in sorted(
        manifest,
        key=_manifest_order,
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
    stage_counts = Counter(as_str(decision["stage"], "stage") for decision in decisions)
    models = [decision_model(decision) for decision in decisions]
    # response_present is always a bool, so counting truthy rows equals the old sum.
    response_count = sum(1 for model in models if model["response_present"])
    disagreement_count = sum(
        1
        for model in models
        if model["response_present"] and model["agreement"] is False
    )
    run: JsonDict = {
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
        "bucket_counts": list[JsonValue](bucket_counts),
        "decisions": list[JsonValue](decisions),
    }
    return run


def _manifest_order(row: JsonDict) -> tuple[int, str]:
    replay_index = row.get("replay_index", -1)
    if not isinstance(replay_index, (int, float, str)):
        raise TypeError(f"replay_index is not a number: {type(replay_index).__name__}")
    return int(replay_index), str(row.get("decision_id"))
