"""Run-level aggregation across models and decision buckets."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Sequence

from evals.json_types import JsonDict, JsonValue, as_dicts, as_list, as_str
from evals.replay_action_diff.comparisons import _percent, _percentile, build_comparisons
from evals.replay_action_diff.contracts import (
    COMPARISON_PARSER_VERSION,
    SCHEMA_VERSION,
    object_or_empty,
)
from evals.replay_action_diff.identity import utc_now
from evals.replay_action_diff.responses import _latest_responses, normalize_response_selection
from evals.replay_action_diff.shapes import (
    Comparison,
    ModelPair,
    ModelSummary,
    RunSummary,
    SubsetStats,
    TypeStats,
)


def _int(value: JsonValue) -> int:
    """``int(value)`` over the JSON scalars ``int()`` itself accepts."""
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"int() argument must be a number or string, not {type(value).__name__}")


def _float(value: JsonValue) -> float:
    """``float(value)`` over the JSON scalars ``float()`` itself accepts."""
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"float() argument must be a number or string, not {type(value).__name__}")


def _result(row: JsonDict) -> JsonDict:
    return object_or_empty(row.get("result"), "result")


def summarize_run(
    scan: JsonDict,
    response_rows: Sequence[JsonDict],
    models: Sequence[str],
) -> RunSummary:
    manifest = as_dicts(scan["records"], "records")
    normalized_rows = [normalize_response_selection(row) for row in response_rows]
    comparisons = build_comparisons(manifest, normalized_rows, models)
    latest = _latest_responses(normalized_rows)
    classification_counts = Counter(
        as_str(row["classification"], "classification") for row in manifest
    )
    summary_models: dict[str, ModelSummary] = {}

    for model_id in models:
        eligible = len(comparisons)
        present_rows = [
            latest[(item["decision_id"], model_id)]
            for item in comparisons
            if (item["decision_id"], model_id) in latest
        ]
        api_errors = sum(bool(row.get("error")) for row in present_rows)
        valid_rows = [
            row
            for row in present_rows
            if not row.get("error")
            and isinstance(row.get("model_action_index"), int)
        ]
        agreements = sum(bool(row.get("agreement")) for row in valid_rows)
        parse_errors = sum(
            bool(_result(row).get("parse_error"))
            for row in present_rows
            if not row.get("error")
        )
        usage_rows = [
            object_or_empty(_result(row).get("usage"), "usage")
            for row in present_rows
            if not row.get("error")
        ]
        latencies = [
            _float(_result(row).get("latency_ms"))
            for row in present_rows
            if _result(row).get("latency_ms") is not None
        ]
        by_type: dict[str, TypeStats] = {}
        for action_type in sorted(
            {item["effective_action_type"] for item in comparisons}
        ):
            type_items = [
                item for item in comparisons if item["effective_action_type"] == action_type
            ]
            type_rows = [
                latest[(item["decision_id"], model_id)]
                for item in type_items
                if (item["decision_id"], model_id) in latest
                and not latest[(item["decision_id"], model_id)].get("error")
                and isinstance(
                    latest[(item["decision_id"], model_id)].get("model_action_index"),
                    int,
                )
            ]
            type_agreements = sum(bool(row.get("agreement", False)) for row in type_rows)
            by_type[action_type] = {
                "eligible": len(type_items),
                "valid": len(type_rows),
                "agreements": type_agreements,
                "agreement_rate_valid": _percent(type_agreements, len(type_rows)),
            }

        forced_items = [item for item in comparisons if item["forced"]]
        nontrivial_items = [item for item in comparisons if not item["forced"]]

        def subset_stats(items: Sequence[Comparison]) -> SubsetStats:
            rows = [
                latest[(item["decision_id"], model_id)]
                for item in items
                if (item["decision_id"], model_id) in latest
                and not latest[(item["decision_id"], model_id)].get("error")
                and isinstance(
                    latest[(item["decision_id"], model_id)].get("model_action_index"),
                    int,
                )
            ]
            agree = sum(bool(row.get("agreement", False)) for row in rows)
            return {
                "eligible": len(items),
                "valid": len(rows),
                "agreements": agree,
                "agreement_rate_valid": _percent(agree, len(rows)),
                "strict_agreement_rate": _percent(agree, len(items)),
            }

        summary_models[model_id] = {
            "eligible_decisions": eligible,
            "responses_present": len(present_rows),
            "valid_selections": len(valid_rows),
            "coverage": _percent(len(valid_rows), eligible),
            "api_errors": api_errors,
            "parse_errors": parse_errors,
            "agreements": agreements,
            "unscored_followups": sum(
                row.get("followup_scoring") == "unscored" for row in valid_rows
            ),
            "agreement_rate_valid": _percent(agreements, len(valid_rows)),
            "strict_agreement_rate": _percent(agreements, eligible),
            "forced": subset_stats(forced_items),
            "nontrivial": subset_stats(nontrivial_items),
            "by_action_type": by_type,
            "usage": {
                "prompt_tokens": sum(_int(row.get("prompt_tokens", 0)) for row in usage_rows),
                "completion_tokens": sum(
                    _int(row.get("completion_tokens", 0)) for row in usage_rows
                ),
                "total_tokens": sum(_int(row.get("total_tokens", 0)) for row in usage_rows),
                "cost_usd": sum(_float(row.get("cost", 0.0)) for row in usage_rows),
            },
            "latency_ms": {
                "mean": statistics.mean(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies) if latencies else None,
            },
        }

    model_pair: ModelPair | None = None
    if len(models) == 2:
        left, right = models
        both_valid = 0
        same_selection = 0
        both_human = 0
        left_only_human = 0
        right_only_human = 0
        neither_human = 0
        same_alternative = 0
        for item in comparisons:
            left_result = item["models"][left]
            right_result = item["models"][right]
            if not isinstance(left_result["action_index"], int) or not isinstance(
                right_result["action_index"], int
            ):
                continue
            both_valid += 1
            same_model_choice = (
                left_result["action"] == right_result["action"]
                if left_result["action"] is not None
                and right_result["action"] is not None
                else left_result["action_index"] == right_result["action_index"]
            )
            if same_model_choice:
                same_selection += 1
            left_human = left_result["agreement"]
            right_human = right_result["agreement"]
            if left_human and right_human:
                both_human += 1
            elif left_human:
                left_only_human += 1
            elif right_human:
                right_only_human += 1
            else:
                neither_human += 1
                if same_model_choice:
                    same_alternative += 1
        model_pair = {
            "models": [left, right],
            "selection_scope": "primary_action",
            "both_valid": both_valid,
            "same_selection": same_selection,
            "same_selection_rate": _percent(same_selection, both_valid),
            "both_match_human": both_human,
            "left_only_matches_human": left_only_human,
            "right_only_matches_human": right_only_human,
            "neither_matches_human": neither_human,
            "same_nonhuman_alternative": same_alternative,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_parser_version": COMPARISON_PARSER_VERSION,
        "agreement_scope": "primary_action",
        "generated_at": utc_now(),
        "game_id": scan["game_id"],
        "target_player_id": scan["target_player_id"],
        "target_engine_color": scan["target_engine_color"],
        "parsed_action_count": scan["parsed_action_count"],
        "canonicalization_count": len(
            as_list(scan.get("canonicalizations", []), "canonicalizations")
        ),
        "target_action_records": len(manifest),
        "classification_counts": dict(classification_counts),
        "exact_decisions": len(comparisons),
        "forced_exact_decisions": sum(bool(item["forced"]) for item in comparisons),
        "nontrivial_exact_decisions": sum(not item["forced"] for item in comparisons),
        "step_statuses": scan["step_statuses"],
        "semantic_error_count": len(as_list(scan["semantic_errors"], "semantic_errors")),
        "models": summary_models,
        "model_pair": model_pair,
    }

