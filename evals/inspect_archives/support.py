"""Metadata readers, coercions, digests, and path helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inspect_ai.model import ModelUsage, StopReason
from inspect_ai.solver import TaskState

from evals.catan_board_bench.paths import PROJECT_ROOT
from evals.inspect_archives.config import InspectArchiveBundle, JsonDict
from evals.json_types import JsonValue, as_dict

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "stop",
    "max_tokens": "max_tokens",
    "model_length": "model_length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "unknown": "unknown",
}


def _archive_metadata(state: TaskState) -> JsonDict:
    archive = state.metadata.get("archive")
    if not isinstance(archive, dict):
        raise ValueError("Inspect archive sample is missing archive metadata")
    return archive


def _policy_metadata(state: TaskState) -> JsonDict:
    policy = state.metadata.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Inspect archive sample is missing policy metadata")
    return policy


def _stored_score(archive: JsonDict) -> JsonDict:
    stored = archive.get("score")
    if not isinstance(stored, dict):
        raise ValueError("strict archive sample is missing its stored score")
    return stored


def _completion(state: TaskState) -> str:
    return state.output.completion if state.output else ""


def _expected_inspect_scores(bundle: InspectArchiveBundle) -> dict[str, float]:
    metrics = bundle.expected_metrics
    if bundle.input_mode == "raw_image":
        requests = _integer(metrics.get("requests"))
        if requests != bundle.imported_records or requests == 0:
            return {}
        return {
            "strict_exact": _integer(metrics.get("exact")) / requests,
            "strict_json_valid": _integer(metrics.get("json_valid")) / requests,
            "strict_protocol_exact": _integer(metrics.get("protocol_exact")) / requests,
        }

    responses = _integer(metrics.get("response_count"))
    nontrivial = _integer(metrics.get("nontrivial_responses"))
    if responses != bundle.imported_records or responses == 0:
        return {}
    return {
        "policy_selection_valid": _integer(metrics.get("valid_selections")) / responses,
        "policy_parse_clean": (responses - _integer(metrics.get("parse_warnings")))
        / responses,
        "recorded_human_action_match": _integer(metrics.get("exact_human_matches"))
        / responses,
        "nontrivial_recorded_human_action_match": (
            _integer(metrics.get("nontrivial_human_matches")) / nontrivial
            if nontrivial
            else 0.0
        ),
    }


def _strict_explanation(archive: JsonDict) -> str:
    score = _stored_score(archive)
    if score.get("error"):
        return f"Archived strict scorer error: {score['error']}"
    if score.get("correct"):
        return "Archived strict_typed_json/v2 engine-oracle answer is exact."
    return (
        "Archived strict_typed_json/v2 engine-oracle mismatch. "
        f"Parsed response: {json.dumps(score.get('parsed'), sort_keys=True)}"
    )


def _agreement_explanation(policy: JsonDict) -> str:
    relation = "matches" if policy.get("agreement") else "differs from"
    return (
        "Descriptive replay agreement only; the recorded human action is not a "
        f"policy-quality oracle. Model #{policy.get('model_action_index')} "
        f"{relation} human #{policy.get('human_action_index')}. "
        f"Model: {policy.get('model_description')} Human: {policy.get('human_description')}"
    )


def _model_usage(raw: JsonValue) -> ModelUsage:
    usage = raw if isinstance(raw, dict) else {}
    prompt_tokens = _integer(usage.get("prompt_tokens"))
    completion_tokens = _integer(usage.get("completion_tokens"))
    total_tokens = _integer(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    return ModelUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        input_tokens_cache_write=_nested_integer(prompt_details, "cache_write_tokens"),
        input_tokens_cache_read=_nested_integer(prompt_details, "cached_tokens"),
        reasoning_tokens=_nested_integer(completion_details, "reasoning_tokens"),
    )


def _stop_reason(value: JsonValue) -> StopReason:
    return _STOP_REASONS.get(value, "unknown") if isinstance(value, str) else "unknown"


def _seconds(value: JsonValue) -> float | None:
    # Non-numeric JSON (lists, objects) failed float() with TypeError -> None.
    if value is None or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value) / 1000
    except ValueError:
        return None


def _integer(value: JsonValue) -> int:
    candidate = value or 0
    # Non-numeric JSON (lists, objects) failed int() with TypeError -> 0.
    if not isinstance(candidate, (int, float, str)):
        return 0
    try:
        return int(candidate)
    except ValueError:
        return 0


def _nested_integer(value: JsonValue, key: str) -> int | None:
    if not isinstance(value, dict) or value.get(key) is None:
        return None
    return _integer(value[key])


def _unique_rows(rows: list[JsonDict], key: str, source_dir: Path) -> dict[str, JsonDict]:
    indexed: dict[str, JsonDict] = {}
    for row in rows:
        value = str(row.get(key))
        if value in indexed:
            raise ValueError(f"duplicate {key}={value} in {source_dir}")
        indexed[value] = row
    return indexed


def _resolved_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> JsonDict:
    return as_dict(json.loads(path.read_text()), path.name)


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        as_dict(json.loads(line), f"{path.name} row")
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
