"""Response records and admissibility for the tile prompt ablation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    CATEGORY,
    ScoreDict,
    TileTruth,
    score_response,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.defaults import (
    TileJob,
    has_explicit_reasoning_usage,
    job_key,
    usage_reasoning_tokens,
)
from scripts.board_bench.shapes import JsonDict, obj, text, values

__all__ = ["accepted_records", "admissible_record", "response_record", "score_json", "truth_json"]


def truth_json(truth: TileTruth) -> JsonDict:
    """Render one ground truth as plain JSON."""
    return {"resource": truth["resource"], "number": truth["number"]}


def score_json(score: ScoreDict) -> JsonDict:
    """Render one scoring breakdown as plain JSON."""
    parsed = score["parsed"]
    return {
        "protocol_valid": score["protocol_valid"],
        "protocol_exact": score["protocol_exact"],
        "resource_correct": score["resource_correct"],
        "number_correct": score["number_correct"],
        "pair_correct": score["pair_correct"],
        "wrapper_rescued_content_correct": score["wrapper_rescued_content_correct"],
        "parsed": None if parsed is None else truth_json(parsed),
    }


def _usage_of(payload: JsonDict) -> JsonDict:
    usage = payload.get("usage") or {}
    return usage if isinstance(usage, dict) else {}


def response_record(job: TileJob, result: JsonDict, plan: JsonDict) -> JsonDict:
    error = result.get("error")
    usage = _usage_of(result)
    providers = values(plan["provider_order"], "provider_order")
    if not error and not has_explicit_reasoning_usage(usage):
        error = "reasoning-token usage field is missing or invalid"
    if not error and (
        usage_reasoning_tokens(usage) or result.get("native_reasoning")
    ):
        error = (
            "reasoning control violation: "
            f"tokens={usage_reasoning_tokens(usage)}, "
            f"native_reasoning={result.get('native_reasoning')!r}"
        )
    if not error and result.get("served_model") != plan["model"]:
        error = (
            "served model mismatch: "
            f"expected={plan['model']!r}, got={result.get('served_model')!r}"
        )
    if not error and result.get("provider") != providers[0]:
        error = (
            "provider mismatch: "
            f"expected={providers[0]!r}, "
            f"got={result.get('provider')!r}"
        )
    if not error and result.get("finish_reason") != "stop":
        error = f"finish_reason is not stop: {result.get('finish_reason')!r}"
    if not error and not result.get("response"):
        error = "empty final response content"
    response = text(result.get("response", ""), "response")
    scorer = obj(plan["scorer"], "plan scorer")
    return {
        "condition": job["condition"],
        "question_id": job["qa"]["id"],
        "sample_id": job["qa"]["sample_id"],
        "category": CATEGORY,
        "truth": truth_json(job["qa"]["truth"]),
        "expected_response": job["expected_response"],
        "response": response,
        "score": score_json(score_response(job["qa"]["truth"], response, job["condition"])),
        "error": error,
        "model_id": plan["model"],
        "served_model": result.get("served_model"),
        "provider": result.get("provider"),
        "prompt_sha256": job["prompt_sha256"],
        "image_path": job["qa"]["image_path"],
        "image_sha256": job["qa"]["image_sha256"],
        "submitted_image_sha256": result.get("submitted_image_sha256"),
        "scorer_version": scorer["version"],
        "scorer_sha256": scorer["sha256"],
        "latency_ms": result.get("latency_ms"),
        "usage": usage,
        "finish_reason": result.get("finish_reason"),
        "native_reasoning": result.get("native_reasoning") or {},
        "request_id": result.get("request_id"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def admissible_record(record: JsonDict, job: TileJob, plan: JsonDict) -> bool:
    if record.get("error") is not None:
        return False
    usage = _usage_of(record)
    scorer = obj(plan["scorer"], "plan scorer")
    providers = values(plan["provider_order"], "provider_order")
    expected_score = score_json(
        score_response(
            job["qa"]["truth"], text(record.get("response", ""), "response"), job["condition"]
        )
    )
    return bool(
        record.get("condition") == job["condition"]
        and record.get("question_id") == job["qa"]["id"]
        and record.get("sample_id") == job["qa"]["sample_id"]
        and record.get("truth") == job["qa"]["truth"]
        and record.get("expected_response") == job["expected_response"]
        and record.get("model_id") == plan["model"]
        and record.get("served_model") == plan["model"]
        and record.get("provider") == providers[0]
        and record.get("prompt_sha256") == job["prompt_sha256"]
        and record.get("image_path") == job["qa"]["image_path"]
        and record.get("image_sha256") == job["qa"]["image_sha256"]
        and record.get("submitted_image_sha256") == job["qa"]["image_sha256"]
        and record.get("scorer_version") == scorer["version"]
        and record.get("scorer_sha256") == scorer["sha256"]
        and record.get("finish_reason") == "stop"
        and bool(record.get("response"))
        and record.get("score") == expected_score
        and not record.get("native_reasoning")
        and has_explicit_reasoning_usage(usage)
        and usage_reasoning_tokens(usage) == 0
    )


def accepted_records(
    records: Sequence[JsonDict],
    *,
    jobs: Sequence[TileJob],
    plan: JsonDict,
) -> dict[tuple[str, str], JsonDict]:
    jobs_by_key = {job_key(job): job for job in jobs}
    accepted: dict[tuple[str, str], JsonDict] = {}
    for record in records:
        raw_condition = record.get("condition")
        raw_question = record.get("question_id")
        if not isinstance(raw_condition, str) or not isinstance(raw_question, str):
            continue
        key = (raw_condition, raw_question)
        job = jobs_by_key.get(key)
        if job is not None and admissible_record(record, job, plan):
            normalized = dict(record)
            normalized["score"] = score_json(
                score_response(
                    job["qa"]["truth"],
                    text(record.get("response", ""), "response"),
                    job["condition"],
                )
            )
            accepted.setdefault(key, normalized)
    return accepted
