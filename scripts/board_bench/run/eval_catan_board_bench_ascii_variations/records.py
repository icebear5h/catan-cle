"""Response records, resume bookkeeping, and admissibility checks."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone

from evals.catan_board_bench.ascii_variations import score_strict_json_answer
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.constants import (
    usage_reasoning_tokens,
)
from scripts.board_bench.shapes import JsonDict, obj, text, values

__all__ = ["latest_records", "response_record", "successful_record"]


def response_record(
    job: JsonDict,
    result: JsonDict,
    plan: JsonDict,
) -> JsonDict:
    qa = obj(job["qa"], "job qa")
    response = text(result.get("response", ""), "response")
    score = score_strict_json_answer(obj(qa["answer"], "expected answer"), response)
    settings = obj(plan["request_settings"], "request_settings")
    scorer = obj(plan["scorer"], "plan scorer")
    error = result.get("error")
    if not error and not response:
        error = "empty response"
    expected_model = plan["model"]
    expected_providers = values(settings["provider_order"], "provider_order")
    if not error and result.get("served_model") != expected_model:
        error = (
            f"unexpected served model {result.get('served_model')!r}; expected {expected_model!r}"
        )
    if not error and expected_providers and result.get("provider") not in expected_providers:
        error = (
            f"unexpected provider {result.get('provider')!r}; "
            f"expected one of {expected_providers!r}"
        )
    if not error and settings["reasoning_disabled"]:
        reasoning_tokens = usage_reasoning_tokens(obj(result.get("usage") or {}, "usage"))
        if reasoning_tokens or result.get("native_reasoning"):
            error = (
                "reasoning control violation: "
                f"tokens={reasoning_tokens}, "
                f"native_reasoning={result.get('native_reasoning')!r}"
            )
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model_id": expected_model,
        "served_model": result.get("served_model"),
        "provider": result.get("provider"),
        "variant": job["variant"],
        "sample_id": qa["sample_id"],
        "question_id": qa["id"],
        "category": qa["category"],
        "question": qa["question"],
        "expected": qa["answer"],
        "expected_text": qa["answer_text"],
        "response": response,
        "score": score,
        "scorer_version": scorer["version"],
        "scorer_sha256": scorer["sha256"],
        "fact_digest": qa["fact_digest"],
        "representation_path": job["representation_path"],
        "prompt_characters": len(text(job["prompt"], "job prompt")),
        "prompt_sha256": hashlib.sha256(
            text(job["prompt"], "job prompt").encode()
        ).hexdigest(),
        "latency_ms": result.get("latency_ms"),
        "usage": result.get("usage", {}),
        "finish_reason": result.get("finish_reason"),
        "native_reasoning": result.get("native_reasoning"),
        "request_id": result.get("request_id"),
        "error": error,
    }


def latest_records(records: Sequence[JsonDict]) -> dict[tuple[str, str], JsonDict]:
    latest: dict[tuple[str, str], JsonDict] = {}
    for record in records:
        latest[
            (
                text(record["variant"], "record variant"),
                text(record["question_id"], "record question_id"),
            )
        ] = record
    return latest


def successful_record(
    record: JsonDict | None,
    *,
    job: JsonDict,
    plan: JsonDict,
) -> bool:
    if not record or record.get("error") or not record.get("response"):
        return False
    qa = obj(job["qa"], "job qa")
    settings = obj(plan["request_settings"], "request_settings")
    scorer = obj(plan["scorer"], "plan scorer")
    expected_provider = values(settings["provider_order"], "provider_order")
    checks = (
        record.get("variant") == job["variant"],
        record.get("question_id") == qa["id"],
        record.get("sample_id") == qa["sample_id"],
        record.get("fact_digest") == qa["fact_digest"],
        record.get("expected") == qa["answer"],
        record.get("model_id") == plan["model"],
        record.get("served_model") == plan["model"],
        record.get("prompt_sha256")
        == hashlib.sha256(text(job["prompt"], "job prompt").encode()).hexdigest(),
        record.get("scorer_version") == scorer["version"],
        record.get("scorer_sha256") == scorer["sha256"],
        not expected_provider or record.get("provider") in expected_provider,
        not settings["reasoning_disabled"]
        or (
            usage_reasoning_tokens(obj(record.get("usage") or {}, "usage")) == 0
            and not record.get("native_reasoning")
        ),
    )
    return all(checks)
