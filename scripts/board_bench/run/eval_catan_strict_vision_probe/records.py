"""Response records, admissibility, and score recomputation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone

from cle.players.data import JsonValue
from evals.catan_board_bench.ascii_variations import score_strict_json_answer
from scripts.board_bench.run.eval_catan_strict_vision_probe.constants import (
    VisionJob,
    usage_reasoning_tokens,
)
from scripts.board_bench.shapes import JsonDict, obj, text

__all__ = [
    "accepted_records",
    "admissible_record",
    "recompute_record_score",
    "response_record",
    "served_model_matches",
]


def served_model_matches(requested: str, served: JsonValue) -> bool:
    if served == requested:
        return True
    return requested.endswith(":free") and served == requested.removesuffix(":free")


def response_record(job: VisionJob, result: JsonDict, plan: JsonDict) -> JsonDict:
    qa = job["qa"]
    response = text(result.get("response", ""), "response")
    settings = obj(plan["request_settings"], "request_settings")
    scorer = obj(plan["scorer"], "plan scorer")
    error = result.get("error")
    if not error and not response:
        error = "empty response"
    if not error and not served_model_matches(
        text(plan["model"], "plan model"), result.get("served_model")
    ):
        error = (
            f"unexpected served model {result.get('served_model')!r}; expected {plan['model']!r}"
        )
    expected_provider = settings["provider"]
    served_provider = result.get("provider")
    if not error and expected_provider == "novita" and served_provider != "novita":
        error = f"unexpected provider {served_provider!r}; expected 'novita'"
    if not error and expected_provider == "openrouter" and not served_provider:
        error = "OpenRouter response did not identify the serving provider"
    reasoning_tokens = usage_reasoning_tokens(obj(result.get("usage") or {}, "usage"))
    if not error and reasoning_tokens:
        error = f"reasoning control violation: tokens={reasoning_tokens}"
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model_id": plan["model"],
        "served_model": result.get("served_model"),
        "provider": result.get("provider"),
        "sample_id": qa["sample_id"],
        "question_id": qa["id"],
        "category": qa["category"],
        "question": qa["question"],
        "expected": qa["answer"],
        "expected_text": qa["answer_text"],
        "response": response,
        "score": score_strict_json_answer(obj(qa["answer"], "expected answer"), response),
        "scorer_version": scorer["version"],
        "scorer_sha256": scorer["sha256"],
        "engine_state_sha256": qa["engine_state_sha256"],
        "image_path": job["image_path"],
        "image_sha256": job["image_sha256"],
        "contract_path": job["contract_path"],
        "contract_sha256": job["contract_sha256"],
        "prompt_characters": len(job["prompt"]),
        "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
        "latency_ms": result.get("latency_ms"),
        "usage": result.get("usage", {}),
        "error": error,
    }


def accepted_records(
    records: Sequence[JsonDict],
    *,
    jobs: Sequence[VisionJob],
    plan: JsonDict,
) -> dict[str, JsonDict]:
    jobs_by_id = {text(job["qa"]["id"], "question id"): job for job in jobs}
    accepted: dict[str, JsonDict] = {}
    for record in records:
        raw_id = record.get("question_id")
        question_id = raw_id if isinstance(raw_id, str) else None
        job = jobs_by_id.get(question_id) if question_id is not None else None
        if job is None or question_id is None:
            continue
        if not admissible_record(record, job=job, plan=plan):
            continue
        accepted.setdefault(question_id, recompute_record_score(record, job["qa"]))
    return accepted


def admissible_record(
    record: JsonDict | None,
    *,
    job: VisionJob,
    plan: JsonDict,
) -> bool:
    if not record or record.get("error") or not record.get("response"):
        return False
    qa = job["qa"]
    settings = obj(plan["request_settings"], "request_settings")
    scorer = obj(plan["scorer"], "plan scorer")
    return all(
        (
            record.get("question_id") == qa["id"],
            record.get("sample_id") == qa["sample_id"],
            record.get("engine_state_sha256") == qa["engine_state_sha256"],
            record.get("expected") == qa["answer"],
            record.get("model_id") == plan["model"],
            served_model_matches(
                text(plan["model"], "plan model"), record.get("served_model")
            ),
            (
                record.get("provider") == "novita"
                if settings["provider"] == "novita"
                else bool(record.get("provider"))
            ),
            record.get("image_sha256") == job["image_sha256"],
            record.get("contract_sha256") == job["contract_sha256"],
            record.get("prompt_sha256") == hashlib.sha256(job["prompt"].encode()).hexdigest(),
            record.get("scorer_version") == scorer["version"],
            record.get("scorer_sha256") == scorer["sha256"],
            usage_reasoning_tokens(obj(record.get("usage") or {}, "usage")) == 0,
        )
    )


def recompute_record_score(record: JsonDict, qa: JsonDict) -> JsonDict:
    normalized = dict(record)
    normalized["score"] = score_strict_json_answer(
        obj(qa["answer"], "expected answer"), text(record["response"], "response")
    )
    return normalized
