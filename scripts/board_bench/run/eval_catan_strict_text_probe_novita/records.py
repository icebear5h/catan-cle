"""Response records, admissibility, and the run summary."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

from evals.catan_board_bench.ascii_variations import score_strict_json_answer
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats import FormatJob
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.constants import FORMAT_NAME
from scripts.board_bench.run.eval_catan_strict_vision_probe import (
    summarize_group,
    usage_reasoning_tokens,
)
from scripts.board_bench.shapes import JsonDict, obj, text

__all__ = [
    "accepted_records",
    "admissible_record",
    "recompute_record_score",
    "response_record",
    "summarize",
]


def response_record(job: FormatJob, result: JsonDict, plan: JsonDict) -> JsonDict:
    qa = job["qa"]
    response = text(result.get("response", ""), "response")
    scorer = obj(plan["scorer"], "plan scorer")
    error = result.get("error")
    if not error and not response:
        error = "empty response"
    if not error and result.get("served_model") != plan["model"]:
        error = (
            f"unexpected served model {result.get('served_model')!r}; expected {plan['model']!r}"
        )
    if not error and result.get("provider") != "novita":
        error = f"unexpected provider {result.get('provider')!r}; expected 'novita'"
    reasoning_tokens = usage_reasoning_tokens(obj(result.get("usage") or {}, "usage"))
    if not error and reasoning_tokens:
        error = f"reasoning control violation: tokens={reasoning_tokens}"
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model_id": plan["model"],
        "served_model": result.get("served_model"),
        "provider": result.get("provider"),
        "format": FORMAT_NAME,
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
        "fact_digest": qa["fact_digest"],
        "representation_path": job["representation_path"],
        "representation_sha256": job["representation_sha256"],
        "prompt_characters": len(job["prompt"]),
        "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
        "latency_ms": result.get("latency_ms"),
        "usage": result.get("usage", {}),
        "error": error,
    }


def accepted_records(
    records: Sequence[JsonDict],
    *,
    jobs: Sequence[FormatJob],
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
        accepted.setdefault(
            question_id,
            recompute_record_score(record, job["qa"]),
        )
    return accepted


def admissible_record(
    record: JsonDict | None,
    *,
    job: FormatJob,
    plan: JsonDict,
) -> bool:
    if not record or record.get("error") or not record.get("response"):
        return False
    qa = job["qa"]
    scorer = obj(plan["scorer"], "plan scorer")
    return all(
        (
            record.get("format") == FORMAT_NAME,
            record.get("question_id") == qa["id"],
            record.get("sample_id") == qa["sample_id"],
            record.get("fact_digest") == qa["fact_digest"],
            record.get("expected") == qa["answer"],
            record.get("model_id") == plan["model"],
            record.get("served_model") == plan["model"],
            record.get("provider") == "novita",
            record.get("representation_sha256") == job["representation_sha256"],
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


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    normalized: list[JsonDict] = []
    for record in records:
        copy = dict(record)
        copy["score"] = score_strict_json_answer(
            obj(record["expected"], "expected answer"), text(record["response"], "response")
        )
        normalized.append(copy)
    by_category: dict[str, list[JsonDict]] = defaultdict(list)
    for record in normalized:
        by_category[text(record["category"], "category")].append(record)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "records": len(normalized),
        "complete": len(normalized) == plan["request_count"],
        "overall": summarize_group(normalized),
        "categories": {
            category: summarize_group(rows) for category, rows in sorted(by_category.items())
        },
    }
