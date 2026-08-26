#!/usr/bin/env python
"""Evaluate strict Catan full-graph ASCII variations through OpenRouter."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import httpx
from dotenv import load_dotenv

from data_pipeline.catan_board_bench.ascii_variations import (
    ASCII_VARIANTS,
    STRICT_SCORER_VERSION,
    score_strict_json_answer,
    strict_scorer_digest,
)


load_dotenv()

JsonDict = Dict[str, Any]
DEFAULT_DATASET_DIR = Path("data_pipeline/catan_board_bench/datasets/ascii_variation_probe")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = """You are answering strict engine-scored questions about an authoritative public Catan board graph encoded as ASCII text.

Rules:
- Use only the supplied board graph. Entity IDs are opaque and board-local.
- Every tile, node, edge, and port record is explicit. A dash (-) means absent/empty.
- Pointy-top cube direction deltas are:
  LEFT=(-1,+1,0), RIGHT=(+1,-1,0),
  UP-LEFT=(0,+1,-1), UP-RIGHT=(+1,0,-1),
  DOWN-LEFT=(-1,0,+1), DOWN-RIGHT=(0,-1,+1).
- For nominal roll production: a settlement produces 1, a city produces 2, and a robber blocks its tile. Aggregate by color and resource.
- Return exactly one JSON object matching the requested shape. No markdown, prose, or extra keys.
- Use null for absent scalar values and [] for empty lists.
- Sort entity-ID lists lexicographically. Preserve tuple associations in lists of objects.
- Wrap color, resource, and building values in angle brackets, for example <BLUE>, <WOOD>, and <CITY>. Board-local entity IDs such as T03, N14, and E27 do not use angle brackets.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--variants", default=",".join(ASCII_VARIANTS))
    parser.add_argument("--categories", default="")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--provider-order", default="AkashML")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = split_csv(args.variants)
    unknown = set(variants) - set(ASCII_VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")
    if not variants:
        raise SystemExit("At least one variant is required")

    questions = read_jsonl(args.dataset_dir / "qa.jsonl")
    categories = split_csv(args.categories)
    if categories:
        questions = [row for row in questions if row["category"] in categories]
    if args.max_questions is not None:
        questions = questions[: args.max_questions]
    if not questions:
        raise SystemExit("No questions selected")

    jobs = build_jobs(
        args.dataset_dir,
        questions=questions,
        variants=variants,
        max_requests=args.max_requests,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provider_order = split_csv(args.provider_order)
    plan = build_plan(
        args,
        questions=questions,
        variants=variants,
        jobs=jobs,
        provider_order=provider_order,
    )
    plan_path = args.output_dir / "plan.json"
    validate_or_write_plan(plan_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))

    if args.dry_run:
        for job in jobs[: min(6, len(jobs))]:
            print(f"\n--- {job['variant']} / {job['qa']['id']} ---\n{job['prompt'][:5000]}")
        return

    response_path = args.output_dir / "responses.jsonl"
    prior_records = read_jsonl(response_path) if response_path.exists() else []
    latest = latest_records(prior_records)
    pending = [
        job
        for job in jobs
        if not successful_record(
            latest.get(job_key(job)),
            job=job,
            plan=plan,
        )
    ]
    print(
        f"Running {len(pending)} pending of {len(jobs)} requests "
        f"with concurrency={args.concurrency}"
    )

    if pending:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set")
        with response_path.open("a") as output:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(
                        call_openrouter_text,
                        api_key,
                        args.model,
                        prompt=job["prompt"],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        provider_order=provider_order,
                    ): job
                    for job in pending
                }
                for completed, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    job = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"error": str(exc)}
                    record = response_record(job, result, plan)
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    latest[job_key(job)] = record
                    status = (
                        "ERR"
                        if record["error"]
                        else ("OK" if record["score"]["correct"] else "MISS")
                    )
                    print(
                        f"[{completed}/{len(pending)}] {status} "
                        f"{job['variant']} {job['qa']['id']} "
                        f"got={record['response'][:100]!r}"
                    )

    final_records = [latest[job_key(job)] for job in jobs if job_key(job) in latest]
    summary = summarize(final_records, plan)
    write_json(args.output_dir / "summary.json", summary)
    print("\nSummary")
    print(json.dumps(summary["variants"], indent=2, sort_keys=True))


def build_jobs(
    dataset_dir: Path,
    *,
    questions: Sequence[JsonDict],
    variants: Sequence[str],
    max_requests: int | None,
) -> list[JsonDict]:
    jobs = []
    for question_index, qa in enumerate(questions):
        rotation = question_index % len(variants)
        rotated = [*variants[rotation:], *variants[:rotation]]
        for variant in rotated:
            representation_path = (
                dataset_dir / "representations" / qa["sample_id"] / f"{variant}.txt"
            )
            board_text = representation_path.read_text().rstrip("\n")
            jobs.append(
                {
                    "variant": variant,
                    "qa": qa,
                    "representation_path": str(representation_path),
                    "prompt": build_prompt(variant, board_text, qa),
                }
            )
            if max_requests is not None and len(jobs) >= max_requests:
                return jobs
    return jobs


def build_prompt(variant: str, board_text: str, qa: JsonDict) -> str:
    return (
        f"ASCII representation variant: {variant}\n"
        "Authoritative public board graph:\n"
        f"{board_text}\n\n"
        f"Question: {qa['question']}\n"
        f"Required JSON shape: {qa['output_schema']}\n"
        "Return exactly one JSON object."
    )


def build_plan(
    args: argparse.Namespace,
    *,
    questions: Sequence[JsonDict],
    variants: Sequence[str],
    jobs: Sequence[JsonDict],
    provider_order: Sequence[str],
) -> JsonDict:
    manifest_payload = {
        "dataset_metadata": json.loads((args.dataset_dir / "metadata.json").read_text()),
        "question_ids": [row["id"] for row in questions],
        "question_payload_sha256": hashlib.sha256(
            json.dumps(
                [
                    {
                        "id": row["id"],
                        "sample_id": row["sample_id"],
                        "category": row["category"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "answer_text": row["answer_text"],
                        "output_schema": row["output_schema"],
                        "fact_digest": row["fact_digest"],
                    }
                    for row in questions
                ],
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "fact_digests": sorted({row["fact_digest"] for row in questions}),
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "variants": list(variants),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "job_prompt_sha256": {
            f"{job['variant']}:{job['qa']['id']}": hashlib.sha256(
                job["prompt"].encode()
            ).hexdigest()
            for job in jobs
        },
    }
    reasoning_disabled = model_supports_reasoning_control(args.model)
    return {
        "schema": "catan_ascii_variation_eval/v1",
        "suite": "ascii_variation_probe",
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "variants": list(variants),
        "categories": sorted({row["category"] for row in questions}),
        "question_count": len(questions),
        "request_count": len(jobs),
        "question_ids": [row["id"] for row in questions],
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                manifest_payload,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "scorer": {
            "version": STRICT_SCORER_VERSION,
            "sha256": strict_scorer_digest(),
        },
        "system_prompt": SYSTEM_PROMPT,
        "request_settings": {
            "provider": "openrouter",
            "provider_order": list(provider_order),
            "allow_fallbacks": False if provider_order else None,
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "requested_max_tokens": args.max_tokens,
            "effective_max_tokens": max(args.max_tokens, 256)
            if reasoning_disabled
            else args.max_tokens,
            "reasoning_disabled": reasoning_disabled,
            "timeout_seconds": args.timeout,
            "image_input": False,
            "response_format_api_constraint": False,
            "strict_local_json_scoring": True,
        },
    }


def validate_or_write_plan(path: Path, plan: JsonDict) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        immutable_keys = (
            "schema",
            "suite",
            "model",
            "variants",
            "question_ids",
            "request_count",
            "manifest_sha256",
            "scorer",
            "request_settings",
        )
        mismatched = [key for key in immutable_keys if previous.get(key) != plan.get(key)]
        if mismatched:
            raise SystemExit("Existing output plan is incompatible on: " + ", ".join(mismatched))
        return
    write_json(path, plan)


def response_record(
    job: JsonDict,
    result: JsonDict,
    plan: JsonDict,
) -> JsonDict:
    qa = job["qa"]
    response = result.get("response", "")
    score = score_strict_json_answer(qa["answer"], response)
    error = result.get("error")
    if not error and not response:
        error = "empty response"
    expected_model = plan["model"]
    expected_providers = plan["request_settings"]["provider_order"]
    if not error and result.get("served_model") != expected_model:
        error = (
            f"unexpected served model {result.get('served_model')!r}; expected {expected_model!r}"
        )
    if not error and expected_providers and result.get("provider") not in expected_providers:
        error = (
            f"unexpected provider {result.get('provider')!r}; "
            f"expected one of {expected_providers!r}"
        )
    if not error and plan["request_settings"]["reasoning_disabled"]:
        reasoning_tokens = usage_reasoning_tokens(result.get("usage") or {})
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
        "scorer_version": plan["scorer"]["version"],
        "scorer_sha256": plan["scorer"]["sha256"],
        "fact_digest": qa["fact_digest"],
        "representation_path": job["representation_path"],
        "prompt_characters": len(job["prompt"]),
        "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
        "latency_ms": result.get("latency_ms"),
        "usage": result.get("usage", {}),
        "finish_reason": result.get("finish_reason"),
        "native_reasoning": result.get("native_reasoning"),
        "request_id": result.get("request_id"),
        "error": error,
    }


def call_openrouter_text(
    api_key: str,
    model_id: str,
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_order: Sequence[str],
) -> JsonDict:
    payload: JsonDict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model_supports_reasoning_control(model_id):
        payload["reasoning"] = {"enabled": False}
        payload["max_tokens"] = max(max_tokens, 256)
    if provider_order:
        payload["provider"] = {
            "order": list(provider_order),
            "allow_fallbacks": False,
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "CatanBoardBench ASCII Variation Probe",
    }
    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
    latency_ms = int((time.time() - start) * 1000)
    if response.status_code >= 400:
        return {
            "requested_model": model_id,
            "error": f"HTTP {response.status_code}: {response.text[:800]}",
            "latency_ms": latency_ms,
        }
    data = response.json()
    choice = data["choices"][0]
    message = choice["message"]
    native_reasoning = {
        field: message.get(field)
        for field in ("reasoning", "reasoning_content", "reasoning_details")
        if message.get(field)
    }
    return {
        "requested_model": model_id,
        "response": extract_message_text(message),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model"),
        "provider": data.get("provider"),
        "finish_reason": choice.get("finish_reason"),
        "native_reasoning": native_reasoning,
        "request_id": data.get("id"),
    }


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    by_variant: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        by_variant[record["variant"]].append(record)
    variants = {}
    for variant, rows in sorted(by_variant.items()):
        result = summarize_group(rows)
        by_category: dict[str, list[JsonDict]] = defaultdict(list)
        for row in rows:
            by_category[row["category"]].append(row)
        result["categories"] = {
            category: summarize_group(category_rows)
            for category, category_rows in sorted(by_category.items())
        }
        variants[variant] = result
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "records": len(records),
        "complete": len(records) == plan["request_count"],
        "overall": summarize_group(records),
        "variants": variants,
    }


def summarize_group(records: Sequence[JsonDict]) -> JsonDict:
    successful = [row for row in records if not row.get("error")]
    latencies = sorted(row["latency_ms"] for row in successful if row.get("latency_ms") is not None)
    usage = [row.get("usage") or {} for row in successful]
    exact = sum(row["score"]["correct"] for row in successful)
    json_valid = sum(row["score"]["json_valid"] for row in successful)
    protocol_exact = sum(row["score"]["protocol_exact"] for row in successful)
    return {
        "requests": len(records),
        "successful": len(successful),
        "errors": len(records) - len(successful),
        "exact": exact,
        "exact_accuracy": exact / len(successful) if successful else 0.0,
        "json_valid": json_valid,
        "json_valid_accuracy": json_valid / len(successful) if successful else 0.0,
        "protocol_exact": protocol_exact,
        "protocol_exact_accuracy": (protocol_exact / len(successful) if successful else 0.0),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) or 0 for item in usage),
        "completion_tokens": sum(item.get("completion_tokens", 0) or 0 for item in usage),
        "reasoning_tokens": sum(usage_reasoning_tokens(item) for item in usage),
        "providers": dict(
            Counter(row["provider"] for row in successful if row.get("provider") is not None)
        ),
        "cost": sum(item.get("cost", 0.0) or 0.0 for item in usage),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95) if latencies else None,
    }


def latest_records(records: Sequence[JsonDict]) -> dict[tuple[str, str], JsonDict]:
    latest = {}
    for record in records:
        latest[(record["variant"], record["question_id"])] = record
    return latest


def successful_record(
    record: JsonDict | None,
    *,
    job: JsonDict,
    plan: JsonDict,
) -> bool:
    if not record or record.get("error") or not record.get("response"):
        return False
    qa = job["qa"]
    expected_provider = plan["request_settings"]["provider_order"]
    checks = (
        record.get("variant") == job["variant"],
        record.get("question_id") == qa["id"],
        record.get("sample_id") == qa["sample_id"],
        record.get("fact_digest") == qa["fact_digest"],
        record.get("expected") == qa["answer"],
        record.get("model_id") == plan["model"],
        record.get("served_model") == plan["model"],
        record.get("prompt_sha256") == hashlib.sha256(job["prompt"].encode()).hexdigest(),
        record.get("scorer_version") == plan["scorer"]["version"],
        record.get("scorer_sha256") == plan["scorer"]["sha256"],
        not expected_provider or record.get("provider") in expected_provider,
        not plan["request_settings"]["reasoning_disabled"]
        or (
            usage_reasoning_tokens(record.get("usage") or {}) == 0
            and not record.get("native_reasoning")
        ),
    )
    return all(checks)


def usage_reasoning_tokens(usage: JsonDict) -> int:
    return int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)


def job_key(job: JsonDict) -> tuple[str, str]:
    return (job["variant"], job["qa"]["id"])


def model_supports_reasoning_control(model_id: str) -> bool:
    value = model_id.lower()
    return any(
        family in value
        for family in (
            "qwen3.5",
            "qwen3.6",
            "qwen3.7",
            "qwen3.8",
        )
    )


def extract_message_text(message: JsonDict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content).strip()


def percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    index = max(0, math.ceil(fraction * len(values)) - 1)
    return values[index]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
