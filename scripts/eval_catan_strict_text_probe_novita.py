#!/usr/bin/env python
"""Evaluate the frozen strict 60-question text projection through Novita."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv

from evals.catan_board_bench import text_format_optimization as optimization
from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    score_strict_json_answer,
    strict_scorer_digest,
)
from scripts import eval_catan_board_bench_full_graph_formats as text_evaluator
from scripts.eval_catan_board_bench_openrouter import (
    NOVITA_URL,
    extract_message_text,
)
from scripts.eval_catan_strict_vision_probe import (
    acquire_run_lock,
    release_run_lock,
    summarize_group,
    usage_reasoning_tokens,
)
from scripts.render_catan_strict_vision_probe import (
    json_digest,
    read_jsonl,
    write_json,
)


load_dotenv()

JsonDict = dict[str, Any]
EVAL_SCHEMA = "catan_strict_text_novita_eval/v1"
SUITE_NAME = "strict_text_probe_60"
FORMAT_NAME = "indexed_tile_rows"
SYSTEM_PROMPT = text_evaluator.SYSTEM_PROMPT
DEFAULT_DATASET_DIR = optimization.DEFAULT_OUTPUT_DIR


@contextlib.contextmanager
def configured_text_evaluator():
    overrides = {
        "DATASET_SCHEMA": optimization.DATASET_SCHEMA,
        "DEFAULT_DATASET_DIR": optimization.DEFAULT_OUTPUT_DIR,
        "FORMAT_NAMES": optimization.FORMAT_NAMES,
        "FORMAT_EXTENSIONS": optimization.FORMAT_EXTENSIONS,
        "parse_full_graph_format": optimization.parse_text_format,
        "EVAL_SCHEMA": optimization.EVAL_SCHEMA,
        "SUITE_NAME": optimization.SUITE_NAME,
    }
    previous = {name: getattr(text_evaluator, name) for name in overrides}
    for name, value in overrides.items():
        setattr(text_evaluator, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(text_evaluator, name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", default="qwen/qwen3.8-max")
    parser.add_argument("--categories", default="")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--request-interval", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_cli_args(args)
    with configured_text_evaluator():
        metadata = json.loads((args.dataset_dir / "metadata.json").read_text())
        text_evaluator.validate_dataset_metadata(metadata)
        all_questions = text_evaluator.validate_dataset_integrity(args.dataset_dir, metadata)
        questions = select_questions(
            all_questions,
            categories=split_csv(args.categories),
            max_questions=args.max_questions,
        )
        jobs = text_evaluator.build_jobs(
            args.dataset_dir,
            questions=questions,
            formats=[FORMAT_NAME],
            max_requests=None,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, lock_fd = acquire_run_lock(args.output_dir)
    try:
        run_evaluation(args, metadata=metadata, questions=questions, jobs=jobs)
    finally:
        release_run_lock(lock_path, lock_fd)
    return 0


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.max_questions is not None and args.max_questions <= 0:
        raise SystemExit("--max-questions must be positive")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.max_tokens <= 0 or args.timeout <= 0:
        raise SystemExit("--max-tokens and --timeout must be positive")
    if args.request_interval < 0:
        raise SystemExit("--request-interval must be non-negative")


def select_questions(
    questions: Sequence[JsonDict],
    *,
    categories: Sequence[str],
    max_questions: int | None,
) -> list[JsonDict]:
    selected = list(questions)
    if categories:
        category_set = set(categories)
        unknown = category_set - {row["category"] for row in questions}
        if unknown:
            raise ValueError(f"unknown categories: {sorted(unknown)}")
        selected = [row for row in selected if row["category"] in category_set]
    if max_questions is not None:
        selected = selected[:max_questions]
    if not selected:
        raise ValueError("no questions selected")
    return selected


def run_evaluation(
    args: argparse.Namespace,
    *,
    metadata: JsonDict,
    questions: Sequence[JsonDict],
    jobs: Sequence[JsonDict],
) -> None:
    plan = build_plan(args, metadata=metadata, questions=questions, jobs=jobs)
    validate_or_write_plan(args.output_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        print(f"\n--- {jobs[0]['qa']['id']} ---\n{jobs[0]['prompt'][:8000]}")
        return

    api_key = os.getenv("NOVITA_API_KEY")
    if not api_key:
        raise SystemExit("NOVITA_API_KEY is not set")
    response_path = args.output_dir / "responses.jsonl"
    existing = read_jsonl(response_path) if args.resume and response_path.exists() else []
    accepted = accepted_records(existing, jobs=jobs, plan=plan)
    pending = [job for job in jobs if job["qa"]["id"] not in accepted]
    print(
        f"Running {len(pending)} pending of {len(jobs)} requests with concurrency={args.concurrency}"
    )

    mode = "a" if args.resume else "w"
    with response_path.open(mode) as output:
        if pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(call_job, api_key, args, job): job for job in pending}
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
                    if admissible_record(record, job=job, plan=plan):
                        accepted.setdefault(
                            job["qa"]["id"],
                            recompute_record_score(record, job["qa"]),
                        )
                    status = (
                        "ERR"
                        if record["error"]
                        else ("OK" if record["score"]["correct"] else "MISS")
                    )
                    print(
                        f"[{completed}/{len(pending)}] {status} {job['qa']['id']} "
                        f"got={record['response'][:120]!r}"
                    )

    final_records = [accepted[job["qa"]["id"]] for job in jobs if job["qa"]["id"] in accepted]
    summary = summarize(final_records, plan)
    write_json(args.output_dir / "summary.json", summary)
    print("\nSummary")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


def call_job(api_key: str, args: argparse.Namespace, job: JsonDict) -> JsonDict:
    try:
        return call_novita_text(
            api_key,
            args.model,
            prompt=job["prompt"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    finally:
        if args.request_interval:
            time.sleep(args.request_interval)


def call_novita_text(
    api_key: str,
    model_id: str,
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> JsonDict:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(NOVITA_URL, headers=headers, json=payload)
    latency_ms = int((time.time() - start) * 1000)
    if response.status_code >= 400:
        return {
            "error": f"HTTP {response.status_code}: {response.text[:800]}",
            "latency_ms": latency_ms,
        }
    data = response.json()
    message = data["choices"][0]["message"]
    return {
        "response": extract_message_text(message).strip(),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model", model_id),
        "provider": "novita",
    }


def build_plan(
    args: argparse.Namespace,
    *,
    metadata: JsonDict,
    questions: Sequence[JsonDict],
    jobs: Sequence[JsonDict],
) -> JsonDict:
    manifest_payload = {
        "dataset_metadata_sha256": json_digest(metadata),
        "question_payload_sha256": json_digest(
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
            ]
        ),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "jobs": {
            job["qa"]["id"]: {
                "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
                "representation_sha256": job["representation_sha256"],
                "answer_sha256": json_digest(job["qa"]["answer"]),
            }
            for job in jobs
        },
    }
    return {
        "schema": EVAL_SCHEMA,
        "suite": SUITE_NAME,
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "format": FORMAT_NAME,
        "categories": sorted({row["category"] for row in questions}),
        "question_count": len(questions),
        "request_count": len(jobs),
        "question_ids": [row["id"] for row in questions],
        "manifest_sha256": json_digest(manifest_payload),
        "scorer": {
            "version": STRICT_SCORER_VERSION,
            "sha256": strict_scorer_digest(),
        },
        "system_prompt": SYSTEM_PROMPT,
        "request_settings": {
            "provider": "novita",
            "provider_routing": "novita_direct",
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "requested_max_tokens": args.max_tokens,
            "reasoning_mode": "disabled_for_all",
            "timeout_seconds": args.timeout,
            "request_interval_seconds": args.request_interval,
            "image_input": False,
            "response_format_api_constraint": False,
            "strict_local_json_scoring": True,
        },
    }


def response_record(job: JsonDict, result: JsonDict, plan: JsonDict) -> JsonDict:
    qa = job["qa"]
    response = result.get("response", "")
    error = result.get("error")
    if not error and not response:
        error = "empty response"
    if not error and result.get("served_model") != plan["model"]:
        error = (
            f"unexpected served model {result.get('served_model')!r}; expected {plan['model']!r}"
        )
    if not error and result.get("provider") != "novita":
        error = f"unexpected provider {result.get('provider')!r}; expected 'novita'"
    reasoning_tokens = usage_reasoning_tokens(result.get("usage") or {})
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
        "score": score_strict_json_answer(qa["answer"], response),
        "scorer_version": plan["scorer"]["version"],
        "scorer_sha256": plan["scorer"]["sha256"],
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
    jobs: Sequence[JsonDict],
    plan: JsonDict,
) -> dict[str, JsonDict]:
    jobs_by_id = {job["qa"]["id"]: job for job in jobs}
    accepted = {}
    for record in records:
        question_id = record.get("question_id")
        job = jobs_by_id.get(question_id)
        if job is None or not admissible_record(record, job=job, plan=plan):
            continue
        accepted.setdefault(
            question_id,
            recompute_record_score(record, job["qa"]),
        )
    return accepted


def admissible_record(
    record: JsonDict | None,
    *,
    job: JsonDict,
    plan: JsonDict,
) -> bool:
    if not record or record.get("error") or not record.get("response"):
        return False
    qa = job["qa"]
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
            record.get("scorer_version") == plan["scorer"]["version"],
            record.get("scorer_sha256") == plan["scorer"]["sha256"],
            usage_reasoning_tokens(record.get("usage") or {}) == 0,
        )
    )


def recompute_record_score(record: JsonDict, qa: JsonDict) -> JsonDict:
    normalized = dict(record)
    normalized["score"] = score_strict_json_answer(qa["answer"], record["response"])
    return normalized


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    normalized = []
    for record in records:
        copy = dict(record)
        copy["score"] = score_strict_json_answer(record["expected"], record["response"])
        normalized.append(copy)
    by_category: dict[str, list[JsonDict]] = defaultdict(list)
    for record in normalized:
        by_category[record["category"]].append(record)
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


def validate_or_write_plan(path: Path, plan: JsonDict) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        immutable_keys = (
            "schema",
            "suite",
            "model",
            "format",
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


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
