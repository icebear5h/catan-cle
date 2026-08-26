#!/usr/bin/env python
"""Evaluate six lossless full-graph Catan text formats through OpenRouter."""

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
    STRICT_SCORER_VERSION,
    full_fact_digest,
    score_strict_json_answer,
    strict_scorer_digest,
)
from data_pipeline.catan_board_bench.full_graph_format_probe import (
    DATASET_SCHEMA,
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR,
)
from data_pipeline.catan_board_bench.full_graph_formats import (
    FORMAT_EXTENSIONS,
    FORMAT_NAMES,
    parse_full_graph_format,
)


load_dotenv()

JsonDict = Dict[str, Any]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EVAL_SCHEMA = "catan_full_graph_format_eval/v1"
SUITE_NAME = "full_graph_format_probe"
SYSTEM_PROMPT = """You are answering strict engine-scored questions about an authoritative public Catan board graph serialized as text.

Rules:
- Use only the supplied board graph. Entity IDs are opaque and board-local.
- Every one of the 19 tiles, 54 nodes, 72 edges, and 9 ports is represented. Native null, NULL, none, or dash values mean absent/empty.
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
    parser.add_argument("--formats", default=",".join(FORMAT_NAMES))
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
    formats = split_csv(args.formats)
    unknown = set(formats) - set(FORMAT_NAMES)
    if unknown:
        raise SystemExit(f"Unknown formats: {sorted(unknown)}")
    if not formats:
        raise SystemExit("At least one format is required")
    if len(formats) != len(set(formats)):
        raise SystemExit("Duplicate formats are not allowed")
    if args.max_questions is not None and args.max_questions <= 0:
        raise SystemExit("--max-questions must be positive")
    if args.max_requests is not None and args.max_requests <= 0:
        raise SystemExit("--max-requests must be positive")

    dataset_metadata = json.loads((args.dataset_dir / "metadata.json").read_text())
    validate_dataset_metadata(dataset_metadata)
    all_questions = validate_dataset_integrity(
        args.dataset_dir,
        dataset_metadata,
    )
    questions = all_questions
    categories = split_csv(args.categories)
    if categories:
        questions = [row for row in questions if row["category"] in categories]
    if args.max_questions is not None:
        questions = questions[: args.max_questions]
    if not questions:
        raise SystemExit("No questions selected")

    provider_order = split_csv(args.provider_order)
    validate_request_contract(args.model, provider_order)
    jobs = build_jobs(
        args.dataset_dir,
        questions=questions,
        formats=formats,
        max_requests=args.max_requests,
    )
    if len({job_key(job) for job in jobs}) != len(jobs):
        raise SystemExit("Generated format/question job keys are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, lock_fd = acquire_run_lock(args.output_dir)
    try:
        _run_evaluation(
            args,
            dataset_metadata=dataset_metadata,
            questions=questions,
            formats=formats,
            jobs=jobs,
            provider_order=provider_order,
        )
    finally:
        release_run_lock(lock_path, lock_fd)


def _run_evaluation(
    args: argparse.Namespace,
    *,
    dataset_metadata: JsonDict,
    questions: Sequence[JsonDict],
    formats: Sequence[str],
    jobs: Sequence[JsonDict],
    provider_order: Sequence[str],
) -> None:
    plan = build_plan(
        args,
        dataset_metadata=dataset_metadata,
        questions=questions,
        formats=formats,
        jobs=jobs,
        provider_order=provider_order,
    )
    plan_path = args.output_dir / "plan.json"
    validate_or_write_plan(plan_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))

    if args.dry_run:
        for job in jobs[: min(6, len(jobs))]:
            print(f"\n--- {job['format']} / {job['qa']['id']} ---\n{job['prompt'][:5000]}")
        return

    response_path = args.output_dir / "responses.jsonl"
    attempts = read_jsonl(response_path) if response_path.exists() else []
    accepted = accepted_records(attempts, jobs=jobs, plan=plan)
    pending = [job for job in jobs if job_key(job) not in accepted]
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
                    if admissible_record(record, job=job, plan=plan):
                        accepted.setdefault(
                            job_key(job),
                            recompute_record_score(record, job["qa"]),
                        )
                    status = (
                        "ERR"
                        if record["error"]
                        else ("OK" if record["score"]["correct"] else "MISS")
                    )
                    print(
                        f"[{completed}/{len(pending)}] {status} "
                        f"{job['format']} {job['qa']['id']} "
                        f"got={record['response'][:100]!r}"
                    )

    final_records = [accepted[job_key(job)] for job in jobs if job_key(job) in accepted]
    summary = summarize(final_records, plan)
    write_json(args.output_dir / "summary.json", summary)
    print("\nSummary")
    print(json.dumps(summary["formats"], indent=2, sort_keys=True))


def validate_dataset_metadata(metadata: JsonDict) -> None:
    expected = {
        "schema": DATASET_SCHEMA,
        "formats": list(FORMAT_NAMES),
        "board_count": 12,
        "question_count": 60,
        "request_count": 60 * len(FORMAT_NAMES),
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "incident_list_format": False,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"Dataset metadata mismatch: {mismatches}")


def validate_dataset_integrity(
    dataset_dir: Path,
    metadata: JsonDict,
) -> list[JsonDict]:
    source_lock = metadata.get("source_lock")
    if not isinstance(source_lock, dict):
        raise SystemExit("Dataset source lock is missing")
    if _json_digest(source_lock) != metadata.get("source_lock_sha256"):
        raise SystemExit("Dataset source-lock digest mismatch")

    source_dir = Path(metadata["source_dataset"])
    for relative_path, expected_sha256 in source_lock.items():
        source_path = source_dir / relative_path
        if not source_path.is_file():
            raise SystemExit(f"Locked source file is missing: {source_path}")
        if _file_sha256(source_path) != expected_sha256:
            raise SystemExit(f"Locked source file changed: {source_path}")

        generated_path: Path | None = None
        if relative_path == "qa.jsonl":
            generated_path = dataset_dir / "qa.jsonl"
        elif relative_path.startswith("facts/"):
            generated_path = dataset_dir / relative_path
        elif relative_path.startswith("aliases/"):
            generated_path = dataset_dir / relative_path
        elif relative_path.endswith("/tile_rows.txt"):
            generated_path = dataset_dir / relative_path
        if generated_path is not None:
            if not generated_path.is_file():
                raise SystemExit(f"Generated locked file is missing: {generated_path}")
            if _file_sha256(generated_path) != expected_sha256:
                raise SystemExit(f"Generated locked file changed: {generated_path}")

    questions = read_jsonl(dataset_dir / "qa.jsonl")
    manifest = read_jsonl(dataset_dir / "manifest.jsonl")
    manifest_by_sample = {row["sample_id"]: row for row in manifest}
    if len(manifest) != 12 or len(manifest_by_sample) != 12:
        raise SystemExit("Generated manifest must contain 12 unique boards")
    if len(questions) != 60 or len({row["id"] for row in questions}) != 60:
        raise SystemExit("Generated QA must contain 60 unique questions")

    metadata_hashes = metadata.get("representation_hashes")
    if not isinstance(metadata_hashes, dict):
        raise SystemExit("Representation hash manifest is missing")
    expected_representation_paths = set()
    fact_digests = {}
    for sample_id, manifest_row in sorted(manifest_by_sample.items()):
        fact_path = dataset_dir / "facts" / f"{sample_id}.json"
        facts = json.loads(fact_path.read_text())
        digest = full_fact_digest(facts)
        fact_digests[sample_id] = digest
        if digest != manifest_row.get("fact_digest"):
            raise SystemExit(f"Fact/manifest digest mismatch for {sample_id}")
        if set(metadata_hashes.get(sample_id, {})) != set(FORMAT_NAMES):
            raise SystemExit(f"Incomplete representation hashes for {sample_id}")

        metrics = manifest_row.get("representation_metrics") or {}
        for format_name in FORMAT_NAMES:
            extension = FORMAT_EXTENSIONS[format_name]
            representation_path = (
                dataset_dir / "representations" / sample_id / f"{format_name}{extension}"
            )
            expected_representation_paths.add(representation_path.resolve())
            if not representation_path.is_file():
                raise SystemExit(f"Representation is missing: {representation_path}")
            actual_sha256 = _file_sha256(representation_path)
            if actual_sha256 != metadata_hashes[sample_id][format_name]:
                raise SystemExit(f"Representation hash mismatch: {representation_path}")
            if actual_sha256 != (metrics.get(format_name) or {}).get("sha256"):
                raise SystemExit(f"Representation/manifest hash mismatch: {representation_path}")
            text = representation_path.read_text().rstrip("\n")
            try:
                parsed = parse_full_graph_format(format_name, text)
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Representation failed to parse: {representation_path}: {exc}"
                ) from exc
            if full_fact_digest(parsed) != digest:
                raise SystemExit(f"Representation fact mismatch: {representation_path}")

    actual_representation_paths = {
        path.resolve() for path in (dataset_dir / "representations").glob("*/*") if path.is_file()
    }
    if actual_representation_paths != expected_representation_paths:
        raise SystemExit("Unexpected or missing representation files")
    if set(metadata_hashes) != set(manifest_by_sample):
        raise SystemExit("Representation hash sample set mismatch")

    for question in questions:
        sample_id = question.get("sample_id")
        if sample_id not in fact_digests:
            raise SystemExit(f"Question references unknown board: {sample_id}")
        if question.get("fact_digest") != fact_digests[sample_id]:
            raise SystemExit(f"Question fact digest mismatch: {question.get('id')}")
    return questions


def validate_request_contract(
    model_id: str,
    provider_order: Sequence[str],
) -> None:
    if len(provider_order) != 1:
        raise SystemExit("Exactly one pinned provider is required")
    if not model_supports_reasoning_control(model_id):
        raise SystemExit(
            "This suite requires a model with an explicit hard reasoning-disable contract"
        )


def acquire_run_lock(output_dir: Path) -> tuple[Path, int]:
    lock_path = output_dir / ".evaluation.lock"
    try:
        file_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise SystemExit(f"Evaluation output is already locked: {lock_path}") from exc
    os.write(file_descriptor, f"pid={os.getpid()}\n".encode())
    return lock_path, file_descriptor


def release_run_lock(lock_path: Path, file_descriptor: int) -> None:
    os.close(file_descriptor)
    lock_path.unlink(missing_ok=True)


def build_jobs(
    dataset_dir: Path,
    *,
    questions: Sequence[JsonDict],
    formats: Sequence[str],
    max_requests: int | None,
) -> list[JsonDict]:
    if len(formats) != len(set(formats)):
        raise ValueError("duplicate formats are not allowed")
    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive")
    jobs = []
    for question_index, qa in enumerate(questions):
        rotation = question_index % len(formats)
        rotated = [*formats[rotation:], *formats[:rotation]]
        for format_name in rotated:
            extension = FORMAT_EXTENSIONS[format_name]
            representation_path = (
                dataset_dir / "representations" / qa["sample_id"] / f"{format_name}{extension}"
            )
            representation_bytes = representation_path.read_bytes()
            board_text = representation_bytes.decode().rstrip("\n")
            prompt = build_prompt(format_name, board_text, qa)
            if qa["answer_text"] in prompt:
                raise ValueError(f"expected answer leaked into prompt for {format_name}/{qa['id']}")
            jobs.append(
                {
                    "format": format_name,
                    "qa": qa,
                    "representation_path": str(representation_path),
                    "representation_sha256": hashlib.sha256(representation_bytes).hexdigest(),
                    "prompt": prompt,
                }
            )
            if max_requests is not None and len(jobs) >= max_requests:
                return jobs
    return jobs


def build_prompt(format_name: str, board_text: str, qa: JsonDict) -> str:
    return (
        f"Representation format: {format_name}\n"
        "Authoritative public board graph begins:\n"
        f"{board_text}\n"
        "Authoritative public board graph ends.\n\n"
        f"Question: {qa['question']}\n"
        f"Required JSON shape: {qa['output_schema']}\n"
        "Return exactly one JSON object."
    )


def build_plan(
    args: argparse.Namespace,
    *,
    dataset_metadata: JsonDict,
    questions: Sequence[JsonDict],
    formats: Sequence[str],
    jobs: Sequence[JsonDict],
    provider_order: Sequence[str],
) -> JsonDict:
    question_payload = [
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
    manifest_payload = {
        "dataset_metadata_sha256": _json_digest(dataset_metadata),
        "source_lock_sha256": dataset_metadata["source_lock_sha256"],
        "question_payload_sha256": _json_digest(question_payload),
        "fact_digests": sorted({row["fact_digest"] for row in questions}),
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "formats": list(formats),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "jobs": {
            f"{job['format']}:{job['qa']['id']}": {
                "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
                "representation_sha256": job["representation_sha256"],
                "answer_sha256": _json_digest(job["qa"]["answer"]),
            }
            for job in jobs
        },
    }
    reasoning_disabled = model_supports_reasoning_control(args.model)
    return {
        "schema": EVAL_SCHEMA,
        "suite": SUITE_NAME,
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "formats": list(formats),
        "categories": sorted({row["category"] for row in questions}),
        "question_count": len(questions),
        "request_count": len(jobs),
        "question_ids": [row["id"] for row in questions],
        "manifest_sha256": _json_digest(manifest_payload),
        "scorer": {
            "version": STRICT_SCORER_VERSION,
            "sha256": strict_scorer_digest(),
        },
        "system_prompt": SYSTEM_PROMPT,
        "request_settings": {
            "provider": "openrouter",
            "provider_order": list(provider_order),
            "allow_fallbacks": False,
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "requested_max_tokens": args.max_tokens,
            "effective_max_tokens": (
                max(args.max_tokens, 256) if reasoning_disabled else args.max_tokens
            ),
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
            "formats",
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
    if not error and result.get("provider") not in expected_providers:
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
        "format": job["format"],
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
        "representation_sha256": job["representation_sha256"],
        "prompt_characters": len(job["prompt"]),
        "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
        "latency_ms": result.get("latency_ms"),
        "usage": result.get("usage", {}),
        "finish_reason": result.get("finish_reason"),
        "native_reasoning": result.get("native_reasoning"),
        "request_id": result.get("request_id"),
        "error": error,
    }


def accepted_records(
    records: Sequence[JsonDict],
    *,
    jobs: Sequence[JsonDict],
    plan: JsonDict,
) -> dict[tuple[str, str], JsonDict]:
    jobs_by_key = {job_key(job): job for job in jobs}
    accepted = {}
    for record in records:
        key = (record.get("format"), record.get("question_id"))
        job = jobs_by_key.get(key)
        if job is None or not admissible_record(record, job=job, plan=plan):
            continue
        accepted.setdefault(
            key,
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
    expected_providers = plan["request_settings"]["provider_order"]
    checks = (
        record.get("format") == job["format"],
        record.get("question_id") == qa["id"],
        record.get("sample_id") == qa["sample_id"],
        record.get("fact_digest") == qa["fact_digest"],
        record.get("expected") == qa["answer"],
        record.get("model_id") == plan["model"],
        record.get("served_model") == plan["model"],
        record.get("provider") in expected_providers,
        record.get("representation_sha256") == job["representation_sha256"],
        record.get("prompt_sha256") == hashlib.sha256(job["prompt"].encode()).hexdigest(),
        record.get("scorer_version") == plan["scorer"]["version"],
        record.get("scorer_sha256") == plan["scorer"]["sha256"],
        not plan["request_settings"]["reasoning_disabled"]
        or (
            usage_reasoning_tokens(record.get("usage") or {}) == 0
            and not record.get("native_reasoning")
        ),
    )
    return all(checks)


def recompute_record_score(record: JsonDict, qa: JsonDict) -> JsonDict:
    normalized = dict(record)
    normalized["score"] = score_strict_json_answer(
        qa["answer"],
        record["response"],
    )
    return normalized


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
        "provider": {
            "order": list(provider_order),
            "allow_fallbacks": False,
        },
    }
    if model_supports_reasoning_control(model_id):
        payload["reasoning"] = {"enabled": False}
        payload["max_tokens"] = max(max_tokens, 256)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "CatanBoardBench Full-Graph Format Probe",
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
    normalized = []
    answers = {(record["format"], record["question_id"]): record["expected"] for record in records}
    for record in records:
        expected = answers[(record["format"], record["question_id"])]
        copy = dict(record)
        copy["score"] = score_strict_json_answer(expected, record["response"])
        normalized.append(copy)

    by_format: dict[str, list[JsonDict]] = defaultdict(list)
    for record in normalized:
        by_format[record["format"]].append(record)
    formats = {}
    for format_name, rows in sorted(by_format.items()):
        result = summarize_group(rows)
        by_category: dict[str, list[JsonDict]] = defaultdict(list)
        for row in rows:
            by_category[row["category"]].append(row)
        result["categories"] = {
            category: summarize_group(category_rows)
            for category, category_rows in sorted(by_category.items())
        }
        formats[format_name] = result
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "records": len(normalized),
        "complete": len(normalized) == plan["request_count"],
        "overall": summarize_group(normalized),
        "formats": formats,
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


def usage_reasoning_tokens(usage: JsonDict) -> int:
    return int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)


def job_key(job: JsonDict) -> tuple[str, str]:
    return (job["format"], job["qa"]["id"])


def model_supports_reasoning_control(model_id: str) -> bool:
    value = model_id.lower()
    return any(family in value for family in ("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8"))


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


if __name__ == "__main__":
    main()
