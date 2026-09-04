#!/usr/bin/env python
"""Evaluate hosted VLMs on the strict 60-question raw-image cohort."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    score_strict_json_answer,
    strict_scorer_digest,
)
from evals.catan_board_bench.tokens import atlas_metadata
from scripts.eval_catan_board_bench_openrouter import call_novita
from scripts.render_catan_strict_vision_probe import (
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR,
    OUTPUT_SCHEMA as DATASET_SCHEMA,
    file_sha256,
    json_digest,
    read_jsonl,
    write_json,
)


load_dotenv()

JsonDict = dict[str, Any]
EVAL_SCHEMA = "catan_strict_vision_eval/v2"
SUITE_NAME = "strict_vision_probe_60"
SYSTEM_PROMPT = """You are answering strict engine-scored questions about an ordinary public Catan board screenshot.

Rules:
- Use only the raw screenshot and supplied fixed engine-atlas context.
- The screenshot is not annotated. JSON engine state is used only by the scorer and is not supplied to you.
- Txx, Nxx, Exx_yy, and Pxx are canonical engine identifiers with fixed board geometry.
- Read resources, dice numbers, robber state, road colors, buildings, and colors from the screenshot.
- Pointy-top directions are LEFT, RIGHT, UP-LEFT, UP-RIGHT, DOWN-LEFT, and DOWN-RIGHT.
- For nominal roll production: a settlement produces 1, a city produces 2, and a robber blocks its tile. Aggregate by color and resource.
- Return exactly one JSON object matching the requested shape. No markdown, prose, or extra keys.
- Use null for absent scalar values and [] for empty lists.
- Sort entity-ID lists lexicographically. Preserve tuple associations in lists of objects.
- Wrap color, resource, and building values in angle brackets, for example <BLUE>, <WOOD>, and <CITY>. Canonical entity IDs do not use angle brackets in JSON.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", required=True)
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
    metadata, manifest, all_questions = validate_dataset(args.dataset_dir)
    questions = select_questions(
        all_questions,
        categories=split_csv(args.categories),
        max_questions=args.max_questions,
    )
    if not questions:
        raise SystemExit("No questions selected")
    jobs = build_jobs(args.dataset_dir, manifest=manifest, questions=questions)

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


def validate_dataset(
    dataset_dir: Path,
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "schema": DATASET_SCHEMA,
        "board_count": 12,
        "question_count": 60,
        "rendered_images": 12,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"dataset metadata mismatch: {mismatches}")
    identity_projection = metadata.get("identity_projection") or {}
    if identity_projection.get("target") != "canonical engine tile/node/edge/port IDs":
        raise ValueError("dataset does not use canonical engine IDs")
    if identity_projection.get("image_contains_entity_labels") is not False:
        raise ValueError("dataset image annotation policy changed")
    if (metadata.get("render_variant") or {}).get("image_annotation") is not None:
        raise ValueError("raw-image dataset unexpectedly declares an annotation")

    source_dir = Path(metadata["source_dataset"])
    source_lock = metadata.get("source_lock")
    if not isinstance(source_lock, dict) or json_digest(source_lock) != metadata.get(
        "source_lock_sha256"
    ):
        raise ValueError("dataset source lock is missing or invalid")
    for reference, expected_sha256 in source_lock.items():
        if reference in {"metadata.json", "manifest.jsonl", "qa.jsonl"} or reference.startswith(
            ("aliases/", "facts/")
        ):
            path = source_dir / reference
        else:
            path = Path(reference)
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise ValueError(f"locked source changed: {path}")

    manifest = read_jsonl(dataset_dir / "manifest.jsonl")
    questions = read_jsonl(dataset_dir / "qa.jsonl")
    if json_digest(manifest) != metadata["visual_manifest_sha256"]:
        raise ValueError("visual manifest digest mismatch")
    if json_digest(questions) != metadata["question_payload_sha256"]:
        raise ValueError("question payload digest mismatch")
    manifest_by_sample = {row["sample_id"]: row for row in manifest}
    if len(manifest) != 12 or len(manifest_by_sample) != 12:
        raise ValueError("visual manifest must contain 12 unique boards")
    if len(questions) != 60 or len({row["id"] for row in questions}) != 60:
        raise ValueError("visual QA must contain 60 unique questions")

    for row in manifest:
        image_path = dataset_dir / row["image_path"]
        contract_path = dataset_dir / row["contract_path"]
        if file_sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"image hash mismatch: {image_path}")
        if file_sha256(contract_path) != row["contract_sha256"]:
            raise ValueError(f"contract hash mismatch: {contract_path}")
        contract = json.loads(contract_path.read_text())
        if json_digest(contract) != row["engine_state_sha256"]:
            raise ValueError(f"engine-state digest mismatch: {contract_path}")

    for question in questions:
        sample = manifest_by_sample.get(question["sample_id"])
        if sample is None:
            raise ValueError(f"question references unknown board: {question['id']}")
        if question.get("engine_state_sha256") != sample["engine_state_sha256"]:
            raise ValueError(f"question engine-state mismatch: {question['id']}")
        if question.get("identity_projection") != "canonical_engine_ids":
            raise ValueError(f"question identity projection mismatch: {question['id']}")
    return metadata, manifest, questions


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
    return selected


def build_jobs(
    dataset_dir: Path,
    *,
    manifest: Sequence[JsonDict],
    questions: Sequence[JsonDict],
) -> list[JsonDict]:
    manifest_by_sample = {row["sample_id"]: row for row in manifest}
    contract_cache: dict[str, JsonDict] = {}
    jobs = []
    for qa in questions:
        row = manifest_by_sample[qa["sample_id"]]
        contract_path = dataset_dir / row["contract_path"]
        if qa["sample_id"] not in contract_cache:
            contract_cache[qa["sample_id"]] = json.loads(contract_path.read_text())
        contract = contract_cache[qa["sample_id"]]
        prompt = build_prompt(contract, qa)
        if qa["answer_text"] in prompt:
            raise ValueError(f"expected answer leaked into prompt for {qa['id']}")
        jobs.append(
            {
                "qa": qa,
                "image_path": str(dataset_dir / row["image_path"]),
                "image_sha256": row["image_sha256"],
                "contract_path": str(contract_path),
                "contract_sha256": row["contract_sha256"],
                "prompt": prompt,
            }
        )
    if len(jobs) != len({job["qa"]["id"] for job in jobs}):
        raise ValueError("vision job question IDs are not unique")
    return jobs


def build_prompt(contract: JsonDict, qa: JsonDict) -> str:
    context = canonical_atlas_context(contract, qa)
    return (
        f"Fixed engine-atlas context:\n{context}\n\n"
        f"Question: {qa['question']}\n"
        f"Required JSON shape: {qa['output_schema']}\n"
        "Return exactly one JSON object."
    )


def canonical_atlas_context(contract: JsonDict, qa: JsonDict) -> str:
    lines = [tile_rows_text(contract)]
    category = qa["category"]
    target = qa["target"]
    if category in {"node_state", "node_adjacent_tiles"}:
        lines.append(node_anchor_text(target["node"]))
    elif category == "nodes_connected":
        lines.extend(node_anchor_text(node) for node in target["nodes"])
    elif category == "edge_state":
        lines.append(edge_anchor_text(target["edge"]))
    elif category == "port_occupancy":
        port_id = int(target["port"][1:])
        port = next(row for row in contract["ports"] if row["id"] == port_id)
        endpoint_text = " ".join(f"N{node_id:02d}" for node_id in port["attached_nodes"])
        lines.append(
            f"P{port_id:02d} is the port at cube {port['coord']} facing "
            f"{port['direction']} and has endpoints {endpoint_text}."
        )
    return "\n".join(lines)


def tile_rows_text(contract: JsonDict) -> str:
    rows: dict[int, list[JsonDict]] = defaultdict(list)
    for tile in contract["tiles"]:
        rows[tile["coord"][2]].append(tile)
    parts = []
    for row_index, z in enumerate(sorted(rows)):
        tiles = sorted(rows[z], key=lambda tile: tile["coord"][0])
        parts.append(
            f"row {row_index} left-to-right: " + " ".join(f"T{tile['id']:02d}" for tile in tiles)
        )
    return "Tile rows top-to-bottom: " + "; ".join(parts) + "."


def node_anchor_text(node_token: str) -> str:
    node_id = int(node_token[1:])
    candidates = []
    for tile in atlas_metadata()["tiles"]:
        for direction, candidate in tile["nodes"].items():
            if candidate == node_id:
                candidates.append((tile["id"], direction))
    if not candidates:
        raise ValueError(f"unknown canonical node: {node_token}")
    tile_id, direction = min(candidates)
    return f"{node_token} is the {direction} corner of T{tile_id:02d}."


def edge_anchor_text(edge_token: str) -> str:
    node_text = edge_token[1:]
    node_a, node_b = (int(value) for value in node_text.split("_"))
    target = tuple(sorted((node_a, node_b)))
    candidates = []
    for tile in atlas_metadata()["tiles"]:
        for direction, edge in tile["edges"].items():
            if tuple(sorted(edge)) == target:
                candidates.append((tile["id"], direction))
    if not candidates:
        raise ValueError(f"unknown canonical edge: {edge_token}")
    tile_id, direction = min(candidates)
    return (
        f"{edge_token} connects N{node_a:02d} and N{node_b:02d}; it is the "
        f"{direction} side of T{tile_id:02d}."
    )


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
        print(f"\n--- {jobs[0]['qa']['id']} ---\n{jobs[0]['prompt']}")
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
        if args.concurrency == 1:
            for completed, job in enumerate(pending, start=1):
                try:
                    result = call_job_with_hard_deadline(api_key, args, job)
                except Exception as exc:
                    result = {"error": str(exc)}
                persist_result(
                    output,
                    job=job,
                    result=result,
                    plan=plan,
                    accepted=accepted,
                    completed=completed,
                    pending_count=len(pending),
                )
        elif pending:
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
                    persist_result(
                        output,
                        job=job,
                        result=result,
                        plan=plan,
                        accepted=accepted,
                        completed=completed,
                        pending_count=len(pending),
                    )

    final_records = [accepted[job["qa"]["id"]] for job in jobs if job["qa"]["id"] in accepted]
    summary = summarize(final_records, plan)
    write_json(args.output_dir / "summary.json", summary)
    print("\nSummary")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


def persist_result(
    output,
    *,
    job: JsonDict,
    result: JsonDict,
    plan: JsonDict,
    accepted: dict[str, JsonDict],
    completed: int,
    pending_count: int,
) -> None:
    record = response_record(job, result, plan)
    output.write(json.dumps(record, sort_keys=True) + "\n")
    output.flush()
    if admissible_record(record, job=job, plan=plan):
        accepted.setdefault(job["qa"]["id"], recompute_record_score(record, job["qa"]))
    status = "ERR" if record["error"] else ("OK" if record["score"]["correct"] else "MISS")
    print(
        f"[{completed}/{pending_count}] {status} {job['qa']['id']} got={record['response'][:120]!r}"
    )


def call_job_with_hard_deadline(
    api_key: str,
    args: argparse.Namespace,
    job: JsonDict,
) -> JsonDict:
    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    context = multiprocessing.get_context(method)
    result_receiver, result_sender = context.Pipe(duplex=False)
    process = context.Process(
        target=call_job_in_child,
        args=(result_sender, api_key, args, job),
    )
    started = time.monotonic()
    process.start()
    result_sender.close()
    process.join(args.timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        result = {
            "error": f"request exceeded hard wall-clock timeout of {args.timeout}s",
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
    elif result_receiver.poll(timeout=1):
        result = result_receiver.recv()
    else:
        result = {
            "error": f"request worker exited without a result (exitcode={process.exitcode})",
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
    result_receiver.close()
    return result


def call_job_in_child(
    result_sender,
    api_key: str,
    args: argparse.Namespace,
    job: JsonDict,
) -> None:
    try:
        result_sender.send(call_job(api_key, args, job))
    except BaseException as exc:
        result_sender.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        result_sender.close()


def call_job(api_key: str, args: argparse.Namespace, job: JsonDict) -> JsonDict:
    try:
        return call_novita(
            api_key,
            args.model,
            image_bytes=Path(job["image_path"]).read_bytes(),
            prompt=job["prompt"],
            system_prompt=SYSTEM_PROMPT,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    finally:
        if args.request_interval:
            time.sleep(args.request_interval)


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
                    "engine_state_sha256": row["engine_state_sha256"],
                }
                for row in questions
            ]
        ),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "jobs": {
            job["qa"]["id"]: {
                "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
                "image_sha256": job["image_sha256"],
                "contract_sha256": job["contract_sha256"],
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
            "image_input": True,
            "image_size": metadata["image_size"],
            "image_annotation": None,
            "identity_projection": "canonical_engine_ids",
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
        accepted.setdefault(question_id, recompute_record_score(record, job["qa"]))
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
            record.get("question_id") == qa["id"],
            record.get("sample_id") == qa["sample_id"],
            record.get("engine_state_sha256") == qa["engine_state_sha256"],
            record.get("expected") == qa["answer"],
            record.get("model_id") == plan["model"],
            record.get("served_model") == plan["model"],
            record.get("provider") == "novita",
            record.get("image_sha256") == job["image_sha256"],
            record.get("contract_sha256") == job["contract_sha256"],
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
        "protocol_exact_accuracy": protocol_exact / len(successful) if successful else 0.0,
        "prompt_tokens": sum(item.get("prompt_tokens", 0) or 0 for item in usage),
        "completion_tokens": sum(item.get("completion_tokens", 0) or 0 for item in usage),
        "reasoning_tokens": sum(usage_reasoning_tokens(item) for item in usage),
        "providers": dict(Counter(row["provider"] for row in successful)),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
    }


def usage_reasoning_tokens(usage: JsonDict) -> int:
    return int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)


def validate_or_write_plan(path: Path, plan: JsonDict) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        immutable_keys = (
            "schema",
            "suite",
            "model",
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


def acquire_run_lock(output_dir: Path) -> tuple[Path, int]:
    lock_path = output_dir / ".evaluation.lock"
    try:
        file_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f"Evaluation output is already locked: {lock_path}") from exc
    os.write(file_descriptor, f"pid={os.getpid()}\n".encode())
    return lock_path, file_descriptor


def release_run_lock(lock_path: Path, file_descriptor: int) -> None:
    os.close(file_descriptor)
    lock_path.unlink(missing_ok=True)


def percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
