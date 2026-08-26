#!/usr/bin/env python
"""Evaluate isolated Catan tile prompts across label and guidance styles."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv

load_dotenv()

JsonDict = dict[str, Any]

SCHEMA = "catan_tile_prompt_ablation/v1"
SCORER_VERSION = "strict_tile_label/v2"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_QA_PATH = Path(
    "artifacts/runs/catan_board_bench/piece_recognition/"
    "qwen3_8_27b_piece_full_20260816/qa_snapshot.jsonl"
)
DEFAULT_IMAGE_ROOT = Path("artifacts/generated/catan_board_bench/piece_recognition")
DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_PROVIDER = "AkashML"
CATEGORY = "isolated_tile_resource_number"
RESOURCE_CLASSES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", "DESERT")
NUMBER_CLASSES = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
CONDITIONS = (
    "angle_labels",
    "plain_labels",
    "angle_described",
    "plain_described",
)
ANGLE_CONDITIONS = frozenset({"angle_labels", "angle_described"})
DESCRIBED_CONDITIONS = frozenset({"angle_described", "plain_described"})
DESCRIPTION_BY_RESOURCE = {
    "WOOD": "green tile with an evergreen tree",
    "BRICK": "orange-red tile with stacked brick blocks",
    "SHEEP": "yellow-green tile with a white sheep",
    "WHEAT": "golden-yellow tile with wheat stalks",
    "ORE": "gray tile with rocks",
    "DESERT": "beige tile with a cactus",
}
SYSTEM_PROMPT = """You are performing closed-set visual classification of one isolated Catan terrain tile.

Use only the supplied image.
Return exactly one line that follows the output contract.
Do not explain your answer or add any other text."""

ANGLE_NUMBER_RE = re.compile(
    r"\A<(WOOD|BRICK|SHEEP|WHEAT|ORE)> "
    r"(2|3|4|5|6|8|9|10|11|12)\Z"
)
ANGLE_DESERT_RE = re.compile(r"\A<DESERT>\Z")
PLAIN_NUMBER_RE = re.compile(
    r"\A(WOOD|BRICK|SHEEP|WHEAT|ORE) "
    r"(2|3|4|5|6|8|9|10|11|12)\Z"
)
PLAIN_DESERT_RE = re.compile(r"\ADESERT\Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider-order", default=DEFAULT_PROVIDER)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seed", type=int, default=38_271)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def json_digest(value: Any) -> str:
    return sha256_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )


def read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def is_angle(condition: str) -> bool:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return condition in ANGLE_CONDITIONS


def has_descriptions(condition: str) -> bool:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return condition in DESCRIBED_CONDITIONS


def format_label(resource: str, condition: str) -> str:
    return f"<{resource}>" if is_angle(condition) else resource


def build_prompt(condition: str) -> str:
    labels = ", ".join(format_label(resource, condition) for resource in RESOURCE_CLASSES)
    lines = [
        "Task: identify the terrain resource and read the production number "
        "printed on the tile.",
        "",
        f"Resource classes: {labels}.",
    ]
    if has_descriptions(condition):
        lines.extend(["", "Visual guide for these renderer assets:"])
        lines.extend(
            f"{format_label(resource, condition)}: "
            f"{DESCRIPTION_BY_RESOURCE[resource]}."
            for resource in RESOURCE_CLASSES
        )
    lines.extend(
        [
            "",
            "Write the selected resource class exactly as listed above.",
            "For a numbered resource tile, output the resource field, one ASCII "
            "space, and the number.",
            "For a desert tile, output only the resource field.",
            "Allowed numbers: 2, 3, 4, 5, 6, 8, 9, 10, 11, 12.",
            "",
            "What resource and production number are shown?",
        ]
    )
    return "\n".join(lines)


def truth_from_qa(qa: JsonDict) -> JsonDict:
    expected = str(qa.get("answer", "")).strip()
    match = re.fullmatch(
        r"<(WOOD|BRICK|SHEEP|WHEAT|ORE)> "
        r"(2|3|4|5|6|8|9|10|11|12)",
        expected,
    )
    if match:
        return {"resource": match.group(1), "number": int(match.group(2))}
    if expected == "<DESERT>":
        return {"resource": "DESERT", "number": None}
    raise ValueError(f"Unsupported tile answer: {expected!r}")


def expected_response(truth: JsonDict, condition: str) -> str:
    label = format_label(truth["resource"], condition)
    if truth["number"] is None:
        return label
    return f"{label} {truth['number']}"


def parse_response(response: str, *, angle: bool) -> JsonDict | None:
    number_re = ANGLE_NUMBER_RE if angle else PLAIN_NUMBER_RE
    desert_re = ANGLE_DESERT_RE if angle else PLAIN_DESERT_RE
    match = number_re.fullmatch(response)
    if match:
        return {"resource": match.group(1), "number": int(match.group(2))}
    if desert_re.fullmatch(response):
        return {"resource": "DESERT", "number": None}
    return None


def score_response(truth: JsonDict, response: str, condition: str) -> JsonDict:
    parsed = parse_response(response, angle=is_angle(condition))
    alternate = parse_response(response, angle=not is_angle(condition))
    protocol_valid = parsed is not None
    resource_correct = bool(parsed and parsed["resource"] == truth["resource"])
    if truth["number"] is None:
        number_correct: bool | None = None
    else:
        number_correct = bool(parsed and parsed["number"] == truth["number"])
    pair_correct = parsed == truth
    return {
        "protocol_valid": protocol_valid,
        "protocol_exact": response == expected_response(truth, condition),
        "resource_correct": resource_correct,
        "number_correct": number_correct,
        "pair_correct": pair_correct,
        "wrapper_rescued_content_correct": alternate == truth,
        "parsed": parsed,
    }


def scorer_sha256() -> str:
    payload = {
        "version": SCORER_VERSION,
        "sources": {
            function.__name__: inspect.getsource(function)
            for function in (
                score_response,
                parse_response,
                expected_response,
                format_label,
                is_angle,
            )
        },
        "regexes": {
            name: {"pattern": regex.pattern, "flags": regex.flags}
            for name, regex in (
                ("angle_number", ANGLE_NUMBER_RE),
                ("angle_desert", ANGLE_DESERT_RE),
                ("plain_number", PLAIN_NUMBER_RE),
                ("plain_desert", PLAIN_DESERT_RE),
            )
        },
        "conditions": CONDITIONS,
        "angle_conditions": sorted(ANGLE_CONDITIONS),
        "described_conditions": sorted(DESCRIBED_CONDITIONS),
        "resources": RESOURCE_CLASSES,
        "numbers": NUMBER_CLASSES,
    }
    return json_digest(payload)


def load_questions(qa_path: Path, image_root: Path) -> list[JsonDict]:
    rows = [row for row in read_jsonl(qa_path) if row.get("category") == CATEGORY]
    if len(rows) != 102 or len({row.get("id") for row in rows}) != 102:
        raise SystemExit("Expected 102 unique isolated tile questions")
    if len({row.get("sample_id") for row in rows}) != 102:
        raise SystemExit("Expected 102 unique isolated tile sample IDs")
    if len({row.get("image_path") for row in rows}) != 102:
        raise SystemExit("Expected 102 unique isolated tile image paths")
    selected = []
    for row in rows:
        truth = truth_from_qa(row)
        image_path = image_root / row["image_path"]
        if not image_path.is_file():
            raise SystemExit(f"Missing image: {image_path}")
        selected.append(
            {
                "id": row["id"],
                "sample_id": row["sample_id"],
                "category": row["category"],
                "image_path": str(image_path),
                "image_sha256": file_sha256(image_path),
                "source_row_sha256": json_digest(row),
                "truth": truth,
            }
        )
    resource_counts = Counter(row["truth"]["resource"] for row in selected)
    expected_counts = {resource: 20 for resource in RESOURCE_CLASSES[:-1]}
    expected_counts["DESERT"] = 2
    if dict(resource_counts) != expected_counts:
        raise SystemExit(f"Unexpected resource balance: {dict(resource_counts)}")
    cross_product = Counter(
        (row["truth"]["resource"], row["truth"]["number"])
        for row in selected
    )
    expected_cross_product = {
        (resource, number): 2
        for resource in RESOURCE_CLASSES[:-1]
        for number in NUMBER_CLASSES
    }
    expected_cross_product[("DESERT", None)] = 2
    if dict(cross_product) != expected_cross_product:
        raise SystemExit("Isolated tile resource/number cross-product changed")
    if len({row["image_sha256"] for row in selected}) != 102:
        raise SystemExit("Expected 102 unique isolated tile image hashes")
    return selected


def rotated_conditions(conditions: Sequence[str], offset: int) -> list[str]:
    index = offset % len(conditions)
    return list(conditions[index:]) + list(conditions[:index])


def build_jobs(
    questions: Sequence[JsonDict],
    *,
    conditions: Sequence[str],
    seed: int,
    preflight: bool = False,
    max_requests: int | None = None,
) -> list[JsonDict]:
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("Conditions must be nonempty and unique")
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if preflight and max_requests is not None:
        raise ValueError("Preflight cannot be combined with max_requests")

    shuffled = [dict(question) for question in questions]
    random.Random(seed).shuffle(shuffled)
    jobs = []
    for question_index, qa in enumerate(shuffled):
        for condition in rotated_conditions(conditions, question_index):
            prompt = build_prompt(condition)
            jobs.append(
                {
                    "condition": condition,
                    "qa": qa,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "expected_response": expected_response(qa["truth"], condition),
                }
            )

    if preflight:
        if tuple(conditions) != CONDITIONS:
            raise ValueError("Preflight requires all four canonical conditions")
        numbered = [qa for qa in shuffled if qa["truth"]["number"] is not None]
        deserts = [qa for qa in shuffled if qa["truth"]["number"] is None]
        selected_pairs = (
            ("angle_labels", numbered[0]),
            ("plain_labels", deserts[0]),
            ("angle_described", deserts[1]),
            ("plain_described", numbered[1]),
        )
        jobs = []
        for condition, qa in selected_pairs:
            prompt = build_prompt(condition)
            jobs.append(
                {
                    "condition": condition,
                    "qa": qa,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "expected_response": expected_response(qa["truth"], condition),
                }
            )
        if len(jobs) != 4 or {job["condition"] for job in jobs} != set(CONDITIONS):
            raise ValueError("Preflight must contain exactly one job per condition")
    if max_requests is not None:
        jobs = jobs[:max_requests]
    if len({job_key(job) for job in jobs}) != len(jobs):
        raise ValueError("Generated duplicate jobs")
    return jobs


def build_plan(
    args: argparse.Namespace,
    *,
    questions: Sequence[JsonDict],
    conditions: Sequence[str],
    provider_order: Sequence[str],
    jobs: Sequence[JsonDict],
) -> JsonDict:
    question_lock = [
        {
            "id": row["id"],
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "source_row_sha256": row["source_row_sha256"],
            "truth": row["truth"],
        }
        for row in questions
    ]
    job_lock = [
        {
            "condition": job["condition"],
            "question_id": job["qa"]["id"],
            "prompt_sha256": job["prompt_sha256"],
            "image_sha256": job["qa"]["image_sha256"],
        }
        for job in jobs
    ]
    return {
        "schema": SCHEMA,
        "model": args.model,
        "provider_order": list(provider_order),
        "conditions": list(conditions),
        "condition_prompts": {
            condition: {
                "text": build_prompt(condition),
                "sha256": sha256_text(build_prompt(condition)),
            }
            for condition in conditions
        },
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "qa_path": str(args.qa_path),
        "qa_file_sha256": file_sha256(args.qa_path),
        "image_root": str(args.image_root),
        "question_count": len(questions),
        "question_lock_sha256": json_digest(question_lock),
        "image_hashes": {
            row["sample_id"]: row["image_sha256"] for row in questions
        },
        "seed": args.seed,
        "preflight": bool(args.preflight),
        "request_count": len(jobs),
        "job_lock_sha256": json_digest(job_lock),
        "scorer": {
            "version": SCORER_VERSION,
            "sha256": scorer_sha256(),
        },
        "request_settings": {
            "temperature": args.temperature,
            "requested_max_tokens": args.max_tokens,
            "effective_max_tokens": max(args.max_tokens, 256),
            "timeout_seconds": args.timeout,
            "concurrency": args.concurrency,
            "reasoning_disabled": True,
            "allow_fallbacks": False,
            "image_input": True,
            "response_format_api_constraint": False,
        },
        "output_dir": str(args.output_dir),
    }


def validate_or_write_plan(path: Path, plan: JsonDict) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != plan:
            raise SystemExit("Existing plan does not match requested run")
        return
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def acquire_run_lock(output_dir: Path) -> tuple[Path, int]:
    lock_path = output_dir / ".evaluation.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f"Output directory is already locked: {lock_path}") from exc
    os.write(fd, f"pid={os.getpid()}\n".encode())
    return lock_path, fd


def release_run_lock(lock_path: Path, lock_fd: int) -> None:
    os.close(lock_fd)
    lock_path.unlink(missing_ok=True)


def validate_request_contract(model_id: str, provider_order: Sequence[str]) -> None:
    if len(provider_order) != 1:
        raise SystemExit("Exactly one OpenRouter provider is required")
    if not model_supports_reasoning_control(model_id):
        raise SystemExit(f"No explicit reasoning-disable control for model: {model_id}")


def model_supports_reasoning_control(model_id: str) -> bool:
    return model_id == DEFAULT_MODEL


def extract_message_text(message: JsonDict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def call_openrouter_image(
    api_key: str,
    model_id: str,
    *,
    image_bytes: bytes,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_order: Sequence[str],
) -> JsonDict:
    submitted_image_sha256 = sha256_bytes(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload: JsonDict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max(max_tokens, 256),
        "reasoning": {"enabled": False},
        "provider": {
            "order": list(provider_order),
            "allow_fallbacks": False,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "CatanBoardBench Tile Prompt Ablation",
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
            "submitted_image_sha256": submitted_image_sha256,
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
        "submitted_image_sha256": submitted_image_sha256,
    }


def reported_reasoning_tokens(usage: JsonDict) -> int | None:
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict) or "reasoning_tokens" not in details:
        return None
    value = details["reasoning_tokens"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not float(value).is_integer():
        return None
    return int(value)


def has_explicit_reasoning_usage(usage: JsonDict) -> bool:
    return reported_reasoning_tokens(usage) is not None


def usage_reasoning_tokens(usage: JsonDict) -> int:
    value = reported_reasoning_tokens(usage)
    if value is None:
        raise ValueError("reasoning-token usage field is invalid")
    return value


def job_key(job: JsonDict) -> tuple[str, str]:
    return (job["condition"], job["qa"]["id"])


def response_record(job: JsonDict, result: JsonDict, plan: JsonDict) -> JsonDict:
    error = result.get("error")
    usage = result.get("usage") or {}
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
    if not error and result.get("provider") != plan["provider_order"][0]:
        error = (
            "provider mismatch: "
            f"expected={plan['provider_order'][0]!r}, "
            f"got={result.get('provider')!r}"
        )
    if not error and result.get("finish_reason") != "stop":
        error = f"finish_reason is not stop: {result.get('finish_reason')!r}"
    if not error and not result.get("response"):
        error = "empty final response content"
    response = result.get("response", "")
    return {
        "condition": job["condition"],
        "question_id": job["qa"]["id"],
        "sample_id": job["qa"]["sample_id"],
        "category": CATEGORY,
        "truth": job["qa"]["truth"],
        "expected_response": job["expected_response"],
        "response": response,
        "score": score_response(
            job["qa"]["truth"], response, job["condition"]
        ),
        "error": error,
        "model_id": plan["model"],
        "served_model": result.get("served_model"),
        "provider": result.get("provider"),
        "prompt_sha256": job["prompt_sha256"],
        "image_path": job["qa"]["image_path"],
        "image_sha256": job["qa"]["image_sha256"],
        "submitted_image_sha256": result.get("submitted_image_sha256"),
        "scorer_version": plan["scorer"]["version"],
        "scorer_sha256": plan["scorer"]["sha256"],
        "latency_ms": result.get("latency_ms"),
        "usage": usage,
        "finish_reason": result.get("finish_reason"),
        "native_reasoning": result.get("native_reasoning") or {},
        "request_id": result.get("request_id"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def admissible_record(record: JsonDict, job: JsonDict, plan: JsonDict) -> bool:
    if record.get("error") is not None:
        return False
    usage = record.get("usage") or {}
    expected_score = score_response(
        job["qa"]["truth"], record.get("response", ""), job["condition"]
    )
    return bool(
        record.get("condition") == job["condition"]
        and record.get("question_id") == job["qa"]["id"]
        and record.get("sample_id") == job["qa"]["sample_id"]
        and record.get("truth") == job["qa"]["truth"]
        and record.get("expected_response") == job["expected_response"]
        and record.get("model_id") == plan["model"]
        and record.get("served_model") == plan["model"]
        and record.get("provider") == plan["provider_order"][0]
        and record.get("prompt_sha256") == job["prompt_sha256"]
        and record.get("image_path") == job["qa"]["image_path"]
        and record.get("image_sha256") == job["qa"]["image_sha256"]
        and record.get("submitted_image_sha256")
        == job["qa"]["image_sha256"]
        and record.get("scorer_version") == plan["scorer"]["version"]
        and record.get("scorer_sha256") == plan["scorer"]["sha256"]
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
    jobs: Sequence[JsonDict],
    plan: JsonDict,
) -> dict[tuple[str, str], JsonDict]:
    jobs_by_key = {job_key(job): job for job in jobs}
    accepted: dict[tuple[str, str], JsonDict] = {}
    for record in records:
        key = (record.get("condition"), record.get("question_id"))
        job = jobs_by_key.get(key)
        if job is not None and admissible_record(record, job, plan):
            normalized = dict(record)
            normalized["score"] = score_response(
                job["qa"]["truth"],
                record.get("response", ""),
                job["condition"],
            )
            accepted.setdefault(key, normalized)
    return accepted


def percentile(values: Sequence[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(probability * len(ordered)) - 1]


def metric_summary(records: Sequence[JsonDict]) -> JsonDict:
    numbered = [row for row in records if row["truth"]["number"] is not None]
    latencies = [
        int(row["latency_ms"])
        for row in records
        if row.get("latency_ms") is not None
    ]
    confusion = Counter(
        (
            row["truth"]["resource"],
            (row["score"]["parsed"] or {}).get("resource", "INVALID"),
        )
        for row in records
    )
    return {
        "requests": len(records),
        "pair_exact": sum(row["score"]["pair_correct"] for row in records),
        "pair_accuracy": (
            sum(row["score"]["pair_correct"] for row in records) / len(records)
            if records
            else None
        ),
        "resource_exact": sum(
            row["score"]["resource_correct"] for row in records
        ),
        "resource_accuracy": (
            sum(row["score"]["resource_correct"] for row in records)
            / len(records)
            if records
            else None
        ),
        "number_exact": sum(
            row["score"]["number_correct"] is True for row in numbered
        ),
        "number_requests": len(numbered),
        "number_accuracy": (
            sum(row["score"]["number_correct"] is True for row in numbered)
            / len(numbered)
            if numbered
            else None
        ),
        "protocol_valid": sum(
            row["score"]["protocol_valid"] for row in records
        ),
        "protocol_exact": sum(
            row["score"]["protocol_exact"] for row in records
        ),
        "wrapper_rescued": sum(
            row["score"]["wrapper_rescued_content_correct"] for row in records
        ),
        "prompt_tokens": sum(
            int((row.get("usage") or {}).get("prompt_tokens", 0) or 0)
            for row in records
        ),
        "completion_tokens": sum(
            int((row.get("usage") or {}).get("completion_tokens", 0) or 0)
            for row in records
        ),
        "reasoning_tokens": sum(
            usage_reasoning_tokens(row.get("usage") or {}) for row in records
        ),
        "cost": sum(
            float((row.get("usage") or {}).get("cost", 0) or 0)
            for row in records
        ),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "resource_confusion": {
            expected: dict(sorted(predicted.items()))
            for expected, predicted in _nested_confusion(confusion).items()
        },
    }


def _nested_confusion(
    confusion: Counter[tuple[str, str]],
) -> dict[str, Counter[str]]:
    nested: dict[str, Counter[str]] = defaultdict(Counter)
    for (expected, predicted), count in confusion.items():
        nested[expected][predicted] += count
    return dict(sorted(nested.items()))


def paired_effect(
    by_key: dict[tuple[str, str], JsonDict],
    first: str,
    second: str,
    metric: str,
) -> JsonDict:
    question_ids = sorted(
        question_id
        for condition, question_id in by_key
        if condition == first and (second, question_id) in by_key
    )
    values = []
    for question_id in question_ids:
        first_value = by_key[(first, question_id)]["score"][metric]
        second_value = by_key[(second, question_id)]["score"][metric]
        if first_value is None or second_value is None:
            continue
        values.append((bool(first_value), bool(second_value)))
    first_only = sum(a and not b for a, b in values)
    second_only = sum(b and not a for a, b in values)
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "pairs": len(values),
        "first_exact": sum(a for a, _ in values),
        "second_exact": sum(b for _, b in values),
        "first_only": first_only,
        "second_only": second_only,
        "effect_percentage_points": (
            100 * (sum(a for a, _ in values) - sum(b for _, b in values))
            / len(values)
            if values
            else None
        ),
    }


def factorial_summary(by_key: dict[tuple[str, str], JsonDict]) -> JsonDict:
    question_sets = {
        condition: {
            question_id
            for name, question_id in by_key
            if name == condition
        }
        for condition in CONDITIONS
    }
    if (
        len(by_key) != 408
        or any(len(question_sets[condition]) != 102 for condition in CONDITIONS)
        or len({frozenset(ids) for ids in question_sets.values()}) != 1
    ):
        return {"complete": False}
    condition_rates = {}
    for condition in CONDITIONS:
        rows = [
            row for (name, _), row in by_key.items() if name == condition
        ]
        condition_rates[condition] = (
            sum(row["score"]["pair_correct"] for row in rows) / len(rows)
            if rows
            else None
        )
    if any(condition_rates[name] is None for name in CONDITIONS):
        return {"complete": False}
    angle_rate = statistics.mean(
        condition_rates[name] for name in CONDITIONS if name in ANGLE_CONDITIONS
    )
    plain_rate = statistics.mean(
        condition_rates[name] for name in CONDITIONS if name not in ANGLE_CONDITIONS
    )
    described_rate = statistics.mean(
        condition_rates[name]
        for name in CONDITIONS
        if name in DESCRIBED_CONDITIONS
    )
    labels_rate = statistics.mean(
        condition_rates[name]
        for name in CONDITIONS
        if name not in DESCRIBED_CONDITIONS
    )
    angle_description_gain = (
        condition_rates["angle_described"] - condition_rates["angle_labels"]
    )
    plain_description_gain = (
        condition_rates["plain_described"] - condition_rates["plain_labels"]
    )
    return {
        "complete": True,
        "condition_pair_accuracy": condition_rates,
        "angle_minus_plain_percentage_points": 100 * (angle_rate - plain_rate),
        "described_minus_labels_percentage_points": 100
        * (described_rate - labels_rate),
        "interaction_percentage_points": 100
        * (angle_description_gain - plain_description_gain),
    }


def summarize(
    accepted: dict[tuple[str, str], JsonDict],
    *,
    jobs: Sequence[JsonDict],
    plan: JsonDict,
) -> JsonDict:
    by_condition = {}
    for condition in plan["conditions"]:
        rows = [
            row
            for (name, _), row in accepted.items()
            if name == condition
        ]
        per_resource = {
            resource: metric_summary(
                [row for row in rows if row["truth"]["resource"] == resource]
            )
            for resource in RESOURCE_CLASSES
        }
        by_condition[condition] = {
            **metric_summary(rows),
            "per_resource": per_resource,
        }
    comparisons = {}
    if set(plan["conditions"]) == set(CONDITIONS):
        for metric in ("pair_correct", "resource_correct", "number_correct"):
            for first, second in (
                ("angle_labels", "plain_labels"),
                ("angle_described", "plain_described"),
                ("angle_described", "angle_labels"),
                ("plain_described", "plain_labels"),
            ):
                key = f"{metric}:{first}_vs_{second}"
                comparisons[key] = paired_effect(
                    accepted, first, second, metric
                )
    return {
        "complete": len(accepted) == len(jobs),
        "records": len(accepted),
        "planned_requests": len(jobs),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": metric_summary(list(accepted.values())),
        "conditions": by_condition,
        "paired_comparisons": comparisons,
        "factorial_pair_effects": (
            factorial_summary(accepted)
            if set(plan["conditions"]) == set(CONDITIONS)
            else {"complete": False}
        ),
        "plan": plan,
    }


def main() -> None:
    args = parse_args()
    conditions = split_csv(args.conditions)
    provider_order = split_csv(args.provider_order)
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.max_requests is not None and args.max_requests <= 0:
        raise SystemExit("--max-requests must be positive")
    validate_request_contract(args.model, provider_order)
    questions = load_questions(args.qa_path, args.image_root)
    jobs = build_jobs(
        questions,
        conditions=conditions,
        seed=args.seed,
        preflight=args.preflight,
        max_requests=args.max_requests,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, lock_fd = acquire_run_lock(args.output_dir)
    try:
        run(
            args,
            questions=questions,
            conditions=conditions,
            provider_order=provider_order,
            jobs=jobs,
        )
    finally:
        release_run_lock(lock_path, lock_fd)


def run(
    args: argparse.Namespace,
    *,
    questions: Sequence[JsonDict],
    conditions: Sequence[str],
    provider_order: Sequence[str],
    jobs: Sequence[JsonDict],
) -> None:
    plan = build_plan(
        args,
        questions=questions,
        conditions=conditions,
        provider_order=provider_order,
        jobs=jobs,
    )
    plan_path = args.output_dir / "plan.json"
    validate_or_write_plan(plan_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        for condition in conditions:
            print(f"\n--- {condition} ---\n{build_prompt(condition)}")
        return

    response_path = args.output_dir / "responses.jsonl"
    prior = read_jsonl(response_path) if response_path.exists() else []
    accepted = accepted_records(prior, jobs=jobs, plan=plan)
    pending = [job for job in jobs if job_key(job) not in accepted]
    print(
        f"Running {len(pending)} pending of {len(jobs)} requests "
        f"with concurrency={args.concurrency}"
    )
    if pending:
        prepared = []
        for job in pending:
            image_bytes = Path(job["qa"]["image_path"]).read_bytes()
            actual_sha256 = sha256_bytes(image_bytes)
            if actual_sha256 != job["qa"]["image_sha256"]:
                raise SystemExit(
                    "Image changed after plan construction: "
                    f"{job['qa']['image_path']}"
                )
            prepared.append((job, image_bytes))
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set")
        with response_path.open("a") as output:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as pool:
                futures = {
                    pool.submit(
                        call_openrouter_image,
                        api_key,
                        args.model,
                        image_bytes=image_bytes,
                        prompt=job["prompt"],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        provider_order=provider_order,
                    ): job
                    for job, image_bytes in prepared
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
                    status = (
                        "ERROR"
                        if record["error"]
                        else "OK"
                        if record["score"]["pair_correct"]
                        else "MISS"
                    )
                    print(
                        f"[{completed}/{len(pending)}] {status} "
                        f"{job['condition']} {job['qa']['sample_id']} "
                        f"got={record['response']!r}"
                    )

    all_records = read_jsonl(response_path) if response_path.exists() else []
    accepted = accepted_records(all_records, jobs=jobs, plan=plan)
    summary = summarize(accepted, jobs=jobs, plan=plan)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("\nSummary")
    print(json.dumps(summary["conditions"], indent=2, sort_keys=True))
    if not summary["complete"]:
        label = "Preflight" if args.preflight else "Evaluation"
        raise SystemExit(f"{label} is incomplete; rerun to resume pending jobs")
    if args.preflight and (
        len(jobs) != 4
        or {job["condition"] for job in jobs} != set(CONDITIONS)
    ):
        raise SystemExit("Preflight plan must contain one job per condition")
    if args.preflight and any(
        row["score"]["protocol_valid"] is not True for row in accepted.values()
    ):
        raise SystemExit("Preflight produced a protocol-invalid response")


if __name__ == "__main__":
    main()
