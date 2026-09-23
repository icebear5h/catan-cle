"""Job construction, prompt building, and the immutable run plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    strict_scorer_digest,
)
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.constants import (
    SYSTEM_PROMPT,
    model_supports_reasoning_control,
)
from scripts.board_bench.shapes import (
    JsonDict,
    obj,
    read_json,
    text,
    write_json,
)

__all__ = ["build_jobs", "build_plan", "build_prompt", "validate_or_write_plan"]

IMMUTABLE_PLAN_KEYS = (
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


def build_jobs(
    dataset_dir: Path,
    *,
    questions: Sequence[JsonDict],
    variants: Sequence[str],
    max_requests: int | None,
) -> list[JsonDict]:
    jobs: list[JsonDict] = []
    for question_index, qa in enumerate(questions):
        rotation = question_index % len(variants)
        rotated = [*variants[rotation:], *variants[:rotation]]
        for variant in rotated:
            representation_path = (
                dataset_dir
                / "representations"
                / text(qa["sample_id"], "sample_id")
                / f"{variant}.txt"
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


def _manifest_payload(
    args: argparse.Namespace,
    questions: Sequence[JsonDict],
    variants: Sequence[str],
    jobs: Sequence[JsonDict],
) -> JsonDict:
    return {
        "dataset_metadata": read_json(args.dataset_dir / "metadata.json"),
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
        "fact_digests": list(
            sorted(text(row["fact_digest"], "fact_digest") for row in questions)
        ),
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "variants": list(variants),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "job_prompt_sha256": {
            f"{job['variant']}:{obj(job['qa'], 'job qa')['id']}": hashlib.sha256(
                text(job["prompt"], "job prompt").encode()
            ).hexdigest()
            for job in jobs
        },
    }


def build_plan(
    args: argparse.Namespace,
    *,
    questions: Sequence[JsonDict],
    variants: Sequence[str],
    jobs: Sequence[JsonDict],
    provider_order: Sequence[str],
) -> JsonDict:
    manifest_payload = _manifest_payload(args, questions, variants, jobs)
    reasoning_disabled = model_supports_reasoning_control(args.model)
    return {
        "schema": "catan_ascii_variation_eval/v1",
        "suite": "ascii_variation_probe",
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "variants": list(variants),
        "categories": list(
            sorted({text(row["category"], "category") for row in questions})
        ),
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
        previous = obj(read_json(path), str(path))
        mismatched = [
            key for key in IMMUTABLE_PLAN_KEYS if previous.get(key) != plan.get(key)
        ]
        if mismatched:
            raise SystemExit("Existing output plan is incompatible on: " + ", ".join(mismatched))
        return
    write_json(path, plan)
