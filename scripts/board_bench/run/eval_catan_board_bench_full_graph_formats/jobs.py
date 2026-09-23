"""Format job construction and the immutable run plan."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    strict_scorer_digest,
)
from evals.catan_board_bench.presentation import load_text_board_presentation
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats import dataset_config
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.constants import (
    SYSTEM_PROMPT,
    FormatJob,
    json_digest,
    model_supports_reasoning_control,
    write_json,
)
from scripts.board_bench.shapes import JsonDict, obj, read_json, text

__all__ = ["build_jobs", "build_plan", "build_prompt", "validate_or_write_plan"]

IMMUTABLE_PLAN_KEYS = (
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


def build_jobs(
    dataset_dir: Path,
    *,
    questions: Sequence[JsonDict],
    formats: Sequence[str],
    max_requests: int | None,
) -> list[FormatJob]:
    if len(formats) != len(set(formats)):
        raise ValueError("duplicate formats are not allowed")
    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive")
    extensions = dataset_config.FORMAT_EXTENSIONS
    jobs: list[FormatJob] = []
    for question_index, qa in enumerate(questions):
        rotation = question_index % len(formats)
        rotated = [*formats[rotation:], *formats[:rotation]]
        sample_id = text(qa["sample_id"], "sample_id")
        for format_name in rotated:
            extension = extensions[format_name]
            representation_path = (
                dataset_dir / "representations" / sample_id / f"{format_name}{extension}"
            )
            representation_bytes = representation_path.read_bytes()
            presentation = load_text_board_presentation(
                representation_path,
                source_id=sample_id,
                board_sha256=text(qa["fact_digest"], "fact_digest"),
                aliases_path=(dataset_dir / "aliases" / f"{sample_id}.json"),
                format=format_name,
                renderer_version=("3" if format_name == "indexed_tile_rows" else "1"),
            )
            prompt = build_prompt(format_name, presentation.content, qa)
            if text(qa["answer_text"], "answer_text") in prompt:
                raise ValueError(f"expected answer leaked into prompt for {format_name}/{qa['id']}")
            jobs.append(
                FormatJob(
                    format=format_name,
                    qa=qa,
                    representation_path=str(representation_path),
                    representation_sha256=hashlib.sha256(representation_bytes).hexdigest(),
                    board_presentation=presentation,
                    prompt=prompt,
                )
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
    jobs: Sequence[FormatJob],
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
    manifest_payload: JsonDict = {
        "dataset_metadata_sha256": json_digest(dataset_metadata),
        "source_lock_sha256": dataset_metadata["source_lock_sha256"],
        "question_payload_sha256": json_digest(list(question_payload)),
        "fact_digests": list(
            sorted(text(row["fact_digest"], "fact_digest") for row in questions)
        ),
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "formats": list(formats),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "jobs": {
            f"{job['format']}:{job['qa']['id']}": {
                "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
                "representation_sha256": job["representation_sha256"],
                "answer_sha256": json_digest(job["qa"]["answer"]),
            }
            for job in jobs
        },
    }
    reasoning_disabled = model_supports_reasoning_control(args.model)
    return {
        "schema": dataset_config.EVAL_SCHEMA,
        "suite": dataset_config.SUITE_NAME,
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "formats": list(formats),
        "categories": list(sorted({text(row["category"], "category") for row in questions})),
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
        previous = obj(read_json(path), str(path))
        mismatched = [
            key for key in IMMUTABLE_PLAN_KEYS if previous.get(key) != plan.get(key)
        ]
        if mismatched:
            raise SystemExit("Existing output plan is incompatible on: " + ", ".join(mismatched))
        return
    write_json(path, plan)
