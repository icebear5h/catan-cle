"""Vision job construction and the immutable run plan."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    strict_scorer_digest,
)
from evals.catan_board_bench.presentation import load_raw_board_image_presentation
from scripts.board_bench.builders.render_catan_strict_vision_probe import json_digest
from scripts.board_bench.run.eval_catan_strict_vision_probe.constants import (
    EVAL_SCHEMA,
    SUITE_NAME,
    SYSTEM_PROMPT,
    VisionJob,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.prompts import build_prompt
from scripts.board_bench.shapes import (
    JsonDict,
    obj,
    read_json,
    read_json_object,
    text,
    write_json,
)

__all__ = ["build_jobs", "build_plan", "validate_or_write_plan"]

IMMUTABLE_PLAN_KEYS = (
    "schema",
    "suite",
    "model",
    "question_ids",
    "request_count",
    "manifest_sha256",
    "scorer",
    "request_settings",
)


def build_jobs(
    dataset_dir: Path,
    *,
    manifest: Sequence[JsonDict],
    questions: Sequence[JsonDict],
) -> list[VisionJob]:
    manifest_by_sample = {text(row["sample_id"], "sample_id"): row for row in manifest}
    contract_cache: dict[str, JsonDict] = {}
    presentation_cache = {}
    jobs: list[VisionJob] = []
    for qa in questions:
        sample_id = text(qa["sample_id"], "sample_id")
        row = manifest_by_sample[sample_id]
        contract_path = dataset_dir / text(row["contract_path"], "contract_path")
        if sample_id not in contract_cache:
            contract_cache[sample_id] = read_json_object(contract_path)
        contract = contract_cache[sample_id]
        if sample_id not in presentation_cache:
            presentation_cache[sample_id] = load_raw_board_image_presentation(
                dataset_dir / text(row["image_path"], "image_path"),
                source_id=sample_id,
                board_sha256=text(row["source_fact_digest"], "source_fact_digest"),
                expected_sha256=text(row["image_sha256"], "image_sha256"),
                canonical_id_map_sha256=text(
                    row["canonical_id_map_sha256"], "canonical_id_map_sha256"
                ),
            )
        presentation = presentation_cache[sample_id]
        prompt = build_prompt(contract, qa)
        if text(qa["answer_text"], "answer_text") in prompt:
            raise ValueError(f"expected answer leaked into prompt for {qa['id']}")
        jobs.append(
            VisionJob(
                qa=qa,
                image_path=str(dataset_dir / text(row["image_path"], "image_path")),
                image_sha256=text(row["image_sha256"], "image_sha256"),
                contract_path=str(contract_path),
                contract_sha256=text(row["contract_sha256"], "contract_sha256"),
                board_presentation=presentation,
                prompt=prompt,
            )
        )
    if len(jobs) != len({text(job["qa"]["id"], "question id") for job in jobs}):
        raise ValueError("vision job question IDs are not unique")
    return jobs


def build_plan(
    args: argparse.Namespace,
    *,
    metadata: JsonDict,
    questions: Sequence[JsonDict],
    jobs: Sequence[VisionJob],
) -> JsonDict:
    manifest_payload: JsonDict = {
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
            text(job["qa"]["id"], "question id"): {
                "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
                "image_sha256": job["image_sha256"],
                "contract_sha256": job["contract_sha256"],
                "answer_sha256": json_digest(job["qa"]["answer"]),
            }
            for job in jobs
        },
    }
    provider = getattr(args, "provider", "novita")
    return {
        "schema": EVAL_SCHEMA,
        "suite": SUITE_NAME,
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
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
            "provider": provider,
            "provider_routing": (
                "novita_direct" if provider == "novita" else "openrouter_default_unpinned"
            ),
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
