"""The immutable run plan for the strict text probe."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    strict_scorer_digest,
)
from scripts.board_bench.builders.render_catan_strict_vision_probe import json_digest, write_json
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats import FormatJob
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.constants import (
    EVAL_SCHEMA,
    FORMAT_NAME,
    SUITE_NAME,
    SYSTEM_PROMPT,
)
from scripts.board_bench.shapes import JsonDict, obj, read_json, text

__all__ = ["build_plan", "validate_or_write_plan"]

IMMUTABLE_PLAN_KEYS = (
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


def build_plan(
    args: argparse.Namespace,
    *,
    metadata: JsonDict,
    questions: Sequence[JsonDict],
    jobs: Sequence[FormatJob],
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
                    "fact_digest": row["fact_digest"],
                }
                for row in questions
            ]
        ),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "jobs": {
            text(job["qa"]["id"], "question id"): {
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
