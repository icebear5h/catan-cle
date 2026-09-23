"""The immutable run plan for the tile prompt ablation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    SCHEMA,
    SCORER_VERSION,
    json_digest,
    scorer_sha256,
    sha256_text,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.defaults import (
    TileJob,
    TileQuestion,
    file_sha256,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.prompts import (
    SYSTEM_PROMPT,
    build_prompt,
)
from scripts.board_bench.shapes import JsonDict

__all__ = ["build_plan", "validate_or_write_plan"]


def build_plan(
    args: argparse.Namespace,
    *,
    questions: Sequence[TileQuestion],
    conditions: Sequence[str],
    provider_order: Sequence[str],
    jobs: Sequence[TileJob],
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
        "image_hashes": {row["sample_id"]: row["image_sha256"] for row in questions},
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
