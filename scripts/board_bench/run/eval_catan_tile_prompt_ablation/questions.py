"""Question loading and the counterbalanced job list."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    CATEGORY,
    CONDITIONS,
    NUMBER_CLASSES,
    RESOURCE_CLASSES,
    expected_response,
    json_digest,
    sha256_text,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.defaults import (
    TileJob,
    TileQuestion,
    file_sha256,
    job_key,
    read_jsonl,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.prompts import (
    build_prompt,
    truth_from_qa,
)
from scripts.board_bench.shapes import text

__all__ = ["build_jobs", "load_questions", "rotated_conditions"]


def load_questions(qa_path: Path, image_root: Path) -> list[TileQuestion]:
    rows = [row for row in read_jsonl(qa_path) if row.get("category") == CATEGORY]
    if len(rows) != 102 or len({row.get("id") for row in rows}) != 102:
        raise SystemExit("Expected 102 unique isolated tile questions")
    if len({row.get("sample_id") for row in rows}) != 102:
        raise SystemExit("Expected 102 unique isolated tile sample IDs")
    if len({row.get("image_path") for row in rows}) != 102:
        raise SystemExit("Expected 102 unique isolated tile image paths")
    selected: list[TileQuestion] = []
    for row in rows:
        truth = truth_from_qa(row)
        image_path = image_root / text(row["image_path"], "image_path")
        if not image_path.is_file():
            raise SystemExit(f"Missing image: {image_path}")
        selected.append(
            TileQuestion(
                id=text(row["id"], "question id"),
                sample_id=text(row["sample_id"], "sample_id"),
                category=text(row["category"], "category"),
                image_path=str(image_path),
                image_sha256=file_sha256(image_path),
                source_row_sha256=json_digest(row),
                truth=truth,
            )
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
    expected_cross_product: dict[tuple[str, int | None], int] = {
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


def _job_for(qa: TileQuestion, condition: str) -> TileJob:
    prompt = build_prompt(condition)
    return TileJob(
        condition=condition,
        qa=qa,
        prompt=prompt,
        prompt_sha256=sha256_text(prompt),
        expected_response=expected_response(qa["truth"], condition),
    )


def build_jobs(
    questions: Sequence[TileQuestion],
    *,
    conditions: Sequence[str],
    seed: int,
    preflight: bool = False,
    max_requests: int | None = None,
) -> list[TileJob]:
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("Conditions must be nonempty and unique")
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if preflight and max_requests is not None:
        raise ValueError("Preflight cannot be combined with max_requests")

    shuffled: list[TileQuestion] = [TileQuestion(**question) for question in questions]
    random.Random(seed).shuffle(shuffled)
    jobs: list[TileJob] = []
    for question_index, qa in enumerate(shuffled):
        for condition in rotated_conditions(conditions, question_index):
            jobs.append(_job_for(qa, condition))

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
        jobs = [_job_for(qa, condition) for condition, qa in selected_pairs]
        if len(jobs) != 4 or {job["condition"] for job in jobs} != set(CONDITIONS):
            raise ValueError("Preflight must contain exactly one job per condition")
    if max_requests is not None:
        jobs = jobs[:max_requests]
    if len({job_key(job) for job in jobs}) != len(jobs):
        raise ValueError("Generated duplicate jobs")
    return jobs
