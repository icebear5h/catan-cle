"""Argument and source-dataset validation for the strict vision projection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    full_fact_digest,
    full_public_graph_facts,
    strict_scorer_digest,
)
from evals.catan_board_bench.paths import resolve_benchmark_reference
from evals.catan_board_bench.text_format_optimization import DATASET_SCHEMA
from scripts.board_bench.shapes import (
    JsonDict,
    read_json,
    read_json_object,
    read_jsonl,
    text,
)

__all__ = ["validate_render_args", "validate_source_dataset"]


def validate_render_args(
    source_dir: Path,
    output_dir: Path,
    *,
    image_size: int,
    view_padding_factor: float,
    target_board_canvas_fraction: float,
) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if view_padding_factor < 0:
        raise ValueError("view_padding_factor must be non-negative")
    if not 0 < target_board_canvas_fraction <= 1:
        raise ValueError("target_board_canvas_fraction must be in (0, 1]")
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("source_dir and output_dir must differ")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _check_metadata(metadata: JsonDict) -> None:
    expected_metadata: JsonDict = {
        "schema": DATASET_SCHEMA,
        "board_count": 12,
        "question_count": 60,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "images": False,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"source metadata mismatch: {mismatches}")


def _check_boards(source_dir: Path, manifest: list[JsonDict]) -> None:
    for row in manifest:
        sample_id = text(row["sample_id"], "sample_id")
        contract_path = resolve_benchmark_reference(
            text(row["source_contract"], "source_contract")
        )
        if not contract_path.is_file():
            raise FileNotFoundError(contract_path)
        contract = read_json_object(contract_path)
        regenerated_facts, regenerated_aliases = full_public_graph_facts(
            contract,
            sample_id=sample_id,
        )
        stored_facts = read_json(source_dir / "facts" / f"{sample_id}.json")
        stored_aliases = read_json(source_dir / "aliases" / f"{sample_id}.json")
        if regenerated_facts != stored_facts:
            raise ValueError(f"source facts do not match contract for {sample_id}")
        if regenerated_aliases != stored_aliases:
            raise ValueError(f"source aliases do not match contract for {sample_id}")
        if full_fact_digest(stored_facts) != row["fact_digest"]:
            raise ValueError(f"source fact digest mismatch for {sample_id}")


def validate_source_dataset(
    source_dir: Path,
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    required = ("metadata.json", "manifest.jsonl", "qa.jsonl", "aliases", "facts")
    for name in required:
        if not (source_dir / name).exists():
            raise FileNotFoundError(source_dir / name)

    metadata = read_json_object(source_dir / "metadata.json")
    _check_metadata(metadata)

    manifest = read_jsonl(source_dir / "manifest.jsonl")
    questions = read_jsonl(source_dir / "qa.jsonl")
    manifest_by_sample = {text(row["sample_id"], "sample_id"): row for row in manifest}
    if len(manifest) != 12 or len(manifest_by_sample) != 12:
        raise ValueError("source manifest must contain 12 unique boards")
    if len(questions) != 60 or len({text(row["id"], "question id") for row in questions}) != 60:
        raise ValueError("source QA must contain 60 unique questions")

    _check_boards(source_dir, manifest)

    question_counts = Counter(text(row["category"], "question category") for row in questions)
    if len(question_counts) != 10 or set(question_counts.values()) != {6}:
        raise ValueError(f"expected six questions in ten categories, got {question_counts}")
    for question in questions:
        sample = manifest_by_sample.get(text(question["sample_id"], "sample_id"))
        if sample is None:
            raise ValueError(f"question references unknown board: {question['id']}")
        if question["fact_digest"] != sample["fact_digest"]:
            raise ValueError(f"question fact digest mismatch: {question['id']}")
    return metadata, manifest, questions
