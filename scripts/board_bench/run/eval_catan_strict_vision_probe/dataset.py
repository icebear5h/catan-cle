"""Fail-closed dataset validation and question selection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    strict_scorer_digest,
)
from evals.catan_board_bench.paths import resolve_benchmark_reference
from scripts.board_bench.builders.render_catan_strict_vision_probe import (
    file_sha256,
    json_digest,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.constants import DATASET_SCHEMA
from scripts.board_bench.shapes import (
    JsonDict,
    obj,
    read_json_object,
    read_jsonl,
    text,
)

__all__ = ["select_questions", "validate_dataset"]


def _check_metadata(metadata: JsonDict) -> None:
    expected: JsonDict = {
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
    identity_projection = obj(
        metadata.get("identity_projection") or {}, "identity_projection"
    )
    if identity_projection.get("target") != "canonical engine tile/node/edge/port IDs":
        raise ValueError("dataset does not use canonical engine IDs")
    if identity_projection.get("image_contains_entity_labels") is not False:
        raise ValueError("dataset image annotation policy changed")
    render_variant = obj(metadata.get("render_variant") or {}, "render_variant")
    if render_variant.get("image_annotation") is not None:
        raise ValueError("raw-image dataset unexpectedly declares an annotation")


def _check_source_lock(metadata: JsonDict) -> None:
    source_dir = resolve_benchmark_reference(text(metadata["source_dataset"], "source_dataset"))
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
            path = resolve_benchmark_reference(reference)
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise ValueError(f"locked source changed: {path}")


def _check_boards(dataset_dir: Path, manifest: list[JsonDict]) -> None:
    for row in manifest:
        image_path = dataset_dir / text(row["image_path"], "image_path")
        contract_path = dataset_dir / text(row["contract_path"], "contract_path")
        if file_sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"image hash mismatch: {image_path}")
        if file_sha256(contract_path) != row["contract_sha256"]:
            raise ValueError(f"contract hash mismatch: {contract_path}")
        contract = read_json_object(contract_path)
        if json_digest(contract) != row["engine_state_sha256"]:
            raise ValueError(f"engine-state digest mismatch: {contract_path}")


def validate_dataset(
    dataset_dir: Path,
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = read_json_object(metadata_path)
    _check_metadata(metadata)
    _check_source_lock(metadata)

    manifest = read_jsonl(dataset_dir / "manifest.jsonl")
    questions = read_jsonl(dataset_dir / "qa.jsonl")
    if json_digest(list(manifest)) != metadata["visual_manifest_sha256"]:
        raise ValueError("visual manifest digest mismatch")
    if json_digest(list(questions)) != metadata["question_payload_sha256"]:
        raise ValueError("question payload digest mismatch")
    manifest_by_sample = {text(row["sample_id"], "sample_id"): row for row in manifest}
    if len(manifest) != 12 or len(manifest_by_sample) != 12:
        raise ValueError("visual manifest must contain 12 unique boards")
    if len(questions) != 60 or len({text(row["id"], "question id") for row in questions}) != 60:
        raise ValueError("visual QA must contain 60 unique questions")

    _check_boards(dataset_dir, manifest)

    for question in questions:
        sample = manifest_by_sample.get(text(question["sample_id"], "sample_id"))
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
        unknown = category_set - {text(row["category"], "category") for row in questions}
        if unknown:
            raise ValueError(f"unknown categories: {sorted(unknown)}")
        selected = [row for row in selected if row["category"] in category_set]
    if max_questions is not None:
        selected = selected[:max_questions]
    return selected
