"""Fail-closed dataset metadata, integrity, and request-contract checks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    full_fact_digest,
    strict_scorer_digest,
)
from evals.catan_board_bench.ascii_variations.facts import full_facts_from_json
from evals.catan_board_bench.paths import resolve_benchmark_reference
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats import dataset_config
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.constants import (
    file_sha256,
    json_digest,
    model_supports_reasoning_control,
    read_jsonl,
)
from scripts.board_bench.shapes import JsonDict, obj, read_json_object, text

__all__ = [
    "validate_dataset_integrity",
    "validate_dataset_metadata",
    "validate_request_contract",
]


def validate_dataset_metadata(metadata: JsonDict) -> None:
    format_names = dataset_config.FORMAT_NAMES
    expected: JsonDict = {
        "schema": dataset_config.DATASET_SCHEMA,
        "formats": list(format_names),
        "board_count": 12,
        "question_count": 60,
        "request_count": 60 * len(format_names),
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "incident_list_format": False,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"Dataset metadata mismatch: {mismatches}")


def _check_source_lock(dataset_dir: Path, metadata: JsonDict) -> None:
    source_lock = metadata.get("source_lock")
    if not isinstance(source_lock, dict):
        raise SystemExit("Dataset source lock is missing")
    if json_digest(source_lock) != metadata.get("source_lock_sha256"):
        raise SystemExit("Dataset source-lock digest mismatch")

    source_dir = resolve_benchmark_reference(text(metadata["source_dataset"], "source_dataset"))
    for relative_path, expected_sha256 in source_lock.items():
        source_path = source_dir / relative_path
        if not source_path.is_file():
            raise SystemExit(f"Locked source file is missing: {source_path}")
        if file_sha256(source_path) != expected_sha256:
            raise SystemExit(f"Locked source file changed: {source_path}")

        generated_path: Path | None = None
        if relative_path == "qa.jsonl":
            generated_path = dataset_dir / "qa.jsonl"
        elif relative_path.startswith("facts/"):
            generated_path = dataset_dir / relative_path
        elif relative_path.startswith("aliases/"):
            generated_path = dataset_dir / relative_path
        elif relative_path.endswith("/tile_rows.txt"):
            generated_path = dataset_dir / relative_path
        if generated_path is not None:
            if not generated_path.is_file():
                raise SystemExit(f"Generated locked file is missing: {generated_path}")
            if file_sha256(generated_path) != expected_sha256:
                raise SystemExit(f"Generated locked file changed: {generated_path}")


def _check_representations(
    dataset_dir: Path,
    manifest_by_sample: dict[str, JsonDict],
    metadata_hashes: JsonDict,
) -> dict[str, str]:
    format_names = dataset_config.FORMAT_NAMES
    extensions = dataset_config.FORMAT_EXTENSIONS
    parse_format = dataset_config.parse_full_graph_format
    expected_representation_paths = set()
    fact_digests: dict[str, str] = {}
    for sample_id, manifest_row in sorted(manifest_by_sample.items()):
        fact_path = dataset_dir / "facts" / f"{sample_id}.json"
        facts = full_facts_from_json(read_json_object(fact_path))
        digest = full_fact_digest(facts)
        fact_digests[sample_id] = digest
        if digest != manifest_row.get("fact_digest"):
            raise SystemExit(f"Fact/manifest digest mismatch for {sample_id}")
        if set(obj(metadata_hashes.get(sample_id, {}), "hashes")) != set(format_names):
            raise SystemExit(f"Incomplete representation hashes for {sample_id}")

        metrics = obj(manifest_row.get("representation_metrics") or {}, "metrics")
        sample_hashes = obj(metadata_hashes[sample_id], "sample hashes")
        for format_name in format_names:
            extension = extensions[format_name]
            representation_path = (
                dataset_dir / "representations" / sample_id / f"{format_name}{extension}"
            )
            expected_representation_paths.add(representation_path.resolve())
            if not representation_path.is_file():
                raise SystemExit(f"Representation is missing: {representation_path}")
            actual_sha256 = file_sha256(representation_path)
            if actual_sha256 != sample_hashes[format_name]:
                raise SystemExit(f"Representation hash mismatch: {representation_path}")
            format_metrics = obj(metrics.get(format_name) or {}, "format metrics")
            if actual_sha256 != format_metrics.get("sha256"):
                raise SystemExit(f"Representation/manifest hash mismatch: {representation_path}")
            content = representation_path.read_text().rstrip("\n")
            try:
                parsed = parse_format(format_name, content)
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Representation failed to parse: {representation_path}: {exc}"
                ) from exc
            if full_fact_digest(parsed) != digest:
                raise SystemExit(f"Representation fact mismatch: {representation_path}")

    actual_representation_paths = {
        path.resolve() for path in (dataset_dir / "representations").glob("*/*") if path.is_file()
    }
    if actual_representation_paths != expected_representation_paths:
        raise SystemExit("Unexpected or missing representation files")
    if set(metadata_hashes) != set(manifest_by_sample):
        raise SystemExit("Representation hash sample set mismatch")
    return fact_digests


def validate_dataset_integrity(
    dataset_dir: Path,
    metadata: JsonDict,
) -> list[JsonDict]:
    _check_source_lock(dataset_dir, metadata)

    questions = read_jsonl(dataset_dir / "qa.jsonl")
    manifest = read_jsonl(dataset_dir / "manifest.jsonl")
    manifest_by_sample = {text(row["sample_id"], "sample_id"): row for row in manifest}
    if len(manifest) != 12 or len(manifest_by_sample) != 12:
        raise SystemExit("Generated manifest must contain 12 unique boards")
    if len(questions) != 60 or len({text(row["id"], "question id") for row in questions}) != 60:
        raise SystemExit("Generated QA must contain 60 unique questions")

    metadata_hashes = metadata.get("representation_hashes")
    if not isinstance(metadata_hashes, dict):
        raise SystemExit("Representation hash manifest is missing")
    fact_digests = _check_representations(dataset_dir, manifest_by_sample, metadata_hashes)

    for question in questions:
        raw_sample = question.get("sample_id")
        sample_id = raw_sample if isinstance(raw_sample, str) else None
        if sample_id is None or sample_id not in fact_digests:
            raise SystemExit(f"Question references unknown board: {raw_sample}")
        if question.get("fact_digest") != fact_digests[sample_id]:
            raise SystemExit(f"Question fact digest mismatch: {question.get('id')}")
    return questions


def validate_request_contract(
    model_id: str,
    provider_order: Sequence[str],
) -> None:
    if len(provider_order) != 1:
        raise SystemExit("Exactly one pinned provider is required")
    if not model_supports_reasoning_control(model_id):
        raise SystemExit(
            "This suite requires a model with an explicit hard reasoning-disable contract"
        )
