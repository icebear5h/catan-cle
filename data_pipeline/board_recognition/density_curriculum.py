"""Deterministic early-to-late ordering for semantic board-recognition SFT."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

from jsonschema import Draft202012Validator

from data_pipeline.board_recognition.replay_dataset import PROJECT_ROOT, file_sha256, read_jsonl
from data_pipeline.board_recognition.replay_ms_swift import (
    EXPORT_SCHEMA as SEMANTIC_EXPORT_SCHEMA,
    canonical_sha256,
    validate_replay_v1_ms_swift_semantic,
)


JsonDict = dict[str, Any]
CURRICULUM_SCHEMA = "catan_board_recognition_density_curriculum/v1"
CURRICULUM_INDEX_SCHEMA = "catan_board_recognition_density_curriculum_index/v1"
DEFAULT_CURRICULUM_DIR = "curriculum"
STAGE_DEFINITIONS = (
    {
        "stage": "early",
        "stage_index": 0,
        "density_bins": ("empty", "setup"),
        "minimum_piece_count": 0,
        "maximum_piece_count": 16,
    },
    {
        "stage": "mid",
        "stage_index": 1,
        "density_bins": ("sparse",),
        "minimum_piece_count": 17,
        "maximum_piece_count": 31,
    },
    {
        "stage": "late",
        "stage_index": 2,
        "density_bins": ("dense",),
        "minimum_piece_count": 32,
        "maximum_piece_count": 126,
    },
)
STAGE_BY_DENSITY = {
    density: definition
    for definition in STAGE_DEFINITIONS
    for density in definition["density_bins"]
}


class DensityCurriculumError(RuntimeError):
    """Raised when a density curriculum loses rows or stage identity."""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _stable_rank(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def _stage_for_audit(audit: JsonDict) -> JsonDict:
    density = audit["density_bin"]
    definition = STAGE_BY_DENSITY.get(density)
    if definition is None:
        raise DensityCurriculumError(f"unknown density bin: {density}")
    piece_count = int(audit["piece_count"])
    if not definition["minimum_piece_count"] <= piece_count <= definition["maximum_piece_count"]:
        raise DensityCurriculumError(
            f"query {audit['query_id']} has {piece_count} pieces outside {definition['stage']}"
        )
    return definition


def _interleave_stage(records: Sequence[JsonDict], *, stage: str) -> list[JsonDict]:
    buckets: dict[tuple[str, str], deque[JsonDict]] = defaultdict(deque)
    ordered_records = sorted(
        records,
        key=lambda record: (
            _stable_rank(stage, record["audit"]["query_id"]),
            record["audit"]["query_id"],
        ),
    )
    for record in ordered_records:
        audit = record["audit"]
        buckets[(audit["head"], audit["class_name"])].append(record)
    bucket_order = sorted(
        buckets,
        key=lambda key: (_stable_rank(stage, *key), key),
    )
    interleaved = []
    while bucket_order:
        next_order = []
        for key in bucket_order:
            bucket = buckets[key]
            interleaved.append(bucket.popleft())
            if bucket:
                next_order.append(key)
        bucket_order = next_order
    return interleaved


def ordered_curriculum_records(
    annotations: Sequence[JsonDict],
    audits: Sequence[JsonDict],
) -> list[JsonDict]:
    if len(annotations) != len(audits):
        raise DensityCurriculumError("uniform semantic annotations and audits differ in length")
    by_stage: dict[str, list[JsonDict]] = defaultdict(list)
    query_ids = set()
    for original_index, (annotation, audit) in enumerate(
        zip(annotations, audits, strict=True)
    ):
        query_id = audit["query_id"]
        if query_id in query_ids:
            raise DensityCurriculumError(f"duplicate semantic query ID: {query_id}")
        query_ids.add(query_id)
        definition = _stage_for_audit(audit)
        by_stage[definition["stage"]].append(
            {
                "annotation": annotation,
                "audit": audit,
                "original_index": original_index,
            }
        )

    ordered = []
    for definition in STAGE_DEFINITIONS:
        stage = definition["stage"]
        stage_records = _interleave_stage(by_stage[stage], stage=stage)
        if not stage_records:
            raise DensityCurriculumError(f"curriculum stage {stage} is empty")
        ordered.extend(stage_records)
    return ordered


def _index_row(record: JsonDict, *, curriculum_index: int) -> JsonDict:
    audit = record["audit"]
    definition = _stage_for_audit(audit)
    return {
        "schema": CURRICULUM_INDEX_SCHEMA,
        "curriculum_index": curriculum_index,
        "original_index": record["original_index"],
        "query_id": audit["query_id"],
        "state_id": audit["state_id"],
        "stage": definition["stage"],
        "stage_index": definition["stage_index"],
        "density_bin": audit["density_bin"],
        "piece_count": audit["piece_count"],
        "head": audit["head"],
        "class_name": audit["class_name"],
        "annotation_sha256": canonical_sha256(record["annotation"]),
        "audit_sha256": canonical_sha256(audit),
    }


def _validate_schema_rows(schema_path: Path, rows: Sequence[JsonDict], *, label: str) -> None:
    validator = Draft202012Validator(json.loads(schema_path.read_text()))
    for index, row in enumerate(rows):
        errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        if errors:
            error = errors[0]
            path = ".".join(str(part) for part in error.path) or "<root>"
            raise DensityCurriculumError(
                f"{label}[{index}] failed {schema_path.name} at {path}: {error.message}"
            )


def build_density_curriculum(
    semantic_export_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> JsonDict:
    semantic_root = Path(semantic_export_dir).resolve()
    dataset_root = semantic_root.parent
    validate_replay_v1_ms_swift_semantic(dataset_root, export_dir=semantic_root)
    semantic_metadata_path = semantic_root / "metadata.json"
    semantic_metadata = json.loads(semantic_metadata_path.read_text())
    if semantic_metadata.get("schema") != SEMANTIC_EXPORT_SCHEMA:
        raise DensityCurriculumError("density curriculum requires semantic export v1")
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (semantic_root / DEFAULT_CURRICULUM_DIR).resolve()
    )
    if output == semantic_root:
        raise DensityCurriculumError("curriculum output cannot replace semantic export root")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"curriculum output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_annotation_path = semantic_root / semantic_metadata["files"]["train"]["annotations"]
    source_audit_path = semantic_root / semantic_metadata["files"]["train"]["audit"]
    annotations = read_jsonl(source_annotation_path)
    audits = read_jsonl(source_audit_path)
    records = ordered_curriculum_records(annotations, audits)
    curriculum_annotations = [record["annotation"] for record in records]
    curriculum_audits = [record["audit"] for record in records]
    index_rows = [
        _index_row(record, curriculum_index=index) for index, record in enumerate(records)
    ]

    annotation_path = output / "train_three_stage.jsonl"
    audit_path = output / "audit_three_stage.jsonl"
    index_path = output / "index_three_stage.jsonl"
    write_jsonl(annotation_path, curriculum_annotations)
    write_jsonl(audit_path, curriculum_audits)
    write_jsonl(index_path, index_rows)

    stages = []
    offset = 0
    for definition in STAGE_DEFINITIONS:
        stage_rows = [row for row in index_rows if row["stage"] == definition["stage"]]
        row_count = len(stage_rows)
        stage_audits = curriculum_audits[offset : offset + row_count]
        stages.append(
            {
                **definition,
                "density_bins": list(definition["density_bins"]),
                "start_index": offset,
                "end_index_exclusive": offset + row_count,
                "row_count": row_count,
                "state_count": len({row["state_id"] for row in stage_audits}),
                "head_counts": dict(sorted(Counter(row["head"] for row in stage_audits).items())),
                "class_counts": dict(
                    sorted(
                        Counter(
                            f"{row['head']}.{row['class_name']}" for row in stage_audits
                        ).items()
                    )
                ),
            }
        )
        offset += row_count

    manifest = {
        "schema": CURRICULUM_SCHEMA,
        "strategy": "sequential_three_stage_without_replacement",
        "source_semantic_export": str(semantic_root),
        "source_semantic_metadata_sha256": file_sha256(semantic_metadata_path),
        "source_uniform_annotations": str(source_annotation_path.relative_to(semantic_root)),
        "source_uniform_annotations_sha256": file_sha256(source_annotation_path),
        "source_uniform_audit": str(source_audit_path.relative_to(semantic_root)),
        "source_uniform_audit_sha256": file_sha256(source_audit_path),
        "total_rows": len(records),
        "unique_query_ids": len({row["query_id"] for row in index_rows}),
        "stages": stages,
        "files": {
            "annotations": annotation_path.name,
            "annotations_sha256": file_sha256(annotation_path),
            "audit": audit_path.name,
            "audit_sha256": file_sha256(audit_path),
            "index": index_path.name,
            "index_sha256": file_sha256(index_path),
        },
    }
    identity_payload = {key: value for key, value in manifest.items() if key != "identity"}
    manifest["identity"] = canonical_sha256(identity_payload)
    write_json(output / "manifest.json", manifest)
    report = validate_density_curriculum(semantic_root, curriculum_dir=output)
    report["output_dir"] = str(output)
    return report


def validate_density_curriculum(
    semantic_export_dir: str | Path,
    *,
    curriculum_dir: str | Path | None = None,
) -> JsonDict:
    semantic_root = Path(semantic_export_dir).resolve()
    dataset_root = semantic_root.parent
    validate_replay_v1_ms_swift_semantic(dataset_root, export_dir=semantic_root)
    output = (
        Path(curriculum_dir).resolve()
        if curriculum_dir is not None
        else (semantic_root / DEFAULT_CURRICULUM_DIR).resolve()
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CURRICULUM_SCHEMA:
        raise DensityCurriculumError("density curriculum schema mismatch")
    identity_payload = {key: value for key, value in manifest.items() if key != "identity"}
    if manifest.get("identity") != canonical_sha256(identity_payload):
        raise DensityCurriculumError("density curriculum identity changed")
    schema_root = PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "schemas"
    _validate_schema_rows(
        schema_root / "density_curriculum_v1.schema.json",
        [manifest],
        label="curriculum.manifest",
    )
    semantic_metadata_path = semantic_root / "metadata.json"
    if manifest["source_semantic_metadata_sha256"] != file_sha256(semantic_metadata_path):
        raise DensityCurriculumError("source semantic metadata changed")

    files = manifest["files"]
    for key, hash_key in (
        ("annotations", "annotations_sha256"),
        ("audit", "audit_sha256"),
        ("index", "index_sha256"),
    ):
        if file_sha256(output / files[key]) != files[hash_key]:
            raise DensityCurriculumError(f"curriculum {key} hash changed")
    annotations = read_jsonl(output / files["annotations"])
    audits = read_jsonl(output / files["audit"])
    index_rows = read_jsonl(output / files["index"])
    if not (len(annotations) == len(audits) == len(index_rows) == 8192):
        raise DensityCurriculumError("curriculum does not contain exactly 8,192 aligned rows")

    _validate_schema_rows(
        schema_root / "density_curriculum_index_v1.schema.json",
        index_rows,
        label="curriculum.index",
    )

    semantic_metadata = json.loads(semantic_metadata_path.read_text())
    uniform_annotation_path = semantic_root / semantic_metadata["files"]["train"]["annotations"]
    uniform_audit_path = semantic_root / semantic_metadata["files"]["train"]["audit"]
    if manifest["source_uniform_annotations_sha256"] != file_sha256(uniform_annotation_path):
        raise DensityCurriculumError("source uniform annotations changed")
    if manifest["source_uniform_audit_sha256"] != file_sha256(uniform_audit_path):
        raise DensityCurriculumError("source uniform audit changed")
    uniform_annotations = read_jsonl(uniform_annotation_path)
    uniform_audits = read_jsonl(uniform_audit_path)
    expected_records = ordered_curriculum_records(uniform_annotations, uniform_audits)
    expected_annotations = [record["annotation"] for record in expected_records]
    expected_audits = [record["audit"] for record in expected_records]
    expected_index = [
        _index_row(record, curriculum_index=index)
        for index, record in enumerate(expected_records)
    ]
    if annotations != expected_annotations or audits != expected_audits or index_rows != expected_index:
        raise DensityCurriculumError("curriculum order is not deterministically reproducible")

    uniform_ids = [row["query_id"] for row in uniform_audits]
    curriculum_ids = [row["query_id"] for row in audits]
    if len(set(curriculum_ids)) != 8192 or set(curriculum_ids) != set(uniform_ids):
        raise DensityCurriculumError("curriculum query IDs do not cover uniform train rows once")
    if Counter(canonical_sha256(row) for row in annotations) != Counter(
        canonical_sha256(row) for row in uniform_annotations
    ):
        raise DensityCurriculumError("curriculum annotations differ from uniform train rows")

    stage_counts: dict[str, int] = Counter()
    previous_stage = -1
    for index, (annotation, audit, index_row) in enumerate(
        zip(annotations, audits, index_rows, strict=True)
    ):
        if index_row["curriculum_index"] != index:
            raise DensityCurriculumError("curriculum indexes are not contiguous")
        if index_row["query_id"] != audit["query_id"]:
            raise DensityCurriculumError("curriculum index/audit query IDs disagree")
        if index_row["annotation_sha256"] != canonical_sha256(annotation):
            raise DensityCurriculumError("curriculum annotation identity changed")
        if index_row["audit_sha256"] != canonical_sha256(audit):
            raise DensityCurriculumError("curriculum audit identity changed")
        if index_row["stage_index"] < previous_stage:
            raise DensityCurriculumError("curriculum stages are not monotonic")
        previous_stage = index_row["stage_index"]
        definition = _stage_for_audit(audit)
        if index_row["stage"] != definition["stage"]:
            raise DensityCurriculumError("curriculum stage disagrees with piece count")
        stage_counts[index_row["stage"]] += 1

    if sum(stage_counts.values()) != manifest["total_rows"]:
        raise DensityCurriculumError("curriculum stage counts are stale")
    return {
        "valid": True,
        "identity": manifest["identity"],
        "rows": len(annotations),
        "stage_rows": dict(stage_counts),
        "uniform_control": str(
            semantic_root / semantic_metadata["files"]["train"]["annotations"]
        ),
        "curriculum_annotations": str(output / files["annotations"]),
        "curriculum_index": str(output / files["index"]),
    }
