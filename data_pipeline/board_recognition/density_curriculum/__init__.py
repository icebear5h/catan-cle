"""Deterministic early-to-late ordering for semantic board-recognition SFT."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable, Sequence

from jsonschema import Draft202012Validator

from data_pipeline.board_recognition.density_curriculum.records import (
    StageDefinition,
    StageRecord,
)
from data_pipeline.json_coerce import as_int, as_str
from data_pipeline.json_types import JsonDict

from ..replay_dataset import read_jsonl as read_jsonl
from ..replay_ms_swift import EXPORT_SCHEMA as SEMANTIC_EXPORT_SCHEMA
from ..replay_ms_swift import canonical_sha256 as canonical_sha256
from ..replay_ms_swift import (
    validate_replay_v1_ms_swift_semantic as validate_replay_v1_ms_swift_semantic,
)
from ..sources import PROJECT_ROOT as PROJECT_ROOT
from ..sources import file_sha256 as file_sha256
from .validation import validate_density_curriculum as validate_density_curriculum

CURRICULUM_SCHEMA = "catan_board_recognition_density_curriculum/v1"
CURRICULUM_INDEX_SCHEMA = "catan_board_recognition_density_curriculum_index/v1"
DEFAULT_CURRICULUM_DIR = "curriculum"


STAGE_DEFINITIONS: tuple[StageDefinition, ...] = (
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _stable_rank(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def _stage_for_audit(audit: JsonDict) -> StageDefinition:
    density = as_str(audit["density_bin"])
    definition = STAGE_BY_DENSITY.get(density)
    if definition is None:
        raise DensityCurriculumError(f"unknown density bin: {density}")
    piece_count = as_int(audit["piece_count"])
    if not definition["minimum_piece_count"] <= piece_count <= definition["maximum_piece_count"]:
        raise DensityCurriculumError(
            f"query {audit['query_id']} has {piece_count} pieces outside {definition['stage']}"
        )
    return definition


def _interleave_stage(records: Sequence[StageRecord], *, stage: str) -> list[StageRecord]:
    buckets: dict[tuple[str, str], deque[StageRecord]] = defaultdict(deque)
    ordered_records = sorted(
        records,
        key=lambda record: (
            _stable_rank(stage, record["audit"]["query_id"]),
            as_str(record["audit"]["query_id"]),
        ),
    )
    for record in ordered_records:
        audit = record["audit"]
        buckets[(as_str(audit["head"]), as_str(audit["class_name"]))].append(record)
    bucket_order = sorted(
        buckets,
        key=lambda key: (_stable_rank(stage, *key), key),
    )
    interleaved: list[StageRecord] = []
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
) -> list[StageRecord]:
    if len(annotations) != len(audits):
        raise DensityCurriculumError("uniform semantic annotations and audits differ in length")
    by_stage: dict[str, list[StageRecord]] = defaultdict(list)
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

    ordered: list[StageRecord] = []
    for definition in STAGE_DEFINITIONS:
        stage = definition["stage"]
        stage_records = _interleave_stage(by_stage[stage], stage=stage)
        if not stage_records:
            raise DensityCurriculumError(f"curriculum stage {stage} is empty")
        ordered.extend(stage_records)
    return ordered


def _index_row(record: StageRecord, *, curriculum_index: int) -> JsonDict:
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
    curriculum_annotations: list[JsonDict] = [record["annotation"] for record in records]
    curriculum_audits: list[JsonDict] = [record["audit"] for record in records]
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
                "density_bins": [name for name in definition["density_bins"]],
                "start_index": offset,
                "end_index_exclusive": offset + row_count,
                "row_count": row_count,
                "state_count": len({as_str(row["state_id"]) for row in stage_audits}),
                "head_counts": dict(
                    sorted(Counter(as_str(row["head"]) for row in stage_audits).items())
                ),
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
