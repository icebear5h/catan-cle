"""Identity, ordering, and schema checks for density curricula."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition import density_curriculum as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_int, as_str


def validate_density_curriculum(
    semantic_export_dir: str | Path,
    *,
    curriculum_dir: str | Path | None = None,
) -> JsonDict:
    semantic_root = Path(semantic_export_dir).resolve()
    dataset_root = semantic_root.parent
    api.validate_replay_v1_ms_swift_semantic(dataset_root, export_dir=semantic_root)
    output = (
        Path(curriculum_dir).resolve()
        if curriculum_dir is not None
        else (semantic_root / api.DEFAULT_CURRICULUM_DIR).resolve()
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != api.CURRICULUM_SCHEMA:
        raise api.DensityCurriculumError("density curriculum schema mismatch")
    identity_payload = {key: value for key, value in manifest.items() if key != "identity"}
    if manifest.get("identity") != api.canonical_sha256(identity_payload):
        raise api.DensityCurriculumError("density curriculum identity changed")
    schema_root = api.PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "schemas"
    api._validate_schema_rows(
        schema_root / "density_curriculum_v1.schema.json",
        [manifest],
        label="curriculum.manifest",
    )
    semantic_metadata_path = semantic_root / "metadata.json"
    if manifest["source_semantic_metadata_sha256"] != api.file_sha256(semantic_metadata_path):
        raise api.DensityCurriculumError("source semantic metadata changed")

    files = manifest["files"]
    for key, hash_key in (
        ("annotations", "annotations_sha256"),
        ("audit", "audit_sha256"),
        ("index", "index_sha256"),
    ):
        if api.file_sha256(output / files[key]) != files[hash_key]:
            raise api.DensityCurriculumError(f"curriculum {key} hash changed")
    annotations = api.read_jsonl(output / files["annotations"])
    audits = api.read_jsonl(output / files["audit"])
    index_rows = api.read_jsonl(output / files["index"])
    if not (len(annotations) == len(audits) == len(index_rows) == 8192):
        raise api.DensityCurriculumError("curriculum does not contain exactly 8,192 aligned rows")

    api._validate_schema_rows(
        schema_root / "density_curriculum_index_v1.schema.json",
        index_rows,
        label="curriculum.index",
    )

    semantic_metadata = json.loads(semantic_metadata_path.read_text())
    uniform_annotation_path = semantic_root / semantic_metadata["files"]["train"]["annotations"]
    uniform_audit_path = semantic_root / semantic_metadata["files"]["train"]["audit"]
    if manifest["source_uniform_annotations_sha256"] != api.file_sha256(uniform_annotation_path):
        raise api.DensityCurriculumError("source uniform annotations changed")
    if manifest["source_uniform_audit_sha256"] != api.file_sha256(uniform_audit_path):
        raise api.DensityCurriculumError("source uniform audit changed")
    uniform_annotations = api.read_jsonl(uniform_annotation_path)
    uniform_audits = api.read_jsonl(uniform_audit_path)
    expected_records = api.ordered_curriculum_records(uniform_annotations, uniform_audits)
    expected_annotations = [record["annotation"] for record in expected_records]
    expected_audits = [record["audit"] for record in expected_records]
    expected_index = [
        api._index_row(record, curriculum_index=index)
        for index, record in enumerate(expected_records)
    ]
    if annotations != expected_annotations or audits != expected_audits or index_rows != expected_index:
        raise api.DensityCurriculumError("curriculum order is not deterministically reproducible")

    uniform_ids = [row["query_id"] for row in uniform_audits]
    curriculum_ids = [row["query_id"] for row in audits]
    if len(set(curriculum_ids)) != 8192 or set(curriculum_ids) != set(uniform_ids):
        raise api.DensityCurriculumError("curriculum query IDs do not cover uniform train rows once")
    if Counter(api.canonical_sha256(row) for row in annotations) != Counter(
        api.canonical_sha256(row) for row in uniform_annotations
    ):
        raise api.DensityCurriculumError("curriculum annotations differ from uniform train rows")

    stage_counts: dict[str, int] = Counter()
    previous_stage = -1
    for index, (annotation, audit, index_row) in enumerate(
        zip(annotations, audits, index_rows, strict=True)
    ):
        if index_row["curriculum_index"] != index:
            raise api.DensityCurriculumError("curriculum indexes are not contiguous")
        if index_row["query_id"] != audit["query_id"]:
            raise api.DensityCurriculumError("curriculum index/audit query IDs disagree")
        if index_row["annotation_sha256"] != api.canonical_sha256(annotation):
            raise api.DensityCurriculumError("curriculum annotation identity changed")
        if index_row["audit_sha256"] != api.canonical_sha256(audit):
            raise api.DensityCurriculumError("curriculum audit identity changed")
        if as_int(index_row["stage_index"]) < previous_stage:
            raise api.DensityCurriculumError("curriculum stages are not monotonic")
        previous_stage = as_int(index_row["stage_index"])
        definition = api._stage_for_audit(audit)
        if index_row["stage"] != definition["stage"]:
            raise api.DensityCurriculumError("curriculum stage disagrees with piece count")
        stage_counts[as_str(index_row["stage"])] += 1

    if sum(stage_counts.values()) != manifest["total_rows"]:
        raise api.DensityCurriculumError("curriculum stage counts are stale")
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
