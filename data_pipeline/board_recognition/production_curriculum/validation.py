"""Validate source identity and production curriculum boundaries."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition import production_curriculum as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_str


def validate_production_curriculum(
    dataset_dir: str | Path, *, output_dir: str | Path | None = None
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / api.DEFAULT_OUTPUT_NAME).resolve()
    )
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != api.SCHEMA:
        raise api.ProductionCurriculumError("production curriculum metadata schema mismatch")
    if metadata["source_manifest_sha256"] != api.file_sha256(dataset_root / "manifest.jsonl"):
        raise api.ProductionCurriculumError("source manifest changed")
    supplement_root = dataset_root / api.SUPPLEMENT_OUTPUT_NAME
    if metadata["source_supplement_metadata_sha256"] != api.file_sha256(
        supplement_root / "metadata.json"
    ):
        raise api.ProductionCurriculumError("source supplement changed")
    annotation_path = output / metadata["files"]["annotations"]
    audit_path = output / metadata["files"]["audit"]
    if api.file_sha256(annotation_path) != metadata["files"]["annotations_sha256"]:
        raise api.ProductionCurriculumError("production annotations changed")
    if api.file_sha256(audit_path) != metadata["files"]["audit_sha256"]:
        raise api.ProductionCurriculumError("production audit changed")
    rows = api.read_jsonl(annotation_path)
    audits = api.read_jsonl(audit_path)
    if len(rows) != len(audits) or len(rows) != metadata["rows"]:
        raise api.ProductionCurriculumError("production rows and audits are misaligned")
    stage_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    query_ids = set()
    previous_stage = -1
    for index, (row, audit) in enumerate(zip(rows, audits, strict=True)):
        stage = row.get("curriculum_stage")
        if stage != audit.get("curriculum_stage") or stage not in api.CURRICULUM_STAGES:
            raise api.ProductionCurriculumError(f"invalid stage at row {index}")
        stage_index = api.CURRICULUM_STAGES.index(stage)
        if stage_index < previous_stage:
            raise api.ProductionCurriculumError(f"curriculum order regressed at row {index}")
        previous_stage = stage_index
        if audit.get("curriculum_index") != index:
            raise api.ProductionCurriculumError(f"curriculum index mismatch at row {index}")
        query_id = audit.get("query_id")
        if query_id in query_ids:
            raise api.ProductionCurriculumError(f"duplicate curriculum query ID: {query_id}")
        query_ids.add(query_id)
        images = row.get("images")
        messages = row.get("messages")
        if not isinstance(images, list) or len(images) != 1:
            raise api.ProductionCurriculumError(f"invalid image reference at row {index}")
        if not (dataset_root / "images" / as_str(images[0])).is_file():
            raise api.ProductionCurriculumError(f"missing image at row {index}: {images[0]}")
        if not isinstance(messages, list) or len(messages) != 2:
            raise api.ProductionCurriculumError(f"invalid messages at row {index}")
        stage_counts[stage] += 1
        source_counts[as_str(audit["source_family"])] += 1
    if tuple(stage_counts) != api.CURRICULUM_STAGES:
        raise api.ProductionCurriculumError(
            "production curriculum does not contain four ordered stages"
        )
    if metadata.get("per_device_train_batch_size") != api.PRODUCTION_MICROBATCH_SIZE:
        raise api.ProductionCurriculumError("production microbatch size changed")
    if any(count % api.PRODUCTION_MICROBATCH_SIZE for count in stage_counts.values()):
        raise api.ProductionCurriculumError("a stage boundary crosses a microbatch")
    expected_sources = {
        "bidirectional_base": 12_288,
        "spatial_robber_supplement": 4_104,
        "spatial_replay_padding": 88,
    }
    if dict(source_counts) != expected_sources:
        raise api.ProductionCurriculumError(
            f"production source counts changed: {dict(source_counts)} != {expected_sources}"
        )
    if (
        dict(stage_counts) != metadata["stage_counts"]
        or dict(source_counts) != metadata["source_counts"]
    ):
        raise api.ProductionCurriculumError("production metadata counts changed")
    return {
        "valid": True,
        "output_dir": str(output),
        "rows": len(rows),
        "optimizer_steps": len(rows) // api.PRODUCTION_MICROBATCH_SIZE,
        "stage_counts": dict(stage_counts),
        "source_counts": dict(source_counts),
        "annotations_sha256": api.file_sha256(annotation_path),
    }
