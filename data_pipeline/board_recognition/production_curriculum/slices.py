"""Validated, order-preserving slices of production curricula."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Sequence

from data_pipeline.board_recognition import production_curriculum as api
from data_pipeline.board_recognition.replay_dataset import JsonDict


def parse_stage_list(spec: str) -> tuple[str, ...]:
    """Parse ``clean_board_grounding,pieces_and_colors`` into curriculum stages in order."""

    stages = tuple(part.strip() for part in spec.split(",") if part.strip())
    unknown = [stage for stage in stages if stage not in api.CURRICULUM_STAGES]
    if not stages or unknown or len(set(stages)) != len(stages):
        raise api.ProductionCurriculumError(
            f"stage slice must name distinct stages from {api.CURRICULUM_STAGES}: {spec!r}"
        )
    return tuple(stage for stage in api.CURRICULUM_STAGES if stage in stages)


def slice_production_curriculum(
    dataset_dir: str | Path,
    *,
    stages: Sequence[str],
    output_dir: str | Path,
    source_output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> JsonDict:
    """Write the rows of ``stages`` from a validated curriculum as a standalone stage file.

    The slice keeps the source order, re-indexes ``curriculum_index`` from
    zero, and reuses the shared replay image directory, so a stage can be
    trained on its own through ``--initial-bundle`` from the previous stage's
    final bundle without rebuilding or re-rendering anything.
    """

    dataset_root = Path(dataset_dir).resolve()
    source = (
        Path(source_output_dir).resolve()
        if source_output_dir is not None
        else (dataset_root / api.DEFAULT_OUTPUT_NAME).resolve()
    )
    wanted = api.parse_stage_list(",".join(stages))
    api.validate_production_curriculum(dataset_root, output_dir=source)
    output = Path(output_dir).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise api.ProductionCurriculumError("refusing to overwrite output outside the dataset root")
        shutil.rmtree(output)
    source_metadata = json.loads((source / "metadata.json").read_text())
    rows = api.read_jsonl(source / source_metadata["files"]["annotations"])
    audits = api.read_jsonl(source / source_metadata["files"]["audit"])
    kept_rows: list[JsonDict] = []
    kept_audits: list[JsonDict] = []
    for row, audit in zip(rows, audits, strict=True):
        if row["curriculum_stage"] not in wanted:
            continue
        kept_rows.append(row)
        kept_audits.append({**audit, "curriculum_index": len(kept_audits), "source_curriculum_index": audit["curriculum_index"]})
    if not kept_rows:
        raise api.ProductionCurriculumError(f"no rows for stages {wanted}")
    if len(kept_rows) % api.PRODUCTION_MICROBATCH_SIZE:
        raise api.ProductionCurriculumError("stage slice is not microbatch aligned")
    annotation_path = output / "train.jsonl"
    audit_path = output / "audit" / "train.jsonl"
    api._write_jsonl(annotation_path, kept_rows)
    api._write_jsonl(audit_path, kept_audits)
    stage_counts = Counter(audit["curriculum_stage"] for audit in kept_audits)
    density_counts = Counter(audit.get("density_bin", "unknown") for audit in kept_audits)
    metadata = {
        "schema": api.SLICE_SCHEMA,
        "stages": list(wanted),
        "source_curriculum": str(source),
        "source_curriculum_annotations_sha256": source_metadata["files"]["annotations_sha256"],
        "source_manifest_sha256": source_metadata["source_manifest_sha256"],
        "image_root": str(dataset_root / "images"),
        "per_device_train_batch_size": api.PRODUCTION_MICROBATCH_SIZE,
        "rows": len(kept_rows),
        "optimizer_steps_per_epoch": len(kept_rows) // api.PRODUCTION_MICROBATCH_SIZE,
        "stage_counts": dict(stage_counts),
        "density_counts": dict(density_counts),
        "files": {
            "annotations": str(annotation_path.relative_to(output)),
            "annotations_sha256": api.file_sha256(annotation_path),
            "audit": str(audit_path.relative_to(output)),
            "audit_sha256": api.file_sha256(audit_path),
        },
    }
    api._write_json(output / "metadata.json", metadata)
    return metadata
