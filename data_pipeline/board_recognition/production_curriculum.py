"""Compose the immutable replay corpus and grounding supplement for production SFT."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from data_pipeline.board_recognition.replay_dataset import file_sha256, read_jsonl
from data_pipeline.board_recognition.spatial_robber import (
    CURRICULUM_STAGES,
    DEFAULT_OUTPUT_NAME as SUPPLEMENT_OUTPUT_NAME,
    validate_spatial_robber_supplement,
)


JsonDict = dict[str, Any]
SCHEMA = "catan_production_vision_curriculum/v1"
AUDIT_SCHEMA = "catan_production_vision_curriculum_audit/v1"
SLICE_SCHEMA = "catan_production_vision_curriculum_slice/v1"
DEFAULT_OUTPUT_NAME = "production_curriculum_v1"
PRODUCTION_MICROBATCH_SIZE = 32
BASE_DENSITIES_BY_STAGE = {
    "clean_board_grounding": {"empty", "setup"},
    "pieces_and_colors": {"sparse"},
    "real_game_distribution": {"dense"},
}


class ProductionCurriculumError(RuntimeError):
    """Raised when the composed curriculum loses source or stage identity."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _stable_rank(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def interleave_four_to_one(
    base_records: Sequence[JsonDict], supplement_records: Sequence[JsonDict], *, stage: str
) -> list[JsonDict]:
    """Deterministically spread one supplement row through every four base rows."""

    base = sorted(
        base_records,
        key=lambda record: (_stable_rank(stage, "base", record["query_id"]), record["query_id"]),
    )
    supplement = sorted(
        supplement_records,
        key=lambda record: (
            _stable_rank(stage, "supplement", record["query_id"]),
            record["query_id"],
        ),
    )
    if len(base) != 4 * len(supplement):
        raise ProductionCurriculumError(
            f"{stage} requires an exact 4:1 base/supplement ratio; "
            f"received {len(base)}:{len(supplement)}"
        )
    result = []
    for index, supplement_record in enumerate(supplement):
        result.extend(base[index * 4 : (index + 1) * 4])
        result.append(supplement_record)
    return result


def padding_needed(row_count: int, *, multiple: int = PRODUCTION_MICROBATCH_SIZE) -> int:
    if row_count < 0 or multiple <= 0:
        raise ValueError("row_count must be nonnegative and multiple must be positive")
    return (-row_count) % multiple


def _record(row: JsonDict, audit: JsonDict, *, source_family: str) -> JsonDict:
    return {
        "row": row,
        "audit": audit,
        "query_id": audit["query_id"],
        "source_family": source_family,
    }


def _base_records(dataset_root: Path) -> list[JsonDict]:
    base_root = dataset_root / "ms_swift_bidirectional_v1"
    rows = read_jsonl(base_root / "mixed" / "train.jsonl")
    indices = read_jsonl(base_root / "mixed_index" / "train.jsonl")
    if len(rows) != len(indices):
        raise ProductionCurriculumError("base mixed rows and indices are misaligned")
    density_by_state = {
        state["sample_id"]: state["density_bin"]
        for state in read_jsonl(dataset_root / "manifest.jsonl")
        if state["split"] == "train"
    }
    records = []
    for row, index in zip(rows, indices, strict=True):
        density = density_by_state.get(index["state_id"])
        if density is None:
            raise ProductionCurriculumError(
                f"base state is absent from train manifest: {index['state_id']}"
            )
        audit = {
            "query_id": index["query_id"],
            "state_id": index["state_id"],
            "row_kind": index["row_kind"],
            "density_bin": density,
        }
        records.append(_record(row, audit, source_family="bidirectional_base"))
    return records


def _supplement_records(supplement_root: Path) -> list[JsonDict]:
    rows = read_jsonl(supplement_root / "train.jsonl")
    audits = read_jsonl(supplement_root / "audit" / "train.jsonl")
    if len(rows) != len(audits):
        raise ProductionCurriculumError("supplement rows and audits are misaligned")
    return [
        _record(row, audit, source_family="spatial_robber_supplement")
        for row, audit in zip(rows, audits, strict=True)
    ]


def _training_and_audit_rows(records_by_stage: dict[str, list[JsonDict]]) -> tuple[list, list]:
    spatial_records = records_by_stage[CURRICULUM_STAGES[0]]
    rows: list[JsonDict] = []
    audits: list[JsonDict] = []
    for stage_index, stage in enumerate(CURRICULUM_STAGES):
        stage_records = list(records_by_stage[stage])
        needed = padding_needed(len(stage_records))
        for padding_index in range(needed):
            source = spatial_records[
                _stable_rank(stage, "padding", padding_index) % len(spatial_records)
            ]
            stage_records.append(
                {
                    **source,
                    "source_family": "spatial_replay_padding",
                    "query_id": f"{source['query_id']}#replay-{stage}-{padding_index}",
                    "audit": {
                        **source["audit"],
                        "query_id": f"{source['query_id']}#replay-{stage}-{padding_index}",
                        "replay_of": source["query_id"],
                    },
                }
            )
        if len(stage_records) % PRODUCTION_MICROBATCH_SIZE:
            raise ProductionCurriculumError(f"{stage} is not microbatch aligned")
        for record in stage_records:
            row = dict(record["row"])
            row["curriculum_stage"] = stage
            audit = {
                **record["audit"],
                "schema": AUDIT_SCHEMA,
                "curriculum_index": len(rows),
                "curriculum_stage": stage,
                "curriculum_stage_index": stage_index,
                "source_family": record["source_family"],
                "query_id": record["query_id"],
            }
            rows.append(row)
            audits.append(audit)
    return rows, audits


def build_production_curriculum(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    supplement_root = dataset_root / SUPPLEMENT_OUTPUT_NAME
    validate_spatial_robber_supplement(dataset_root, output_dir=supplement_root)
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise ProductionCurriculumError("refusing to overwrite output outside the dataset root")
        shutil.rmtree(output)

    base_records = _base_records(dataset_root)
    supplement_records = _supplement_records(supplement_root)
    records_by_stage: dict[str, list[JsonDict]] = {}
    spatial_stage = CURRICULUM_STAGES[0]
    records_by_stage[spatial_stage] = [
        record
        for record in supplement_records
        if record["audit"]["curriculum_stage"] == spatial_stage
    ]
    for stage in CURRICULUM_STAGES[1:]:
        base = [
            record
            for record in base_records
            if record["audit"]["density_bin"] in BASE_DENSITIES_BY_STAGE[stage]
        ]
        supplement = [
            record for record in supplement_records if record["audit"]["curriculum_stage"] == stage
        ]
        records_by_stage[stage] = interleave_four_to_one(base, supplement, stage=stage)

    rows, audits = _training_and_audit_rows(records_by_stage)
    annotation_path = output / "train.jsonl"
    audit_path = output / "audit" / "train.jsonl"
    _write_jsonl(annotation_path, rows)
    _write_jsonl(audit_path, audits)
    stage_counts = Counter(audit["curriculum_stage"] for audit in audits)
    source_counts = Counter(audit["source_family"] for audit in audits)
    metadata = {
        "schema": SCHEMA,
        "strategy": "four_stage_sequential_4_to_1_with_microbatch_alignment",
        "per_device_train_batch_size": PRODUCTION_MICROBATCH_SIZE,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "source_supplement_metadata_sha256": file_sha256(supplement_root / "metadata.json"),
        "rows": len(rows),
        "optimizer_steps": len(rows) // PRODUCTION_MICROBATCH_SIZE,
        "stage_counts": dict(stage_counts),
        "source_counts": dict(source_counts),
        "files": {
            "annotations": str(annotation_path.relative_to(output)),
            "annotations_sha256": file_sha256(annotation_path),
            "audit": str(audit_path.relative_to(output)),
            "audit_sha256": file_sha256(audit_path),
        },
    }
    _write_json(output / "metadata.json", metadata)
    return validate_production_curriculum(dataset_root, output_dir=output)


def validate_production_curriculum(
    dataset_dir: str | Path, *, output_dir: str | Path | None = None
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != SCHEMA:
        raise ProductionCurriculumError("production curriculum metadata schema mismatch")
    if metadata["source_manifest_sha256"] != file_sha256(dataset_root / "manifest.jsonl"):
        raise ProductionCurriculumError("source manifest changed")
    supplement_root = dataset_root / SUPPLEMENT_OUTPUT_NAME
    if metadata["source_supplement_metadata_sha256"] != file_sha256(
        supplement_root / "metadata.json"
    ):
        raise ProductionCurriculumError("source supplement changed")
    annotation_path = output / metadata["files"]["annotations"]
    audit_path = output / metadata["files"]["audit"]
    if file_sha256(annotation_path) != metadata["files"]["annotations_sha256"]:
        raise ProductionCurriculumError("production annotations changed")
    if file_sha256(audit_path) != metadata["files"]["audit_sha256"]:
        raise ProductionCurriculumError("production audit changed")
    rows = read_jsonl(annotation_path)
    audits = read_jsonl(audit_path)
    if len(rows) != len(audits) or len(rows) != metadata["rows"]:
        raise ProductionCurriculumError("production rows and audits are misaligned")
    stage_counts = Counter()
    source_counts = Counter()
    query_ids = set()
    previous_stage = -1
    for index, (row, audit) in enumerate(zip(rows, audits, strict=True)):
        stage = row.get("curriculum_stage")
        if stage != audit.get("curriculum_stage") or stage not in CURRICULUM_STAGES:
            raise ProductionCurriculumError(f"invalid stage at row {index}")
        stage_index = CURRICULUM_STAGES.index(stage)
        if stage_index < previous_stage:
            raise ProductionCurriculumError(f"curriculum order regressed at row {index}")
        previous_stage = stage_index
        if audit.get("curriculum_index") != index:
            raise ProductionCurriculumError(f"curriculum index mismatch at row {index}")
        query_id = audit.get("query_id")
        if query_id in query_ids:
            raise ProductionCurriculumError(f"duplicate curriculum query ID: {query_id}")
        query_ids.add(query_id)
        images = row.get("images")
        messages = row.get("messages")
        if not isinstance(images, list) or len(images) != 1:
            raise ProductionCurriculumError(f"invalid image reference at row {index}")
        if not (dataset_root / "images" / images[0]).is_file():
            raise ProductionCurriculumError(f"missing image at row {index}: {images[0]}")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ProductionCurriculumError(f"invalid messages at row {index}")
        stage_counts[stage] += 1
        source_counts[audit["source_family"]] += 1
    if tuple(stage_counts) != CURRICULUM_STAGES:
        raise ProductionCurriculumError(
            "production curriculum does not contain four ordered stages"
        )
    if metadata.get("per_device_train_batch_size") != PRODUCTION_MICROBATCH_SIZE:
        raise ProductionCurriculumError("production microbatch size changed")
    if any(count % PRODUCTION_MICROBATCH_SIZE for count in stage_counts.values()):
        raise ProductionCurriculumError("a stage boundary crosses a microbatch")
    expected_sources = {
        "bidirectional_base": 12_288,
        "spatial_robber_supplement": 4_104,
        "spatial_replay_padding": 88,
    }
    if dict(source_counts) != expected_sources:
        raise ProductionCurriculumError(
            f"production source counts changed: {dict(source_counts)} != {expected_sources}"
        )
    if (
        dict(stage_counts) != metadata["stage_counts"]
        or dict(source_counts) != metadata["source_counts"]
    ):
        raise ProductionCurriculumError("production metadata counts changed")
    return {
        "valid": True,
        "output_dir": str(output),
        "rows": len(rows),
        "optimizer_steps": len(rows) // PRODUCTION_MICROBATCH_SIZE,
        "stage_counts": dict(stage_counts),
        "source_counts": dict(source_counts),
        "annotations_sha256": file_sha256(annotation_path),
    }


def parse_stage_list(spec: str) -> tuple[str, ...]:
    """Parse ``clean_board_grounding,pieces_and_colors`` into curriculum stages in order."""

    stages = tuple(part.strip() for part in spec.split(",") if part.strip())
    unknown = [stage for stage in stages if stage not in CURRICULUM_STAGES]
    if not stages or unknown or len(set(stages)) != len(stages):
        raise ProductionCurriculumError(
            f"stage slice must name distinct stages from {CURRICULUM_STAGES}: {spec!r}"
        )
    return tuple(stage for stage in CURRICULUM_STAGES if stage in stages)


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
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    wanted = parse_stage_list(",".join(stages))
    validate_production_curriculum(dataset_root, output_dir=source)
    output = Path(output_dir).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise ProductionCurriculumError("refusing to overwrite output outside the dataset root")
        shutil.rmtree(output)
    source_metadata = json.loads((source / "metadata.json").read_text())
    rows = read_jsonl(source / source_metadata["files"]["annotations"])
    audits = read_jsonl(source / source_metadata["files"]["audit"])
    kept_rows: list[JsonDict] = []
    kept_audits: list[JsonDict] = []
    for row, audit in zip(rows, audits, strict=True):
        if row["curriculum_stage"] not in wanted:
            continue
        kept_rows.append(row)
        kept_audits.append({**audit, "curriculum_index": len(kept_audits), "source_curriculum_index": audit["curriculum_index"]})
    if not kept_rows:
        raise ProductionCurriculumError(f"no rows for stages {wanted}")
    if len(kept_rows) % PRODUCTION_MICROBATCH_SIZE:
        raise ProductionCurriculumError("stage slice is not microbatch aligned")
    annotation_path = output / "train.jsonl"
    audit_path = output / "audit" / "train.jsonl"
    _write_jsonl(annotation_path, kept_rows)
    _write_jsonl(audit_path, kept_audits)
    stage_counts = Counter(audit["curriculum_stage"] for audit in kept_audits)
    density_counts = Counter(audit.get("density_bin", "unknown") for audit in kept_audits)
    metadata = {
        "schema": SLICE_SCHEMA,
        "stages": list(wanted),
        "source_curriculum": str(source),
        "source_curriculum_annotations_sha256": source_metadata["files"]["annotations_sha256"],
        "source_manifest_sha256": source_metadata["source_manifest_sha256"],
        "image_root": str(dataset_root / "images"),
        "per_device_train_batch_size": PRODUCTION_MICROBATCH_SIZE,
        "rows": len(kept_rows),
        "optimizer_steps_per_epoch": len(kept_rows) // PRODUCTION_MICROBATCH_SIZE,
        "stage_counts": dict(stage_counts),
        "density_counts": dict(density_counts),
        "files": {
            "annotations": str(annotation_path.relative_to(output)),
            "annotations_sha256": file_sha256(annotation_path),
            "audit": str(audit_path.relative_to(output)),
            "audit_sha256": file_sha256(audit_path),
        },
    }
    _write_json(output / "metadata.json", metadata)
    return metadata
