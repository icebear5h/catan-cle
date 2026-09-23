"""Compose the immutable replay corpus and grounding supplement for production SFT."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from data_pipeline.json_coerce import as_dict
from data_pipeline.json_types import JsonDict

from ..replay_dataset import read_jsonl as read_jsonl
from ..sources import file_sha256 as file_sha256
from ..spatial_robber import CURRICULUM_STAGES as CURRICULUM_STAGES
from ..spatial_robber import DEFAULT_OUTPUT_NAME as _SUPPLEMENT_OUTPUT_NAME
from ..spatial_robber import validate_spatial_robber_supplement
from .slices import parse_stage_list as parse_stage_list
from .slices import slice_production_curriculum as slice_production_curriculum
from .validation import validate_production_curriculum as validate_production_curriculum

SUPPLEMENT_OUTPUT_NAME = _SUPPLEMENT_OUTPUT_NAME
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _stable_rank(*parts: object) -> int:
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


def _training_and_audit_rows(records_by_stage: dict[str, list[JsonDict]]) -> tuple[list[JsonDict], list[JsonDict]]:
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
                        **as_dict(source["audit"]),
                        "query_id": f"{source['query_id']}#replay-{stage}-{padding_index}",
                        "replay_of": source["query_id"],
                    },
                }
            )
        if len(stage_records) % PRODUCTION_MICROBATCH_SIZE:
            raise ProductionCurriculumError(f"{stage} is not microbatch aligned")
        for record in stage_records:
            row = dict(as_dict(record["row"]))
            row["curriculum_stage"] = stage
            audit = {
                **as_dict(record["audit"]),
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
        if as_dict(record["audit"])["curriculum_stage"] == spatial_stage
    ]
    for stage in CURRICULUM_STAGES[1:]:
        base = [
            record
            for record in base_records
            if as_dict(record["audit"])["density_bin"] in BASE_DENSITIES_BY_STAGE[stage]
        ]
        supplement = [
            record
            for record in supplement_records
            if as_dict(record["audit"])["curriculum_stage"] == stage
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
