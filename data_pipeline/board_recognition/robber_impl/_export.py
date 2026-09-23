"""Export the spatial and robber supplement."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.robber_impl._config import (
    CURRICULUM_STAGES,
    DEFAULT_OUTPUT_NAME,
    EXPORT_SCHEMA,
    ROBBER_ROWS_PER_STATE,
    SMOKE_ROWS_PER_STAGE,
    SPATIAL_ROWS_PER_EMPTY_STATE,
    SpatialRobberError,
)
from data_pipeline.board_recognition.robber_impl._io import _training_row, _write_json, _write_jsonl
from data_pipeline.board_recognition.robber_impl._queries import (
    _audit_row,
    robber_queries_for_state,
    spatial_queries_for_state,
)
from data_pipeline.board_recognition.robber_impl._smoke import build_curriculum_smoke_rows
from data_pipeline.board_recognition.robber_impl._validate import validate_spatial_robber_supplement
from data_pipeline.json_coerce import as_dict, as_str
from data_pipeline.json_types import JsonDict, JsonValue


def export_spatial_robber_supplement(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    validate_replay_v1_dataset(dataset_root, rerender=False)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise SpatialRobberError("refusing to overwrite output outside the dataset root")
        shutil.rmtree(output)
    (output / "audit").mkdir(parents=True)

    manifest_path = dataset_root / "manifest.jsonl"
    states = read_jsonl(manifest_path)
    files: dict[str, JsonValue] = {}
    split_counts: dict[str, JsonValue] = {}
    split_task_counts: dict[str, JsonValue] = {}
    split_stage_counts: dict[str, JsonValue] = {}
    split_relationship_counts: dict[str, JsonValue] = {}

    for split in ALL_SPLITS:
        split_states = [state for state in states if state["split"] == split]
        pairs_by_stage: dict[str, list[tuple[JsonDict, JsonDict]]] = {
            stage: [] for stage in CURRICULUM_STAGES
        }
        empty_index = 0
        for state_index, state in enumerate(split_states):
            contract_path = dataset_root / as_str(state["contract_path"])
            if file_sha256(contract_path) != as_dict(state["sha256"])["contract"]:
                raise SpatialRobberError(f"contract hash changed: {state['sample_id']}")
            contract = json.loads(contract_path.read_text())
            image_name = Path(as_str(state["image_path"])).name
            queries = robber_queries_for_state(state, contract, state_index=state_index)
            if state["density_bin"] == "empty":
                queries.extend(spatial_queries_for_state(state, state_index=empty_index))
                empty_index += 1
            for query in queries:
                stage_name = as_str(query["curriculum_stage"])
                row = _training_row(
                    as_str(query["prompt"]), as_str(query["answer"]), image_name, stage_name
                )
                audit = _audit_row(state, query, image_name=image_name)
                pairs_by_stage[stage_name].append((row, audit))

        rows = [pair[0] for stage in CURRICULUM_STAGES for pair in pairs_by_stage[stage]]
        audits = [pair[1] for stage in CURRICULUM_STAGES for pair in pairs_by_stage[stage]]
        annotation_path = output / f"{split}.jsonl"
        audit_path = output / "audit" / f"{split}.jsonl"
        _write_jsonl(annotation_path, rows)
        _write_jsonl(audit_path, audits)
        split_counts[split] = len(rows)
        split_task_counts[split] = dict(
            sorted(Counter(as_str(row["task_family"]) for row in audits).items())
        )
        split_stage_counts[split] = dict(
            sorted(Counter(as_str(row["curriculum_stage"]) for row in audits).items())
        )
        split_relationship_counts[split] = dict(
            sorted(Counter(as_str(row["relationship"]) for row in audits).items())
        )
        files[split] = {
            "annotations": annotation_path.name,
            "annotations_sha256": file_sha256(annotation_path),
            "audit": str(audit_path.relative_to(output)),
            "audit_sha256": file_sha256(audit_path),
        }

    train_files = as_dict(files["train"])
    train_rows = read_jsonl(output / as_str(train_files["annotations"]))
    train_audits = read_jsonl(output / as_str(train_files["audit"]))
    smoke_rows = build_curriculum_smoke_rows(dataset_root, train_rows, train_audits)
    smoke_path = output / "curriculum_smoke_32.jsonl"
    _write_jsonl(smoke_path, smoke_rows)

    metadata: JsonDict = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(manifest_path),
        "image_root": str((dataset_root / "images").resolve()),
        "spatial_rows_per_empty_state": SPATIAL_ROWS_PER_EMPTY_STATE,
        "robber_rows_per_state": ROBBER_ROWS_PER_STATE,
        "split_counts": split_counts,
        "task_counts": split_task_counts,
        "stage_counts": split_stage_counts,
        "relationship_counts": split_relationship_counts,
        "curriculum_smoke": {
            "annotations": smoke_path.name,
            "annotations_sha256": file_sha256(smoke_path),
            "rows": len(smoke_rows),
            "rows_per_stage": SMOKE_ROWS_PER_STAGE,
            "optimizer_steps_at_gradient_accumulation_8": len(smoke_rows) // 8,
        },
        "files": files,
    }
    _write_json(output / "metadata.json", metadata)
    return validate_spatial_robber_supplement(dataset_root, output_dir=output)

