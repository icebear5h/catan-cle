"""Revalidate an exported supplement against its recorded fingerprints."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    file_sha256,
    read_jsonl,
)
from data_pipeline.board_recognition.robber_impl._config import (
    ATLAS_TOKENS,
    CURRICULUM_STAGES,
    DEFAULT_OUTPUT_NAME,
    EXPORT_SCHEMA,
    ROBBER_ROWS_PER_STATE,
    SMOKE_ROWS_PER_STAGE,
    SPATIAL_ROWS_PER_EMPTY_STATE,
    SpatialRobberError,
)
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict


def validate_spatial_robber_supplement(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != EXPORT_SCHEMA:
        raise SpatialRobberError("supplement metadata schema mismatch")
    manifest_path = dataset_root / "manifest.jsonl"
    if metadata["source_manifest_sha256"] != file_sha256(manifest_path):
        raise SpatialRobberError("source manifest changed")
    states = read_jsonl(manifest_path)
    observed_total = 0
    train_binary: Counter[str] = Counter()
    train_robber_types: Counter[str] = Counter()
    train_relationships: Counter[str] = Counter()
    train_spatial_tokens: set[str] = set()
    smoke_metadata = metadata.get("curriculum_smoke", {})
    smoke_path = output / str(smoke_metadata.get("annotations", ""))
    if not smoke_path.is_file() or file_sha256(smoke_path) != smoke_metadata.get(
        "annotations_sha256"
    ):
        raise SpatialRobberError("curriculum smoke file is missing or changed")
    smoke_rows = read_jsonl(smoke_path)
    smoke_stage_counts = Counter(row.get("curriculum_stage") for row in smoke_rows)
    if list(smoke_stage_counts) != list(CURRICULUM_STAGES) or set(smoke_stage_counts.values()) != {
        SMOKE_ROWS_PER_STAGE
    }:
        raise SpatialRobberError("curriculum smoke does not contain eight ordered rows per stage")

    for split in ALL_SPLITS:
        file_metadata = metadata["files"][split]
        annotation_path = output / file_metadata["annotations"]
        audit_path = output / file_metadata["audit"]
        if file_sha256(annotation_path) != file_metadata["annotations_sha256"]:
            raise SpatialRobberError(f"{split} annotation hash changed")
        if file_sha256(audit_path) != file_metadata["audit_sha256"]:
            raise SpatialRobberError(f"{split} audit hash changed")
        rows = read_jsonl(annotation_path)
        audits = read_jsonl(audit_path)
        if len(rows) != len(audits) or len(rows) != metadata["split_counts"][split]:
            raise SpatialRobberError(f"{split} rows and audits are misaligned")
        split_states = [state for state in states if state["split"] == split]
        empty_states = sum(state["density_bin"] == "empty" for state in split_states)
        expected = (
            len(split_states) * ROBBER_ROWS_PER_STATE + empty_states * SPATIAL_ROWS_PER_EMPTY_STATE
        )
        if len(rows) != expected:
            raise SpatialRobberError(f"{split} has {len(rows)} rows; expected {expected}")
        previous_stage = -1
        query_ids = set()
        for row, audit in zip(rows, audits, strict=True):
            stage = row.get("curriculum_stage")
            if stage != audit["curriculum_stage"] or stage not in CURRICULUM_STAGES:
                raise SpatialRobberError(f"{split} has invalid curriculum stage")
            stage_index = CURRICULUM_STAGES.index(stage)
            if stage_index < previous_stage:
                raise SpatialRobberError(f"{split} curriculum order regressed")
            previous_stage = stage_index
            if audit["query_id"] in query_ids:
                raise SpatialRobberError(f"{split} has duplicate query IDs")
            query_ids.add(audit["query_id"])
            messages = row.get("messages")
            images = row.get("images")
            if not isinstance(messages, list) or len(messages) != 2:
                raise SpatialRobberError(f"{split} row has invalid messages")
            if not isinstance(images, list) or len(images) != 1:
                raise SpatialRobberError(f"{split} row has invalid images")
            if not (dataset_root / "images" / as_str(images[0])).is_file():
                raise SpatialRobberError(f"{split} row image is missing")
            prompt = as_dict(messages[0]).get("content")
            answer = as_str(as_dict(messages[1])["content"])
            if prompt != f"<image>\n{audit['prompt']}" or answer != audit["answer"]:
                raise SpatialRobberError(f"{split} training and audit text disagree")
            if answer.startswith("<") and answer not in ATLAS_TOKENS:
                raise SpatialRobberError(f"{split} answer has an unknown atlas token")
            if split == "train" and audit["task_family"] == "spatial_grounding":
                train_relationships[as_str(audit["relationship"])] += 1
                train_spatial_tokens.update(as_str(token) for token in as_list(audit["tokens"]))
                if answer in {"yes", "no"}:
                    train_binary[answer] += 1
            if split == "train" and audit["task_family"] == "robber":
                train_robber_types[as_str(audit["task_type"])] += 1
        observed_total += len(rows)

    if train_binary["yes"] != train_binary["no"] or not train_binary["yes"]:
        raise SpatialRobberError("train spatial yes/no rows are not balanced")
    train_state_count = sum(state["split"] == "train" for state in states)
    expected_robber_types = {
        "robber_presence_positive": train_state_count,
        "robber_presence_negative": train_state_count,
        "robber_token_return": train_state_count,
    }
    if dict(train_robber_types) != expected_robber_types:
        raise SpatialRobberError("train robber task types are not exhaustive")
    required_relationships = {"above", "below", "left_of", "right_of", "adjacent", "connected"}
    if not required_relationships.issubset(train_relationships):
        raise SpatialRobberError("train spatial rows omit required relationships")
    expected_spatial_tokens = {
        *(f"<N{index:02d}>" for index in range(54)),
        *(f"<T{index:02d}>" for index in range(19)),
    }
    if train_spatial_tokens != expected_spatial_tokens:
        raise SpatialRobberError("train spatial rows do not cover every node and tile token")
    return {
        "valid": True,
        "output_dir": str(output),
        "rows": observed_total,
        "train_rows": metadata["split_counts"]["train"],
        "train_task_counts": metadata["task_counts"]["train"],
        "train_stage_counts": metadata["stage_counts"]["train"],
        "train_relationship_counts": metadata["relationship_counts"]["train"],
        "train_spatial_binary_counts": dict(train_binary),
        "train_spatial_token_coverage": {
            "nodes": sum(token.startswith("<N") for token in train_spatial_tokens),
            "tiles": sum(token.startswith("<T") for token in train_spatial_tokens),
        },
        "train_robber_type_counts": dict(train_robber_types),
        "curriculum_smoke": {
            "rows": len(smoke_rows),
            "stage_counts": dict(smoke_stage_counts),
            "optimizer_steps_at_gradient_accumulation_8": len(smoke_rows) // 8,
        },
    }
