"""Re-derive every exported row and audit record from the source dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition.dataset import (
    read_jsonl,
    resolve_project_path,
    sample_state_queries,
)
from data_pipeline.board_recognition.replay_sft import validate_replay_v1_qwen_sft
from data_pipeline.board_recognition.sft._config import SFT_EXPORT_SCHEMA, SPLITS, JsonDict
from data_pipeline.board_recognition.sft._io import file_sha256
from data_pipeline.board_recognition.sft._rows import qwen_query_prompt, validate_qwen_row
from data_pipeline.json_coerce import as_dict, as_list, as_str


def validate_qwen_sft_export(
    dataset_dir: str | Path,
    *,
    export_dir: str | Path | None = None,
) -> JsonDict:
    dataset_root = Path(dataset_dir)
    output = Path(export_dir) if export_dir is not None else dataset_root / "qwen_sft"
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") == "catan_board_recognition_qwen_sft/v2":
        return validate_replay_v1_qwen_sft(dataset_root, export_dir=output)
    if metadata.get("schema") != SFT_EXPORT_SCHEMA:
        raise ValueError("Qwen SFT export schema mismatch")
    manifest_path = dataset_root / "manifest.jsonl"
    if metadata.get("source_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Qwen SFT source manifest changed")
    manifest = read_jsonl(manifest_path)
    states = {row["sample_id"]: row for row in manifest}
    spec_path = resolve_project_path(metadata["curriculum_path"])
    if metadata.get("curriculum_sha256") != file_sha256(spec_path):
        raise ValueError("Qwen SFT curriculum changed")
    spec = json.loads(spec_path.read_text())
    base_states_by_group = {
        row["counterfactual_group_id"]: row
        for row in manifest
        if row["counterfactual_role"] == "base"
    }
    total_rows = 0
    entity_counts: Counter[str] = Counter()
    attribute_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    split_counts = {}

    for split in SPLITS:
        annotations_path = output / metadata["files"][split]["annotations"]
        audit_path = output / metadata["files"][split]["audit"]
        if file_sha256(annotations_path) != metadata["files"][split]["annotations_sha256"]:
            raise ValueError(f"Qwen SFT annotation hash mismatch: {split}")
        if file_sha256(audit_path) != metadata["files"][split]["audit_sha256"]:
            raise ValueError(f"Qwen SFT audit hash mismatch: {split}")
        rows = read_jsonl(annotations_path)
        audit_rows = read_jsonl(audit_path)
        if len(rows) != len(audit_rows):
            raise ValueError(f"Qwen SFT annotations/audit length mismatch: {split}")
        expected_queries: dict[str, JsonDict] = {}
        seen_query_ids: set[str] = set()
        for state in (row for row in manifest if row["split"] == split):
            labels = as_dict(json.loads((dataset_root / as_str(state["label_path"])).read_text()))
            base_state = base_states_by_group[state["counterfactual_group_id"]]
            sampling_labels = as_dict(
                json.loads((dataset_root / as_str(base_state["label_path"])).read_text())
            )
            for query in sample_state_queries(
                labels,
                spec=spec,
                queries_per_state=int(metadata["queries_per_state"]),
                seed=int(metadata["seed"]),
                epoch=0,
                forced_query=as_dict(as_dict(state["counterfactual"])["target"]),
                sampling_key=as_str(state["counterfactual_group_id"]),
                sampling_dense_labels=sampling_labels,
            ):
                expected_queries[as_str(query["query_id"])] = query
        for row, audit in zip(rows, audit_rows, strict=True):
            validate_qwen_row(row)
            if audit["query_id"] in seen_query_ids:
                raise ValueError(f"Qwen SFT duplicated query: {audit['query_id']}")
            seen_query_ids.add(as_str(audit["query_id"]))
            audit_state = states.get(audit["state_id"])
            audit_query = expected_queries.get(as_str(audit["query_id"]))
            if audit_state is None or audit_query is None or audit_state["split"] != split:
                raise ValueError(f"Qwen SFT audit references unknown state/query: {audit}")
            if any(
                (
                    audit["split"] != split,
                    audit["stage"] != audit_state["stage"],
                    audit["counterfactual_group_id"] != audit_state["counterfactual_group_id"],
                    audit["counterfactual_role"] != audit_state["counterfactual_role"],
                    audit["label_path"] != audit_state["label_path"],
                )
            ):
                raise ValueError(f"Qwen SFT state provenance mismatch: {audit['query_id']}")
            if (
                row["image"] != audit_state["image_path"]
                or not (dataset_root / as_str(row["image"])).is_file()
            ):
                raise ValueError(f"Qwen SFT image path mismatch: {audit['query_id']}")
            conversations = [as_dict(turn) for turn in as_list(row["conversations"])]
            if conversations[1]["value"] != audit_query["class_name"]:
                raise ValueError(f"Qwen SFT target mismatch: {audit['query_id']}")
            prompt = qwen_query_prompt(audit_query)
            if conversations[0]["value"] != f"<image>\n{prompt}":
                raise ValueError(f"Qwen SFT prompt mismatch: {audit['query_id']}")
            if audit["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
                raise ValueError(f"Qwen SFT prompt hash mismatch: {audit['query_id']}")
            expected_audit = {
                "entity_type": audit_query["entity_type"],
                "attribute": audit_query["attribute"],
                "slot": audit_query["slot"],
                "class_name": audit_query["class_name"],
                "class_index": audit_query["class_index"],
            }
            if any(audit[key] != value for key, value in expected_audit.items()):
                raise ValueError(f"Qwen SFT audit label mismatch: {audit['query_id']}")
            entity_counts[as_str(audit_query["entity_type"])] += 1
            attribute_counts[as_str(audit_query["head"])] += 1
            class_counts[f"{audit_query['head']}.{audit_query['class_name']}"] += 1
        if seen_query_ids != set(expected_queries):
            raise ValueError(f"Qwen SFT split omitted or duplicated queries: {split}")
        split_counts[split] = len(rows)
        total_rows += len(rows)

    if split_counts != metadata["split_counts"] or total_rows != metadata["row_count"]:
        raise ValueError("Qwen SFT metadata row counts are stale")
    if dict(sorted(entity_counts.items())) != metadata["entity_type_counts"]:
        raise ValueError("Qwen SFT entity counts are stale")
    if dict(sorted(attribute_counts.items())) != metadata["attribute_counts"]:
        raise ValueError("Qwen SFT attribute counts are stale")
    if dict(sorted(class_counts.items())) != metadata["class_counts"]:
        raise ValueError("Qwen SFT class counts are stale")
    if len(set(entity_counts.values())) != 1:
        raise ValueError(f"Qwen SFT entity types are not balanced: {dict(entity_counts)}")
    return {
        "valid": True,
        "states": len(manifest),
        "rows": total_rows,
        "splits": {split: count for split, count in split_counts.items()},
        "entity_types": dict(sorted(entity_counts.items())),
    }


__all__ = ["validate_qwen_sft_export"]
