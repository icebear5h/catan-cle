"""Official Qwen conversation projection for dense board-recognition states."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.board_recognition.replay_dataset import DATASET_SCHEMA
from data_pipeline.board_recognition.replay_sft import (
    export_replay_v1_qwen_sft,
    validate_replay_v1_qwen_sft,
)
from data_pipeline.board_recognition.dataset import (
    PROJECT_ROOT,
    read_jsonl,
    resolve_project_path,
    sample_state_queries,
)


JsonDict = dict[str, Any]
SFT_EXPORT_SCHEMA = "catan_board_recognition_qwen_sft/v1"
SFT_ROW_KEYS = {"image", "conversations"}
SPLITS = ("train", "validation", "test")


def export_qwen_sft(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    queries_per_state: int | None = None,
    seed: int = 381_427,
    overwrite: bool = False,
) -> JsonDict:
    dataset_root = Path(dataset_dir)
    dataset_metadata = json.loads((dataset_root / "metadata.json").read_text())
    if dataset_metadata.get("schema") == DATASET_SCHEMA:
        return export_replay_v1_qwen_sft(
            dataset_root,
            output_dir=output_dir,
            queries_per_state=queries_per_state or 8,
            seed=seed,
            overwrite=overwrite,
        )
    queries_per_state = queries_per_state or 16
    export_dir = Path(output_dir) if output_dir is not None else dataset_root / "qwen_sft"
    prepare_output_dir(export_dir, overwrite=overwrite)
    audit_dir = export_dir / "audit"
    audit_dir.mkdir(parents=True)

    metadata = json.loads((dataset_root / "metadata.json").read_text())
    manifest_path = dataset_root / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    spec_path = resolve_project_path(metadata["curriculum_path"])
    spec = json.loads(spec_path.read_text())
    base_states_by_group = {
        row["counterfactual_group_id"]: row
        for row in manifest
        if row["counterfactual_role"] == "base"
    }
    split_counts: dict[str, int] = {}
    entity_counts: Counter[str] = Counter()
    attribute_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    for split in SPLITS:
        sft_rows = []
        audit_rows = []
        split_states = sorted(
            (row for row in manifest if row["split"] == split),
            key=lambda row: row["sample_id"],
        )
        for state in split_states:
            labels = json.loads((dataset_root / state["label_path"]).read_text())
            base_state = base_states_by_group[state["counterfactual_group_id"]]
            sampling_labels = json.loads((dataset_root / base_state["label_path"]).read_text())
            queries = sample_state_queries(
                labels,
                spec=spec,
                queries_per_state=queries_per_state,
                seed=seed,
                epoch=0,
                forced_query=state["counterfactual"]["target"],
                sampling_key=state["counterfactual_group_id"],
                sampling_dense_labels=sampling_labels,
            )
            for query in queries:
                prompt = qwen_query_prompt(query)
                sft_rows.append(
                    {
                        "image": state["image_path"],
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{prompt}"},
                            {"from": "gpt", "value": query["class_name"]},
                        ],
                    }
                )
                audit_rows.append(
                    {
                        "query_id": query["query_id"],
                        "state_id": state["sample_id"],
                        "counterfactual_group_id": state["counterfactual_group_id"],
                        "counterfactual_role": state["counterfactual_role"],
                        "split": split,
                        "stage": state["stage"],
                        "image_path": state["image_path"],
                        "label_path": state["label_path"],
                        "entity_type": query["entity_type"],
                        "attribute": query["attribute"],
                        "slot": query["slot"],
                        "class_name": query["class_name"],
                        "class_index": query["class_index"],
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                )
                entity_counts[query["entity_type"]] += 1
                attribute_counts[query["head"]] += 1
                class_counts[f"{query['head']}.{query['class_name']}"] += 1
        write_jsonl(export_dir / f"{split}.jsonl", sft_rows)
        write_jsonl(audit_dir / f"{split}.jsonl", audit_rows)
        split_counts[split] = len(sft_rows)

    export_metadata = {
        "schema": SFT_EXPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "format": "Qwen3-VL image/conversations JSONL",
        "source_dataset": repository_relative(dataset_root),
        "source_manifest_sha256": file_sha256(manifest_path),
        "curriculum_path": metadata["curriculum_path"],
        "curriculum_sha256": file_sha256(spec_path),
        "queries_per_state": queries_per_state,
        "seed": seed,
        "state_count": len(manifest),
        "row_count": sum(split_counts.values()),
        "split_counts": split_counts,
        "entity_type_counts": dict(sorted(entity_counts.items())),
        "attribute_counts": dict(sorted(attribute_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "qwen_dataset_registration": {
            "annotation_path_relative_to_export": {split: f"{split}.jsonl" for split in SPLITS},
            "data_path": repository_relative(dataset_root),
        },
        "files": {
            split: {
                "annotations": f"{split}.jsonl",
                "annotations_sha256": file_sha256(export_dir / f"{split}.jsonl"),
                "audit": f"audit/{split}.jsonl",
                "audit_sha256": file_sha256(audit_dir / f"{split}.jsonl"),
            }
            for split in SPLITS
        },
    }
    write_json(export_dir / "metadata.json", export_metadata)
    report = validate_qwen_sft_export(dataset_root, export_dir=export_dir)
    report["output_dir"] = str(export_dir)
    return report


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
            labels = json.loads((dataset_root / state["label_path"]).read_text())
            base_state = base_states_by_group[state["counterfactual_group_id"]]
            sampling_labels = json.loads((dataset_root / base_state["label_path"]).read_text())
            for query in sample_state_queries(
                labels,
                spec=spec,
                queries_per_state=int(metadata["queries_per_state"]),
                seed=int(metadata["seed"]),
                epoch=0,
                forced_query=state["counterfactual"]["target"],
                sampling_key=state["counterfactual_group_id"],
                sampling_dense_labels=sampling_labels,
            ):
                expected_queries[query["query_id"]] = query
        for row, audit in zip(rows, audit_rows, strict=True):
            validate_qwen_row(row)
            if audit["query_id"] in seen_query_ids:
                raise ValueError(f"Qwen SFT duplicated query: {audit['query_id']}")
            seen_query_ids.add(audit["query_id"])
            state = states.get(audit["state_id"])
            query = expected_queries.get(audit["query_id"])
            if state is None or query is None or state["split"] != split:
                raise ValueError(f"Qwen SFT audit references unknown state/query: {audit}")
            if any(
                (
                    audit["split"] != split,
                    audit["stage"] != state["stage"],
                    audit["counterfactual_group_id"] != state["counterfactual_group_id"],
                    audit["counterfactual_role"] != state["counterfactual_role"],
                    audit["label_path"] != state["label_path"],
                )
            ):
                raise ValueError(f"Qwen SFT state provenance mismatch: {audit['query_id']}")
            if row["image"] != state["image_path"] or not (dataset_root / row["image"]).is_file():
                raise ValueError(f"Qwen SFT image path mismatch: {audit['query_id']}")
            if row["conversations"][1]["value"] != query["class_name"]:
                raise ValueError(f"Qwen SFT target mismatch: {audit['query_id']}")
            prompt = qwen_query_prompt(query)
            if row["conversations"][0]["value"] != f"<image>\n{prompt}":
                raise ValueError(f"Qwen SFT prompt mismatch: {audit['query_id']}")
            if audit["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
                raise ValueError(f"Qwen SFT prompt hash mismatch: {audit['query_id']}")
            expected_audit = {
                "entity_type": query["entity_type"],
                "attribute": query["attribute"],
                "slot": query["slot"],
                "class_name": query["class_name"],
                "class_index": query["class_index"],
            }
            if any(audit[key] != value for key, value in expected_audit.items()):
                raise ValueError(f"Qwen SFT audit label mismatch: {audit['query_id']}")
            entity_counts[query["entity_type"]] += 1
            attribute_counts[query["head"]] += 1
            class_counts[f"{query['head']}.{query['class_name']}"] += 1
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
        "splits": split_counts,
        "entity_types": dict(sorted(entity_counts.items())),
    }


def qwen_query_prompt(query: JsonDict) -> str:
    classes = ", ".join(query["class_vocabulary"])
    return (
        "Classify one symbolic slot in this ordinary Catan board image.\n"
        f"Entity type: {query['entity_type']}\n"
        f"Slot: {query['slot']}\n"
        f"Attribute: {query['attribute']}\n"
        f"Allowed classes: {classes}\n"
        "Return exactly one allowed class and no explanation."
    )


def validate_qwen_row(row: JsonDict) -> None:
    if set(row) != SFT_ROW_KEYS:
        raise ValueError(f"Qwen SFT row has unsupported keys: {sorted(row)}")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError("Qwen SFT row must contain one human/gpt turn")
    human, assistant = conversations
    if human.get("from") != "human" or assistant.get("from") != "gpt":
        raise ValueError("Qwen SFT conversation roles are invalid")
    if human.get("value", "").count("<image>") != 1:
        raise ValueError("Qwen SFT human turn must contain exactly one <image> tag")
    if "<image>" in assistant.get("value", ""):
        raise ValueError("Qwen SFT answer must not contain an image tag")


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"SFT output directory is not empty: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
