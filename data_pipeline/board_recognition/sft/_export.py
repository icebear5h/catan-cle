"""Project a dense board-recognition dataset into Qwen conversation rows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from data_pipeline.board_recognition.dataset import (
    read_jsonl,
    resolve_project_path,
    sample_state_queries,
)
from data_pipeline.board_recognition.replay_dataset import DATASET_SCHEMA
from data_pipeline.board_recognition.replay_sft import export_replay_v1_qwen_sft
from data_pipeline.board_recognition.sft._config import (
    SFT_EXPORT_SCHEMA,
    SPLITS,
    JsonDict,
)
from data_pipeline.board_recognition.sft._io import (
    file_sha256,
    prepare_output_dir,
    repository_relative,
    write_json,
    write_jsonl,
)
from data_pipeline.board_recognition.sft._rows import qwen_query_prompt
from data_pipeline.board_recognition.sft._validate import validate_qwen_sft_export
from data_pipeline.json_coerce import as_dict, as_str


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
        sft_rows: list[JsonDict] = []
        audit_rows = []
        split_states = sorted(
            (row for row in manifest if row["split"] == split),
            key=lambda row: as_str(row["sample_id"]),
        )
        for state in split_states:
            labels = as_dict(json.loads((dataset_root / as_str(state["label_path"])).read_text()))
            base_state = base_states_by_group[state["counterfactual_group_id"]]
            sampling_labels = as_dict(
                json.loads((dataset_root / as_str(base_state["label_path"])).read_text())
            )
            queries = sample_state_queries(
                labels,
                spec=spec,
                queries_per_state=queries_per_state,
                seed=seed,
                epoch=0,
                forced_query=as_dict(as_dict(state["counterfactual"])["target"]),
                sampling_key=as_str(state["counterfactual_group_id"]),
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
                entity_counts[as_str(query["entity_type"])] += 1
                attribute_counts[as_str(query["head"])] += 1
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


__all__ = ["export_qwen_sft"]
