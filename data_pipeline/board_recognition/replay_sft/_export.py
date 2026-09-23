"""Write the replay_v1 Qwen SFT export."""


from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition.query_schedule import (
    build_split_query_plan,
    validate_split_balance,
)
from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    DATASET_SCHEMA,
    DEFAULT_SEED,
    PRIMARY_SPLITS,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.replay_sft._config import (
    EXPORT_SCHEMA,
    QUERIES_PER_STATE,
    JsonDict,
    ReplaySftExportError,
)
from data_pipeline.board_recognition.replay_sft._io import (
    canonical_sha256,
    prepare_output_dir,
    write_json,
    write_jsonl,
)
from data_pipeline.board_recognition.replay_sft._rows import (
    audit_row,
    qwen_row,
)
from data_pipeline.board_recognition.replay_sft._validate import (
    validate_replay_v1_qwen_sft,
)
from data_pipeline.board_recognition.sources import file_sha256
from data_pipeline.json_coerce import as_dict, as_list, as_str
from evals.catan_board_bench.tokens import recognition_token_inventory


def export_replay_v1_qwen_sft(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    queries_per_state: int = QUERIES_PER_STATE,
    seed: int = DEFAULT_SEED,
    epoch_index: int = 0,
    overwrite: bool = False,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / "qwen_sft").resolve()
    )
    dataset_metadata = json.loads((dataset_root / "metadata.json").read_text())
    if dataset_metadata.get("schema") != DATASET_SCHEMA:
        raise ReplaySftExportError("replay_v1 exporter requires the v2 dataset")
    validate_replay_v1_dataset(dataset_root, rerender=False)
    prepare_output_dir(output, overwrite=overwrite)
    plans_dir = output / "query_plans"
    audit_dir = output / "audit"
    plans_dir.mkdir()
    audit_dir.mkdir()

    manifest_path = dataset_root / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    states_by_id = {row["sample_id"]: row for row in manifest}
    split_counts: dict[str, int] = {}
    split_metrics: dict[str, JsonDict] = {}
    all_class_counts: Counter[str] = Counter()

    for split in ALL_SPLITS:
        split_states = [row for row in manifest if row["split"] == split]
        plans = build_split_query_plan(
            split_states,
            dataset_dir=dataset_root,
            split=split,
            seed=seed,
            queries_per_state=queries_per_state,
            epoch_index=epoch_index,
        )
        split_metrics[split] = validate_split_balance(plans, split=split)
        write_jsonl(plans_dir / f"{split}.jsonl", plans)
        annotations = []
        audits = []
        for plan in plans:
            state = states_by_id[plan["state_id"]]
            image_name = Path(as_str(state["image_path"])).name
            for query in map(as_dict, as_list(plan["queries"])):
                annotations.append(qwen_row(query, image_name=image_name))
                audits.append(audit_row(plan, query, state=state))
                all_class_counts[f"{query['head']}.{query['class_name']}"] += 1
        write_jsonl(output / f"{split}.jsonl", annotations)
        write_jsonl(audit_dir / f"{split}.jsonl", audits)
        split_counts[split] = len(annotations)

    token_inventory = recognition_token_inventory()
    write_json(output / "trainable_tokens.json", token_inventory)
    files = {}
    for split in ALL_SPLITS:
        files[split] = {
            "annotations": f"{split}.jsonl",
            "annotations_sha256": file_sha256(output / f"{split}.jsonl"),
            "audit": f"audit/{split}.jsonl",
            "audit_sha256": file_sha256(audit_dir / f"{split}.jsonl"),
            "query_plan": f"query_plans/{split}.jsonl",
            "query_plan_sha256": file_sha256(plans_dir / f"{split}.jsonl"),
        }
    metadata = {
        "schema": EXPORT_SCHEMA,
        "source_dataset_schema": dataset_metadata["schema"],
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_metadata_sha256": file_sha256(dataset_root / "metadata.json"),
        "image_root": str((dataset_root / "images").resolve()),
        "image_root_identity": canonical_sha256(
            {
                as_str(row["image_path"]): as_dict(row["sha256"])["image"]
                for row in sorted(manifest, key=lambda item: as_str(item["image_path"]))
            }
        ),
        "seed": seed,
        "epoch": epoch_index,
        "queries_per_state": queries_per_state,
        "state_counts": dict(sorted(Counter(row["split"] for row in manifest).items())),
        "row_counts": split_counts,
        "primary_row_count": sum(split_counts[split] for split in PRIMARY_SPLITS),
        "diagnostic_row_count": split_counts["color_diagnostic"],
        "trainable_token_inventory": "trainable_tokens.json",
        "trainable_token_inventory_sha256": file_sha256(output / "trainable_tokens.json"),
        "split_metrics": split_metrics,
        "class_counts": dict(sorted(all_class_counts.items())),
        "files": files,
    }
    write_json(output / "metadata.json", metadata)
    report = validate_replay_v1_qwen_sft(dataset_root, export_dir=output)
    report["output_dir"] = str(output)
    return report


__all__ = ["export_replay_v1_qwen_sft"]
