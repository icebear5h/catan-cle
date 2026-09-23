"""Re-derive and schema-check a written replay_v1 Qwen SFT export."""


from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from jsonschema import Draft202012Validator

from data_pipeline.board_recognition.query_schedule import (
    build_split_query_plan,
    validate_query_plan,
    validate_split_balance,
)
from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    PRIMARY_SPLITS,
    read_jsonl,
)
from data_pipeline.board_recognition.replay_sft._config import (
    EXPORT_SCHEMA,
    JsonDict,
    ReplaySftExportError,
)
from data_pipeline.board_recognition.replay_sft._rows import (
    audit_row,
    qwen_row,
    validate_qwen_row,
)
from data_pipeline.board_recognition.sources import PROJECT_ROOT, file_sha256
from data_pipeline.json_coerce import as_dict, as_list, as_str
from evals.catan_board_bench.tokens import recognition_token_inventory


def _validate_schema_rows(
    schema_path: Path,
    rows: Sequence[JsonDict],
    *,
    label: str,
) -> int:
    validator = Draft202012Validator(json.loads(schema_path.read_text()))
    for index, row in enumerate(rows):
        errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        if errors:
            error = errors[0]
            path = ".".join(str(part) for part in error.path) or "<root>"
            raise ReplaySftExportError(
                f"{label}[{index}] failed {schema_path.name} at {path}: {error.message}"
            )
    return len(rows)


def validate_replay_v1_schemas(
    dataset_root: Path,
    export_root: Path,
) -> dict[str, int]:
    schema_root = PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "schemas"
    manifest = read_jsonl(dataset_root / "manifest.jsonl")
    counts = {
        "manifest": _validate_schema_rows(
            schema_root / "manifest_v2.schema.json",
            manifest,
            label="manifest",
        ),
        "dense_labels": _validate_schema_rows(
            schema_root / "dense_labels_v2.schema.json",
            [
                as_dict(json.loads((dataset_root / as_str(row["label_path"])).read_text()))
                for row in manifest
            ],
            label="dense_labels",
        ),
    }
    for split in ALL_SPLITS:
        counts[f"query_plan.{split}"] = _validate_schema_rows(
            schema_root / "query_plan_v1.schema.json",
            read_jsonl(export_root / "query_plans" / f"{split}.jsonl"),
            label=f"query_plan.{split}",
        )
        counts[f"qwen_sft.{split}"] = _validate_schema_rows(
            schema_root / "qwen_sft_v2.schema.json",
            read_jsonl(export_root / f"{split}.jsonl"),
            label=f"qwen_sft.{split}",
        )
        counts[f"qwen_audit.{split}"] = _validate_schema_rows(
            schema_root / "qwen_audit_v2.schema.json",
            read_jsonl(export_root / "audit" / f"{split}.jsonl"),
            label=f"qwen_audit.{split}",
        )
    return counts


def validate_replay_v1_qwen_sft(
    dataset_dir: str | Path,
    *,
    export_dir: str | Path | None = None,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(export_dir).resolve()
        if export_dir is not None
        else (dataset_root / "qwen_sft").resolve()
    )
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != EXPORT_SCHEMA:
        raise ReplaySftExportError("Qwen SFT export schema mismatch")
    manifest_path = dataset_root / "manifest.jsonl"
    if metadata["source_manifest_sha256"] != file_sha256(manifest_path):
        raise ReplaySftExportError("source manifest changed after Qwen export")
    if metadata["source_metadata_sha256"] != file_sha256(dataset_root / "metadata.json"):
        raise ReplaySftExportError("source metadata changed after Qwen export")
    if Path(metadata["image_root"]).resolve() != (dataset_root / "images").resolve():
        raise ReplaySftExportError("Qwen export image_root is not explicit dataset images root")
    if metadata["trainable_token_inventory_sha256"] != file_sha256(
        output / metadata["trainable_token_inventory"]
    ):
        raise ReplaySftExportError("trainable token inventory changed")
    inventory = json.loads((output / metadata["trainable_token_inventory"]).read_text())
    if inventory != recognition_token_inventory() or inventory["counts"] != {
        "atlas": 154,
        "query": 6,
        "answer": 60,
        "total": 220,
    }:
        raise ReplaySftExportError("trainable token inventory is not exactly 154+6+60")

    manifest = read_jsonl(manifest_path)
    states_by_id = {row["sample_id"]: row for row in manifest}
    observed_counts = {}
    for split in ALL_SPLITS:
        file_metadata = metadata["files"][split]
        for key, hash_key in (
            ("annotations", "annotations_sha256"),
            ("audit", "audit_sha256"),
            ("query_plan", "query_plan_sha256"),
        ):
            if file_sha256(output / file_metadata[key]) != file_metadata[hash_key]:
                raise ReplaySftExportError(f"{split} {key} hash changed")
        plans = read_jsonl(output / file_metadata["query_plan"])
        validate_query_plan(
            plans,
            expected_states=metadata["state_counts"][split],
            split=split,
            queries_per_state=metadata["queries_per_state"],
            epoch_index=metadata["epoch"],
        )
        metrics = validate_split_balance(plans, split=split)
        if metrics != metadata["split_metrics"][split]:
            raise ReplaySftExportError(f"{split} query metrics are stale")
        expected_plans = build_split_query_plan(
            [row for row in manifest if row["split"] == split],
            dataset_dir=dataset_root,
            split=split,
            seed=metadata["seed"],
            queries_per_state=metadata["queries_per_state"],
            epoch_index=metadata["epoch"],
        )
        if plans != expected_plans:
            raise ReplaySftExportError(f"{split} query plan is not reproducible")
        annotations = read_jsonl(output / file_metadata["annotations"])
        audits = read_jsonl(output / file_metadata["audit"])
        expected_rows = sum(len(as_list(plan["queries"])) for plan in plans)
        if len(annotations) != expected_rows or len(audits) != expected_rows:
            raise ReplaySftExportError(f"{split} row count is invalid")
        plan_queries = {
            as_str(as_dict(query)["query_id"]): (plan, as_dict(query))
            for plan in plans
            for query in as_list(plan["queries"])
        }
        seen = set()
        for annotation, audit in zip(annotations, audits, strict=True):
            validate_qwen_row(annotation)
            query_id = audit["query_id"]
            if query_id in seen or query_id not in plan_queries:
                raise ReplaySftExportError(f"invalid or duplicate query ID: {query_id}")
            seen.add(query_id)
            plan, query = plan_queries[query_id]
            state = states_by_id[plan["state_id"]]
            if annotation != qwen_row(query, image_name=Path(as_str(state["image_path"])).name):
                raise ReplaySftExportError(f"annotation mismatch: {query_id}")
            if audit != audit_row(plan, query, state=state):
                raise ReplaySftExportError(f"audit mismatch: {query_id}")
            image_path = Path(as_str(metadata["image_root"])) / as_str(annotation["image"])
            if not image_path.is_file() or file_sha256(image_path) != audit["image_sha256"]:
                raise ReplaySftExportError(f"image root/hash mismatch: {query_id}")
        if seen != set(plan_queries):
            raise ReplaySftExportError(f"{split} omitted scheduled queries")
        observed_counts[split] = len(annotations)
    if observed_counts != metadata["row_counts"]:
        raise ReplaySftExportError("Qwen row counts are stale")
    schema_counts = validate_replay_v1_schemas(dataset_root, output)
    return {
        "valid": True,
        "states": sum(metadata["state_counts"][split] for split in PRIMARY_SPLITS),
        "diagnostic_states": metadata["state_counts"]["color_diagnostic"],
        "rows_per_epoch": sum(observed_counts[split] for split in PRIMARY_SPLITS),
        "train_rows_per_epoch": observed_counts["train"],
        "diagnostic_rows": observed_counts["color_diagnostic"],
        "queries_per_state": metadata["queries_per_state"],
        "epoch": metadata["epoch"],
        "trainable_tokens": inventory["counts"],
        "schema_counts": schema_counts,
    }


__all__ = ["_validate_schema_rows", "validate_replay_v1_qwen_sft", "validate_replay_v1_schemas"]
