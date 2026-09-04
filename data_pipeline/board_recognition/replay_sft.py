"""Compact atomic Qwen projection for the replay_v1 recognition corpus."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from jsonschema import Draft202012Validator

from data_pipeline.board_recognition.query_schedule import (
    build_split_query_plan,
    validate_query_plan,
    validate_split_balance,
)
from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    DATASET_SCHEMA,
    DEFAULT_SEED,
    PRIMARY_SPLITS,
    PROJECT_ROOT,
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from evals.catan_board_bench.tokens import recognition_token_inventory


JsonDict = dict[str, Any]
EXPORT_SCHEMA = "catan_board_recognition_qwen_sft/v2"
QUERIES_PER_STATE = 8
SFT_ROW_KEYS = {"image", "conversations"}


class ReplaySftExportError(RuntimeError):
    """Raised when compact replay_v1 projection invariants fail."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def atomic_prompt(query: JsonDict) -> str:
    return f"{query['slot']}{query['query_token']}"


def qwen_row(query: JsonDict, *, image_name: str) -> JsonDict:
    return {
        "image": image_name,
        "conversations": [
            {"from": "human", "value": f"<image>\n{atomic_prompt(query)}"},
            {"from": "gpt", "value": query["answer_token"]},
        ],
    }


def audit_row(plan: JsonDict, query: JsonDict, *, state: JsonDict) -> JsonDict:
    prompt = atomic_prompt(query)
    return {
        "schema": "catan_board_recognition_qwen_audit/v2",
        "query_id": query["query_id"],
        "state_id": plan["state_id"],
        "split": plan["split"],
        "epoch": plan["epoch"],
        "image_name": Path(state["image_path"]).name,
        "image_sha256": state["sha256"]["image"],
        "label_path": state["label_path"],
        "source_kind": state["source"]["kind"],
        "game_id": state["source"].get("game_id"),
        "trajectory_id": state["source"]["trajectory_id"],
        "density_bin": state["density_bin"],
        "entity_type": query["entity_type"],
        "entity_type_id": query["entity_type_id"],
        "attribute": query["attribute"],
        "attribute_id": query["attribute_id"],
        "head": query["head"],
        "slot": query["slot"],
        "slot_index": query["slot_index"],
        "class_name": query["class_name"],
        "class_index": query["class_index"],
        "query_token": query["query_token"],
        "answer_token": query["answer_token"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


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
            image_name = Path(state["image_path"]).name
            for query in plan["queries"]:
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
                row["image_path"]: row["sha256"]["image"]
                for row in sorted(manifest, key=lambda item: item["image_path"])
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


def validate_qwen_row(row: JsonDict) -> None:
    if set(row) != SFT_ROW_KEYS:
        raise ReplaySftExportError(f"Qwen row keys must be {sorted(SFT_ROW_KEYS)}")
    if not isinstance(row["image"], str) or Path(row["image"]).name != row["image"]:
        raise ReplaySftExportError("Qwen image must be a filename relative to image_root")
    conversations = row["conversations"]
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ReplaySftExportError("Qwen row must have one user and one assistant turn")
    if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
        raise ReplaySftExportError("Qwen conversation roles are invalid")
    prompt = conversations[0].get("value")
    answer = conversations[1].get("value")
    if not isinstance(prompt, str) or not prompt.startswith("<image>\n"):
        raise ReplaySftExportError("Qwen prompt must start with exactly one image marker")
    if prompt.count("<image>") != 1:
        raise ReplaySftExportError("Qwen prompt must contain exactly one image marker")
    if not isinstance(answer, str) or not answer.startswith("<A_") or not answer.endswith(">"):
        raise ReplaySftExportError("Qwen target must be one atomic answer token")


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
            [json.loads((dataset_root / row["label_path"]).read_text()) for row in manifest],
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
        expected_rows = sum(len(plan["queries"]) for plan in plans)
        if len(annotations) != expected_rows or len(audits) != expected_rows:
            raise ReplaySftExportError(f"{split} row count is invalid")
        plan_queries = {
            query["query_id"]: (plan, query) for plan in plans for query in plan["queries"]
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
            if annotation != qwen_row(query, image_name=Path(state["image_path"]).name):
                raise ReplaySftExportError(f"annotation mismatch: {query_id}")
            if audit != audit_row(plan, query, state=state):
                raise ReplaySftExportError(f"audit mismatch: {query_id}")
            image_path = Path(metadata["image_root"]) / annotation["image"]
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
