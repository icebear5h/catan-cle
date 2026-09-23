"""Semantic ms-swift projection for the immutable replay_v1 query plans."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from jsonschema import Draft202012Validator

from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.tokens import (
    semantic_recognition_token_inventory as semantic_recognition_token_inventory,
)

from ..query_schedule import validate_query_plan as validate_query_plan
from ..query_schedule import validate_split_balance as validate_split_balance
from ..replay_dataset import ALL_SPLITS as ALL_SPLITS
from ..replay_dataset import PRIMARY_SPLITS as PRIMARY_SPLITS
from ..replay_dataset import read_jsonl as read_jsonl
from ..replay_dataset import validate_replay_v1_dataset
from ..replay_sft import EXPORT_SCHEMA as _QWEN_EXPORT_SCHEMA
from ..replay_sft import validate_replay_v1_qwen_sft as validate_replay_v1_qwen_sft
from ..semantics import semantic_answer, semantic_prompt
from ..semantics import semantic_candidates as semantic_candidates
from ..semantics import semantic_contract as semantic_contract
from ..sources import PROJECT_ROOT as PROJECT_ROOT
from ..sources import file_sha256 as file_sha256
from .validation import validate_replay_v1_ms_swift_semantic as validate_replay_v1_ms_swift_semantic

QWEN_EXPORT_SCHEMA = _QWEN_EXPORT_SCHEMA
EXPORT_SCHEMA = "catan_board_recognition_ms_swift_semantic/v1"
AUDIT_SCHEMA = "catan_board_recognition_ms_swift_semantic_audit/v1"
QUERIES_PER_STATE = 8
DEFAULT_OUTPUT_NAME = "ms_swift_semantic_v1"
DEFAULT_SOURCE_PROJECTION = "qwen_sft"
ROW_KEYS = {"messages", "images"}


class SemanticExportError(RuntimeError):
    """Raised when the semantic replay projection violates its contract."""


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_output_dir(path: Path, *, dataset_root: Path, overwrite: bool) -> None:
    historical = (dataset_root / DEFAULT_SOURCE_PROJECTION).resolve()
    if path == historical:
        raise SemanticExportError("refusing to overwrite the historical qwen_sft projection")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def semantic_user_content(query: JsonDict) -> str:
    return f"<image>\n{semantic_prompt(as_str(query['head']), as_str(query['slot']))}"


def ms_swift_row(query: JsonDict, *, image_name: str) -> JsonDict:
    return {
        "messages": [
            {"role": "user", "content": semantic_user_content(query)},
            {
                "role": "assistant",
                "content": semantic_answer(as_str(query["head"]), as_str(query["class_name"])),
            },
        ],
        "images": [image_name],
    }


def semantic_audit_row(plan: JsonDict, query: JsonDict, *, state: JsonDict) -> JsonDict:
    prompt = semantic_user_content(query)
    answer = semantic_answer(as_str(query["head"]), as_str(query["class_name"]))
    candidates = list(semantic_candidates(as_str(query["head"])))
    return {
        "schema": AUDIT_SCHEMA,
        "query_id": query["query_id"],
        "state_id": plan["state_id"],
        "split": plan["split"],
        "epoch": plan["epoch"],
        "image_name": Path(as_str(state["image_path"])).name,
        "image_sha256": as_dict(state["sha256"])["image"],
        "label_path": state["label_path"],
        "source_kind": as_dict(state["source"])["kind"],
        "game_id": as_dict(state["source"]).get("game_id"),
        "trajectory_id": as_dict(state["source"])["trajectory_id"],
        "density_bin": state["density_bin"],
        "building_count": state["building_count"],
        "road_count": state["road_count"],
        "piece_count": as_int(state["building_count"]) + as_int(state["road_count"]),
        "entity_type": query["entity_type"],
        "entity_type_id": query["entity_type_id"],
        "attribute": query["attribute"],
        "attribute_id": query["attribute_id"],
        "head": query["head"],
        "slot": query["slot"],
        "slot_index": query["slot_index"],
        "class_name": query["class_name"],
        "class_index": query["class_index"],
        "semantic_prompt": prompt,
        "semantic_answer": answer,
        "candidate_answers": [name for name in candidates],
        "source_query_token": query["query_token"],
        "source_answer_token": query["answer_token"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "candidates_sha256": canonical_sha256(candidates),
    }


def validate_ms_swift_row(row: JsonDict) -> None:
    if set(row) != ROW_KEYS:
        raise SemanticExportError(f"ms-swift row keys must be {sorted(ROW_KEYS)}")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise SemanticExportError("ms-swift row must have one user and one assistant turn")
    if (
        as_dict(messages[0]).get("role") != "user"
        or as_dict(messages[1]).get("role") != "assistant"
    ):
        raise SemanticExportError("ms-swift message roles are invalid")
    prompt = as_dict(messages[0]).get("content")
    answer = as_dict(messages[1]).get("content")
    if not isinstance(prompt, str) or not prompt.startswith("<image>\n"):
        raise SemanticExportError("ms-swift prompt must start with exactly one image marker")
    if prompt.count("<image>") != 1 or prompt.count("<Q_") or prompt.count("<A_"):
        raise SemanticExportError("ms-swift prompt contains an invalid opaque token")
    if not isinstance(answer, str) or not answer or "<" in answer or ">" in answer:
        raise SemanticExportError("ms-swift answer must be non-empty ordinary language")
    images = row["images"]
    if not isinstance(images, list) or len(images) != 1:
        raise SemanticExportError("ms-swift row must reference exactly one image")
    image_name = images[0]
    if not isinstance(image_name, str) or Path(image_name).name != image_name:
        raise SemanticExportError("ms-swift image must be a filename relative to image_root")


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
            raise SemanticExportError(
                f"{label}[{index}] failed {schema_path.name} at {path}: {error.message}"
            )
    return len(rows)


def export_replay_v1_ms_swift_semantic(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    source_projection_dir: str | Path | None = None,
    overwrite: bool = False,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    source_root = (
        Path(source_projection_dir).resolve()
        if source_projection_dir is not None
        else (dataset_root / DEFAULT_SOURCE_PROJECTION).resolve()
    )
    validate_replay_v1_dataset(dataset_root, rerender=False)
    source_report = validate_replay_v1_qwen_sft(dataset_root, export_dir=source_root)
    source_metadata_path = source_root / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text())
    if source_metadata.get("schema") != QWEN_EXPORT_SCHEMA:
        raise SemanticExportError("semantic projection requires the historical Qwen v2 source")
    if source_metadata["epoch"] != 0 or source_metadata["queries_per_state"] != QUERIES_PER_STATE:
        raise SemanticExportError("semantic projection requires the immutable epoch-0 eight-query plan")

    prepare_output_dir(output, dataset_root=dataset_root, overwrite=overwrite)
    audit_dir = output / "audit"
    audit_dir.mkdir()
    manifest_path = dataset_root / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    states_by_id = {row["sample_id"]: row for row in manifest}
    split_counts: dict[str, int] = {}
    split_density_counts: dict[str, dict[str, int]] = {}
    class_counts: Counter[str] = Counter()
    files: dict[str, JsonDict] = {}

    for split in ALL_SPLITS:
        source_file = source_metadata["files"][split]
        source_plan_path = source_root / source_file["query_plan"]
        if file_sha256(source_plan_path) != source_file["query_plan_sha256"]:
            raise SemanticExportError(f"historical {split} query plan hash changed")
        plans = read_jsonl(source_plan_path)
        validate_query_plan(
            plans,
            expected_states=source_metadata["state_counts"][split],
            split=split,
            queries_per_state=QUERIES_PER_STATE,
            epoch_index=0,
        )
        annotations = []
        audits = []
        density_counts: Counter[str] = Counter()
        for plan in plans:
            state = states_by_id[plan["state_id"]]
            image_name = Path(as_str(state["image_path"])).name
            for query in map(as_dict, as_list(plan["queries"])):
                annotations.append(ms_swift_row(query, image_name=image_name))
                audits.append(semantic_audit_row(plan, query, state=state))
                density_counts[as_str(state["density_bin"])] += 1
                class_counts[f"{query['head']}.{query['class_name']}"] += 1
        annotation_path = output / f"{split}.jsonl"
        audit_path = audit_dir / f"{split}.jsonl"
        write_jsonl(annotation_path, annotations)
        write_jsonl(audit_path, audits)
        split_counts[split] = len(annotations)
        split_density_counts[split] = dict(sorted(density_counts.items()))
        files[split] = {
            "annotations": annotation_path.name,
            "annotations_sha256": file_sha256(annotation_path),
            "audit": f"audit/{audit_path.name}",
            "audit_sha256": file_sha256(audit_path),
            "source_query_plan": str(source_plan_path.relative_to(dataset_root)),
            "source_query_plan_sha256": source_file["query_plan_sha256"],
        }

    contract = semantic_contract()
    inventory = semantic_recognition_token_inventory()
    write_json(output / "semantic_contract.json", contract)
    write_json(output / "trainable_tokens.json", inventory)
    metadata = {
        "schema": EXPORT_SCHEMA,
        "source_projection_schema": source_metadata["schema"],
        "source_projection": str(source_root),
        "source_projection_metadata_sha256": file_sha256(source_metadata_path),
        "source_projection_validation": source_report,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_metadata_sha256": file_sha256(dataset_root / "metadata.json"),
        "image_root": str((dataset_root / "images").resolve()),
        "image_root_identity": source_metadata["image_root_identity"],
        "epoch": 0,
        "queries_per_state": QUERIES_PER_STATE,
        "state_counts": source_metadata["state_counts"],
        "row_counts": split_counts,
        "primary_row_count": sum(split_counts[split] for split in PRIMARY_SPLITS),
        "diagnostic_row_count": split_counts["color_diagnostic"],
        "density_row_counts": split_density_counts,
        "class_counts": dict(sorted(class_counts.items())),
        "semantic_contract": "semantic_contract.json",
        "semantic_contract_sha256": file_sha256(output / "semantic_contract.json"),
        "trainable_token_inventory": "trainable_tokens.json",
        "trainable_token_inventory_sha256": file_sha256(output / "trainable_tokens.json"),
        "files": files,
    }
    write_json(output / "metadata.json", metadata)
    report = validate_replay_v1_ms_swift_semantic(dataset_root, export_dir=output)
    report["output_dir"] = str(output)
    return report
