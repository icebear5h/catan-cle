"""Semantic ms-swift projection for the immutable replay_v1 query plans."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from jsonschema import Draft202012Validator

from data_pipeline.board_recognition.query_schedule import (
    validate_query_plan,
    validate_split_balance,
)
from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    PRIMARY_SPLITS,
    PROJECT_ROOT,
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.replay_sft import (
    EXPORT_SCHEMA as QWEN_EXPORT_SCHEMA,
    validate_replay_v1_qwen_sft,
)
from data_pipeline.board_recognition.semantics import (
    semantic_answer,
    semantic_candidates,
    semantic_contract,
    semantic_prompt,
)
from evals.catan_board_bench.tokens import semantic_recognition_token_inventory


JsonDict = dict[str, Any]
EXPORT_SCHEMA = "catan_board_recognition_ms_swift_semantic/v1"
AUDIT_SCHEMA = "catan_board_recognition_ms_swift_semantic_audit/v1"
QUERIES_PER_STATE = 8
DEFAULT_OUTPUT_NAME = "ms_swift_semantic_v1"
DEFAULT_SOURCE_PROJECTION = "qwen_sft"
ROW_KEYS = {"messages", "images"}


class SemanticExportError(RuntimeError):
    """Raised when the semantic replay projection violates its contract."""


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
    return f"<image>\n{semantic_prompt(query['head'], query['slot'])}"


def ms_swift_row(query: JsonDict, *, image_name: str) -> JsonDict:
    return {
        "messages": [
            {"role": "user", "content": semantic_user_content(query)},
            {
                "role": "assistant",
                "content": semantic_answer(query["head"], query["class_name"]),
            },
        ],
        "images": [image_name],
    }


def semantic_audit_row(plan: JsonDict, query: JsonDict, *, state: JsonDict) -> JsonDict:
    prompt = semantic_user_content(query)
    answer = semantic_answer(query["head"], query["class_name"])
    candidates = list(semantic_candidates(query["head"]))
    return {
        "schema": AUDIT_SCHEMA,
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
        "building_count": state["building_count"],
        "road_count": state["road_count"],
        "piece_count": state["building_count"] + state["road_count"],
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
        "candidate_answers": candidates,
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
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise SemanticExportError("ms-swift message roles are invalid")
    prompt = messages[0].get("content")
    answer = messages[1].get("content")
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
            image_name = Path(state["image_path"]).name
            for query in plan["queries"]:
                annotations.append(ms_swift_row(query, image_name=image_name))
                audits.append(semantic_audit_row(plan, query, state=state))
                density_counts[state["density_bin"]] += 1
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


def validate_replay_v1_ms_swift_semantic(
    dataset_dir: str | Path,
    *,
    export_dir: str | Path | None = None,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(export_dir).resolve()
        if export_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != EXPORT_SCHEMA:
        raise SemanticExportError("semantic ms-swift export schema mismatch")
    manifest_path = dataset_root / "manifest.jsonl"
    if metadata["source_manifest_sha256"] != file_sha256(manifest_path):
        raise SemanticExportError("source manifest changed after semantic export")
    if metadata["source_metadata_sha256"] != file_sha256(dataset_root / "metadata.json"):
        raise SemanticExportError("source metadata changed after semantic export")
    if Path(metadata["image_root"]).resolve() != (dataset_root / "images").resolve():
        raise SemanticExportError("semantic export image_root is not the explicit images root")
    if metadata["semantic_contract_sha256"] != file_sha256(
        output / metadata["semantic_contract"]
    ):
        raise SemanticExportError("semantic contract hash changed")
    if json.loads((output / metadata["semantic_contract"]).read_text()) != semantic_contract():
        raise SemanticExportError("semantic contract content changed")
    if metadata["trainable_token_inventory_sha256"] != file_sha256(
        output / metadata["trainable_token_inventory"]
    ):
        raise SemanticExportError("semantic token inventory hash changed")
    inventory = json.loads((output / metadata["trainable_token_inventory"]).read_text())
    if inventory != semantic_recognition_token_inventory() or inventory["counts"]["total"] != 154:
        raise SemanticExportError("semantic inventory is not exactly 154 location tokens")

    source_root = Path(metadata["source_projection"]).resolve()
    source_metadata_path = source_root / "metadata.json"
    if metadata["source_projection_metadata_sha256"] != file_sha256(source_metadata_path):
        raise SemanticExportError("historical source projection metadata changed")
    source_metadata = json.loads(source_metadata_path.read_text())
    if source_metadata.get("schema") != QWEN_EXPORT_SCHEMA:
        raise SemanticExportError("historical source projection schema changed")
    validate_replay_v1_qwen_sft(dataset_root, export_dir=source_root)

    manifest = read_jsonl(manifest_path)
    states_by_id = {row["sample_id"]: row for row in manifest}
    observed_counts: dict[str, int] = {}
    observed_density_counts: dict[str, dict[str, int]] = {}
    image_hashes: dict[str, str] = {}
    schema_root = PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "schemas"
    schema_counts: dict[str, int] = {}

    for split in ALL_SPLITS:
        file_metadata = metadata["files"][split]
        annotation_path = output / file_metadata["annotations"]
        audit_path = output / file_metadata["audit"]
        source_plan_path = dataset_root / file_metadata["source_query_plan"]
        if file_sha256(annotation_path) != file_metadata["annotations_sha256"]:
            raise SemanticExportError(f"{split} semantic annotations hash changed")
        if file_sha256(audit_path) != file_metadata["audit_sha256"]:
            raise SemanticExportError(f"{split} semantic audit hash changed")
        if file_sha256(source_plan_path) != file_metadata["source_query_plan_sha256"]:
            raise SemanticExportError(f"{split} historical query plan hash changed")

        plans = read_jsonl(source_plan_path)
        validate_query_plan(
            plans,
            expected_states=metadata["state_counts"][split],
            split=split,
            queries_per_state=metadata["queries_per_state"],
            epoch_index=metadata["epoch"],
        )
        validate_split_balance(plans, split=split)
        annotations = read_jsonl(annotation_path)
        audits = read_jsonl(audit_path)
        schema_counts[f"annotations.{split}"] = _validate_schema_rows(
            schema_root / "ms_swift_semantic_v1.schema.json",
            annotations,
            label=f"annotations.{split}",
        )
        schema_counts[f"audit.{split}"] = _validate_schema_rows(
            schema_root / "ms_swift_semantic_audit_v1.schema.json",
            audits,
            label=f"audit.{split}",
        )
        expected_rows = sum(len(plan["queries"]) for plan in plans)
        if len(annotations) != expected_rows or len(audits) != expected_rows:
            raise SemanticExportError(f"{split} semantic row count is invalid")
        plan_queries = {
            query["query_id"]: (plan, query) for plan in plans for query in plan["queries"]
        }
        seen = set()
        density_counts: Counter[str] = Counter()
        for annotation, audit in zip(annotations, audits, strict=True):
            validate_ms_swift_row(annotation)
            query_id = audit["query_id"]
            if query_id in seen or query_id not in plan_queries:
                raise SemanticExportError(f"invalid or duplicate semantic query ID: {query_id}")
            seen.add(query_id)
            plan, query = plan_queries[query_id]
            state = states_by_id[plan["state_id"]]
            expected_annotation = ms_swift_row(
                query,
                image_name=Path(state["image_path"]).name,
            )
            if annotation != expected_annotation:
                raise SemanticExportError(f"semantic annotation mismatch: {query_id}")
            if audit != semantic_audit_row(plan, query, state=state):
                raise SemanticExportError(f"semantic audit mismatch: {query_id}")
            if annotation["messages"][1]["content"] not in semantic_candidates(query["head"]):
                raise SemanticExportError(f"semantic answer is not legal for {query_id}")
            image_name = annotation["images"][0]
            image_path = Path(metadata["image_root"]) / image_name
            image_digest = image_hashes.get(image_name)
            if image_digest is None:
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                image_digest = file_sha256(image_path)
                image_hashes[image_name] = image_digest
            if image_digest != audit["image_sha256"]:
                raise SemanticExportError(f"semantic image hash mismatch: {query_id}")
            density_counts[audit["density_bin"]] += 1
        if seen != set(plan_queries):
            raise SemanticExportError(f"{split} omitted scheduled semantic queries")
        observed_counts[split] = len(annotations)
        observed_density_counts[split] = dict(sorted(density_counts.items()))

    if observed_counts != metadata["row_counts"]:
        raise SemanticExportError("semantic row counts are stale")
    if observed_density_counts != metadata["density_row_counts"]:
        raise SemanticExportError("semantic density counts are stale")
    expected_counts = {"train": 8192, "validation": 512, "test": 512, "color_diagnostic": 512}
    if observed_counts != expected_counts:
        raise SemanticExportError(f"semantic split row counts are not canonical: {observed_counts}")
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
        "density_row_counts": observed_density_counts,
        "schema_counts": schema_counts,
    }
