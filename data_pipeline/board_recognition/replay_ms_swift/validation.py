"""Validate the immutable semantic projection against its source query plan."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition import replay_ms_swift as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_dict, as_list, as_str


def validate_replay_v1_ms_swift_semantic(
    dataset_dir: str | Path,
    *,
    export_dir: str | Path | None = None,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(export_dir).resolve()
        if export_dir is not None
        else (dataset_root / api.DEFAULT_OUTPUT_NAME).resolve()
    )
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != api.EXPORT_SCHEMA:
        raise api.SemanticExportError("semantic ms-swift export schema mismatch")
    manifest_path = dataset_root / "manifest.jsonl"
    if metadata["source_manifest_sha256"] != api.file_sha256(manifest_path):
        raise api.SemanticExportError("source manifest changed after semantic export")
    if metadata["source_metadata_sha256"] != api.file_sha256(dataset_root / "metadata.json"):
        raise api.SemanticExportError("source metadata changed after semantic export")
    if Path(metadata["image_root"]).resolve() != (dataset_root / "images").resolve():
        raise api.SemanticExportError("semantic export image_root is not the explicit images root")
    if metadata["semantic_contract_sha256"] != api.file_sha256(
        output / metadata["semantic_contract"]
    ):
        raise api.SemanticExportError("semantic contract hash changed")
    if json.loads((output / metadata["semantic_contract"]).read_text()) != api.semantic_contract():
        raise api.SemanticExportError("semantic contract content changed")
    if metadata["trainable_token_inventory_sha256"] != api.file_sha256(
        output / metadata["trainable_token_inventory"]
    ):
        raise api.SemanticExportError("semantic token inventory hash changed")
    inventory = json.loads((output / metadata["trainable_token_inventory"]).read_text())
    if inventory != api.semantic_recognition_token_inventory() or inventory["counts"]["total"] != 154:
        raise api.SemanticExportError("semantic inventory is not exactly 154 location tokens")

    source_root = Path(metadata["source_projection"]).resolve()
    source_metadata_path = source_root / "metadata.json"
    if metadata["source_projection_metadata_sha256"] != api.file_sha256(source_metadata_path):
        raise api.SemanticExportError("historical source projection metadata changed")
    source_metadata = json.loads(source_metadata_path.read_text())
    if source_metadata.get("schema") != api.QWEN_EXPORT_SCHEMA:
        raise api.SemanticExportError("historical source projection schema changed")
    api.validate_replay_v1_qwen_sft(dataset_root, export_dir=source_root)

    manifest = api.read_jsonl(manifest_path)
    states_by_id = {row["sample_id"]: row for row in manifest}
    observed_counts: dict[str, int] = {}
    observed_density_counts: dict[str, dict[str, int]] = {}
    image_hashes: dict[str, str] = {}
    schema_root = api.PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "schemas"
    schema_counts: dict[str, int] = {}

    for split in api.ALL_SPLITS:
        file_metadata = metadata["files"][split]
        annotation_path = output / file_metadata["annotations"]
        audit_path = output / file_metadata["audit"]
        source_plan_path = dataset_root / file_metadata["source_query_plan"]
        if api.file_sha256(annotation_path) != file_metadata["annotations_sha256"]:
            raise api.SemanticExportError(f"{split} semantic annotations hash changed")
        if api.file_sha256(audit_path) != file_metadata["audit_sha256"]:
            raise api.SemanticExportError(f"{split} semantic audit hash changed")
        if api.file_sha256(source_plan_path) != file_metadata["source_query_plan_sha256"]:
            raise api.SemanticExportError(f"{split} historical query plan hash changed")

        plans = api.read_jsonl(source_plan_path)
        api.validate_query_plan(
            plans,
            expected_states=metadata["state_counts"][split],
            split=split,
            queries_per_state=metadata["queries_per_state"],
            epoch_index=metadata["epoch"],
        )
        api.validate_split_balance(plans, split=split)
        annotations = api.read_jsonl(annotation_path)
        audits = api.read_jsonl(audit_path)
        schema_counts[f"annotations.{split}"] = api._validate_schema_rows(
            schema_root / "ms_swift_semantic_v1.schema.json",
            annotations,
            label=f"annotations.{split}",
        )
        schema_counts[f"audit.{split}"] = api._validate_schema_rows(
            schema_root / "ms_swift_semantic_audit_v1.schema.json",
            audits,
            label=f"audit.{split}",
        )
        expected_rows = sum(len(as_list(plan["queries"])) for plan in plans)
        if len(annotations) != expected_rows or len(audits) != expected_rows:
            raise api.SemanticExportError(f"{split} semantic row count is invalid")
        plan_queries = {
            as_str(as_dict(query)["query_id"]): (plan, as_dict(query))
            for plan in plans
            for query in as_list(plan["queries"])
        }
        seen = set()
        density_counts: Counter[str] = Counter()
        for annotation, audit in zip(annotations, audits, strict=True):
            api.validate_ms_swift_row(annotation)
            query_id = audit["query_id"]
            if query_id in seen or query_id not in plan_queries:
                raise api.SemanticExportError(f"invalid or duplicate semantic query ID: {query_id}")
            seen.add(query_id)
            plan, query = plan_queries[query_id]
            state = states_by_id[plan["state_id"]]
            expected_annotation = api.ms_swift_row(
                query,
                image_name=Path(as_str(state["image_path"])).name,
            )
            if annotation != expected_annotation:
                raise api.SemanticExportError(f"semantic annotation mismatch: {query_id}")
            if audit != api.semantic_audit_row(plan, query, state=state):
                raise api.SemanticExportError(f"semantic audit mismatch: {query_id}")
            messages = [as_dict(turn) for turn in as_list(annotation["messages"])]
            if messages[1]["content"] not in api.semantic_candidates(as_str(query["head"])):
                raise api.SemanticExportError(f"semantic answer is not legal for {query_id}")
            image_name = as_str(as_list(annotation["images"])[0])
            image_path = Path(as_str(metadata["image_root"])) / image_name
            image_digest = image_hashes.get(image_name)
            if image_digest is None:
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                image_digest = api.file_sha256(image_path)
                image_hashes[image_name] = image_digest
            if image_digest != audit["image_sha256"]:
                raise api.SemanticExportError(f"semantic image hash mismatch: {query_id}")
            density_counts[as_str(audit["density_bin"])] += 1
        if seen != set(plan_queries):
            raise api.SemanticExportError(f"{split} omitted scheduled semantic queries")
        observed_counts[split] = len(annotations)
        observed_density_counts[split] = dict(sorted(density_counts.items()))

    if observed_counts != metadata["row_counts"]:
        raise api.SemanticExportError("semantic row counts are stale")
    if observed_density_counts != metadata["density_row_counts"]:
        raise api.SemanticExportError("semantic density counts are stale")
    expected_counts = {"train": 8192, "validation": 512, "test": 512, "color_diagnostic": 512}
    if observed_counts != expected_counts:
        raise api.SemanticExportError(f"semantic split row counts are not canonical: {observed_counts}")
    return {
        "valid": True,
        "states": sum(metadata["state_counts"][split] for split in api.PRIMARY_SPLITS),
        "diagnostic_states": metadata["state_counts"]["color_diagnostic"],
        "rows_per_epoch": sum(observed_counts[split] for split in api.PRIMARY_SPLITS),
        "train_rows_per_epoch": observed_counts["train"],
        "diagnostic_rows": observed_counts["color_diagnostic"],
        "queries_per_state": metadata["queries_per_state"],
        "epoch": metadata["epoch"],
        "trainable_tokens": inventory["counts"],
        "density_row_counts": observed_density_counts,
        "schema_counts": schema_counts,
    }
