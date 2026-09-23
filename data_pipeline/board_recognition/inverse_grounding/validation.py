"""Reproduce and validate the inverse and mixed exports without relaxing hashes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition import inverse_grounding as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_dict, as_str


def validate_replay_v1_ms_swift_bidirectional(
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
        raise api.InverseGroundingError("bidirectional export schema mismatch")
    manifest_path = dataset_root / "manifest.jsonl"
    if metadata["source_manifest_sha256"] != api.file_sha256(manifest_path):
        raise api.InverseGroundingError("source manifest changed after bidirectional export")
    if metadata["source_metadata_sha256"] != api.file_sha256(dataset_root / "metadata.json"):
        raise api.InverseGroundingError("source metadata changed after bidirectional export")
    if Path(metadata["image_root"]).resolve() != (dataset_root / "images").resolve():
        raise api.InverseGroundingError("bidirectional image_root is invalid")
    contract_path = output / metadata["inverse_grounding_contract"]
    if metadata["inverse_grounding_contract_sha256"] != api.file_sha256(contract_path):
        raise api.InverseGroundingError("inverse grounding contract hash changed")
    if json.loads(contract_path.read_text()) != api.inverse_grounding_contract():
        raise api.InverseGroundingError("inverse grounding contract content changed")
    inventory_path = output / metadata["trainable_token_inventory"]
    if metadata["trainable_token_inventory_sha256"] != api.file_sha256(inventory_path):
        raise api.InverseGroundingError("trainable token inventory hash changed")
    inventory = json.loads(inventory_path.read_text())
    if inventory != api.semantic_recognition_token_inventory():
        raise api.InverseGroundingError("bidirectional inventory is not the exact atlas inventory")

    forward_root = Path(metadata["forward_projection"]).resolve()
    forward_metadata_path = forward_root / "metadata.json"
    if metadata["forward_projection_metadata_sha256"] != api.file_sha256(forward_metadata_path):
        raise api.InverseGroundingError("forward projection metadata changed")
    api.validate_replay_v1_ms_swift_semantic(dataset_root, export_dir=forward_root)
    forward_metadata = json.loads(forward_metadata_path.read_text())

    manifest = api.read_jsonl(manifest_path)
    states_by_split = {
        split: [state for state in manifest if state["split"] == split]
        for split in api.ALL_SPLITS
    }
    schema_root = api.PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "schemas"
    schema_counts: dict[str, int] = {}
    observed_counts: dict[str, int] = {}
    observed_mixed_counts: dict[str, int] = {}
    observed_styles: Counter[str] = Counter()
    observed_targets: Counter[str] = Counter()
    observed_qualifiers: Counter[str] = Counter()

    for split in api.ALL_SPLITS:
        file_metadata = metadata["files"][split]
        paths = {
            key: output / file_metadata[key]
            for key in ("annotations", "audit", "mixed_annotations", "mixed_index")
        }
        for key, path in paths.items():
            if api.file_sha256(path) != file_metadata[f"{key}_sha256"]:
                raise api.InverseGroundingError(f"{split} {key} hash changed")
        inverse_rows = api.read_jsonl(paths["annotations"])
        inverse_audits = api.read_jsonl(paths["audit"])
        mixed_rows = api.read_jsonl(paths["mixed_annotations"])
        mixed_index = api.read_jsonl(paths["mixed_index"])
        schema_counts[f"annotations.{split}"] = api._validate_schema_rows(
            schema_root / "ms_swift_inverse_v1.schema.json", inverse_rows, label=f"annotations.{split}",
        )
        schema_counts[f"audit.{split}"] = api._validate_schema_rows(
            schema_root / "ms_swift_inverse_audit_v1.schema.json", inverse_audits, label=f"audit.{split}",
        )
        expected_inverse_rows: list[JsonDict] = []
        expected_inverse_audits: list[JsonDict] = []
        expected_mixed_rows: list[JsonDict] = []
        expected_mixed_index: list[JsonDict] = []
        forward_annotations = api.read_jsonl(forward_root / forward_metadata["files"][split]["annotations"])
        forward_audits = api.read_jsonl(forward_root / forward_metadata["files"][split]["audit"])
        forward_by_state = api._forward_rows_by_state(forward_annotations, forward_audits)
        for state_index, state in enumerate(states_by_split[split]):
            contract_file = dataset_root / as_str(state["contract_path"])
            if api.file_sha256(contract_file) != as_dict(state["sha256"])["contract"]:
                raise api.InverseGroundingError(f"contract hash changed: {state['sample_id']}")
            contract = json.loads(contract_file.read_text())
            queries = api.inverse_queries_for_state(state, contract, state_index=state_index)
            state_rows = [
                api.inverse_ms_swift_row(query, image_name=Path(as_str(state["image_path"])).name)
                for query in queries
            ]
            state_audits = [
                api.inverse_audit_row(state, query, state_index=state_index)
                for query in queries
            ]
            state_mixed, state_mixed_index = api._mixed_rows_for_state(
                forward_by_state[as_str(state["sample_id"])], state_rows, state_audits
            )
            expected_inverse_rows.extend(state_rows)
            expected_inverse_audits.extend(state_audits)
            expected_mixed_rows.extend(state_mixed)
            expected_mixed_index.extend(state_mixed_index)
        for index_value, row in enumerate(expected_mixed_index):
            row["mixed_index"] = index_value
        if inverse_rows != expected_inverse_rows or inverse_audits != expected_inverse_audits:
            raise api.InverseGroundingError(f"{split} inverse rows are not reproducible")
        if mixed_rows != expected_mixed_rows or mixed_index != expected_mixed_index:
            raise api.InverseGroundingError(f"{split} mixed rows are not reproducible")
        for row in inverse_rows:
            api.validate_inverse_ms_swift_row(row)
        for row, index_row in zip(mixed_rows, mixed_index, strict=True):
            if index_row["row_kind"] == "inverse":
                api.validate_inverse_ms_swift_row(row)
            else:
                api.validate_forward_row(row)
        for audit in inverse_audits:
            observed_styles[as_str(audit["description_style"])] += 1
            observed_targets[as_str(audit["target_token"])] += 1
            if audit["visual_qualifier"] is not None:
                observed_qualifiers[as_str(audit["visual_qualifier"])] += 1
        observed_counts[split] = len(inverse_rows)
        observed_mixed_counts[split] = len(mixed_rows)

    if observed_counts != metadata["row_counts"]:
        raise api.InverseGroundingError("inverse row counts are stale")
    if observed_mixed_counts != metadata["mixed_row_counts"]:
        raise api.InverseGroundingError("mixed row counts are stale")
    if dict(sorted(observed_styles.items())) != metadata["description_style_counts"]:
        raise api.InverseGroundingError("description style counts are stale")
    if dict(sorted(observed_targets.items())) != metadata["target_token_counts"]:
        raise api.InverseGroundingError("target token counts are stale")
    if dict(sorted(observed_qualifiers.items())) != metadata["piece_qualifier_counts"]:
        raise api.InverseGroundingError("piece qualifier counts are stale")
    expected_counts = {
        split: len(states_by_split[split]) * api.INVERSE_ROWS_PER_STATE for split in api.ALL_SPLITS
    }
    expected_mixed = {
        split: len(states_by_split[split]) * api.MIXED_ROWS_PER_STATE for split in api.ALL_SPLITS
    }
    if observed_counts != expected_counts or observed_mixed_counts != expected_mixed:
        raise api.InverseGroundingError("bidirectional split counts are not canonical")
    missing_train_targets = api.ATLAS_TOKENS - {
        audit["target_token"] for audit in api.read_jsonl(output / metadata["files"]["train"]["audit"])
    }
    if missing_train_targets:
        raise api.InverseGroundingError(f"train inverse rows omit atlas targets: {sorted(missing_train_targets)}")
    return {
        "valid": True,
        "states": sum(len(states_by_split[split]) for split in api.PRIMARY_SPLITS),
        "diagnostic_states": len(states_by_split["color_diagnostic"]),
        "train_inverse_rows": observed_counts["train"],
        "train_mixed_rows": observed_mixed_counts["train"],
        "inverse_rows_per_state": api.INVERSE_ROWS_PER_STATE,
        "mixed_rows_per_state": api.MIXED_ROWS_PER_STATE,
        "forward_inverse_ratio": "2:1",
        "train_target_coverage": len(api.ATLAS_TOKENS),
        "piece_qualified_rows": sum(observed_qualifiers.values()),
        "schema_counts": {name: count for name, count in schema_counts.items()},
        "identity_sha256": api._canonical_sha256(
            {
                "metadata": api.file_sha256(output / "metadata.json"),
                "contract": metadata["inverse_grounding_contract_sha256"],
                "inventory": metadata["trainable_token_inventory_sha256"],
            }
        ),
    }
