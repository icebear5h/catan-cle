"""Write inverse queries and the reproducible bidirectional projection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition import inverse_grounding as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_dict, as_str


def export_replay_v1_ms_swift_bidirectional(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    forward_projection_dir: str | Path | None = None,
    overwrite: bool = False,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / api.DEFAULT_OUTPUT_NAME).resolve()
    )
    forward_root = (
        Path(forward_projection_dir).resolve()
        if forward_projection_dir is not None
        else (dataset_root / api.DEFAULT_FORWARD_PROJECTION).resolve()
    )
    api.validate_replay_v1_dataset(dataset_root, rerender=False)
    forward_report = api.validate_replay_v1_ms_swift_semantic(dataset_root, export_dir=forward_root)
    api._prepare_output_dir(output, dataset_root=dataset_root, overwrite=overwrite)
    (output / "audit").mkdir()
    (output / "mixed").mkdir()
    (output / "mixed_index").mkdir()

    manifest_path = dataset_root / "manifest.jsonl"
    manifest = api.read_jsonl(manifest_path)
    states_by_split = {
        split: [state for state in manifest if state["split"] == split]
        for split in api.ALL_SPLITS
    }
    forward_metadata_path = forward_root / "metadata.json"
    forward_metadata = json.loads(forward_metadata_path.read_text())
    files: dict[str, JsonDict] = {}
    row_counts: dict[str, int] = {}
    mixed_row_counts: dict[str, int] = {}
    style_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    qualified_counts: Counter[str] = Counter()

    for split in api.ALL_SPLITS:
        states = states_by_split[split]
        forward_annotations = api.read_jsonl(forward_root / forward_metadata["files"][split]["annotations"])
        forward_audits = api.read_jsonl(forward_root / forward_metadata["files"][split]["audit"])
        forward_by_state = api._forward_rows_by_state(forward_annotations, forward_audits)
        inverse_rows: list[JsonDict] = []
        inverse_audits: list[JsonDict] = []
        mixed_rows: list[JsonDict] = []
        mixed_index: list[JsonDict] = []
        for state_index, state in enumerate(states):
            contract_path = dataset_root / as_str(state["contract_path"])
            if api.file_sha256(contract_path) != as_dict(state["sha256"])["contract"]:
                raise api.InverseGroundingError(f"contract hash changed: {state['sample_id']}")
            contract = json.loads(contract_path.read_text())
            queries = api.inverse_queries_for_state(state, contract, state_index=state_index)
            state_rows = [
                api.inverse_ms_swift_row(query, image_name=Path(as_str(state["image_path"])).name)
                for query in queries
            ]
            state_audits = [
                api.inverse_audit_row(state, query, state_index=state_index)
                for query in queries
            ]
            forward = forward_by_state.get(as_str(state["sample_id"]))
            if forward is None:
                raise api.InverseGroundingError(f"forward projection omitted state: {state['sample_id']}")
            state_mixed, state_index_rows = api._mixed_rows_for_state(forward, state_rows, state_audits)
            inverse_rows.extend(state_rows)
            inverse_audits.extend(state_audits)
            mixed_rows.extend(state_mixed)
            mixed_index.extend(state_index_rows)
            for query in queries:
                style_counts[as_str(query["description_style"])] += 1
                target_counts[as_str(query["target_token"])] += 1
                if query["visual_qualifier"] is not None:
                    qualified_counts[as_str(query["visual_qualifier"])] += 1

        for index_value, row in enumerate(mixed_index):
            row["mixed_index"] = index_value
        annotation_path = output / f"{split}.jsonl"
        audit_path = output / "audit" / f"{split}.jsonl"
        mixed_path = output / "mixed" / f"{split}.jsonl"
        mixed_index_path = output / "mixed_index" / f"{split}.jsonl"
        api._write_jsonl(annotation_path, inverse_rows)
        api._write_jsonl(audit_path, inverse_audits)
        api._write_jsonl(mixed_path, mixed_rows)
        api._write_jsonl(mixed_index_path, mixed_index)
        row_counts[split] = len(inverse_rows)
        mixed_row_counts[split] = len(mixed_rows)
        files[split] = {
            "annotations": annotation_path.name,
            "annotations_sha256": api.file_sha256(annotation_path),
            "audit": str(audit_path.relative_to(output)),
            "audit_sha256": api.file_sha256(audit_path),
            "mixed_annotations": str(mixed_path.relative_to(output)),
            "mixed_annotations_sha256": api.file_sha256(mixed_path),
            "mixed_index": str(mixed_index_path.relative_to(output)),
            "mixed_index_sha256": api.file_sha256(mixed_index_path),
        }

    contract_payload = api.inverse_grounding_contract()
    inventory = api.semantic_recognition_token_inventory()
    api._write_json(output / "inverse_grounding_contract.json", contract_payload)
    api._write_json(output / "trainable_tokens.json", inventory)
    metadata = {
        "schema": api.EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": api.file_sha256(manifest_path),
        "source_metadata_sha256": api.file_sha256(dataset_root / "metadata.json"),
        "forward_projection": str(forward_root),
        "forward_projection_metadata_sha256": api.file_sha256(forward_metadata_path),
        "forward_projection_validation": forward_report,
        "image_root": str((dataset_root / "images").resolve()),
        "state_counts": {split: len(states_by_split[split]) for split in api.ALL_SPLITS},
        "inverse_rows_per_state": api.INVERSE_ROWS_PER_STATE,
        "mixed_rows_per_state": api.MIXED_ROWS_PER_STATE,
        "row_counts": row_counts,
        "mixed_row_counts": mixed_row_counts,
        "primary_inverse_row_count": sum(row_counts[split] for split in api.PRIMARY_SPLITS),
        "primary_mixed_row_count": sum(mixed_row_counts[split] for split in api.PRIMARY_SPLITS),
        "description_style_counts": dict(sorted(style_counts.items())),
        "target_token_counts": dict(sorted(target_counts.items())),
        "piece_qualifier_counts": dict(sorted(qualified_counts.items())),
        "inverse_grounding_contract": "inverse_grounding_contract.json",
        "inverse_grounding_contract_sha256": api.file_sha256(output / "inverse_grounding_contract.json"),
        "trainable_token_inventory": "trainable_tokens.json",
        "trainable_token_inventory_sha256": api.file_sha256(output / "trainable_tokens.json"),
        "files": files,
    }
    api._write_json(output / "metadata.json", metadata)
    report = api.validate_replay_v1_ms_swift_bidirectional(dataset_root, export_dir=output)
    report["output_dir"] = str(output)
    return report
