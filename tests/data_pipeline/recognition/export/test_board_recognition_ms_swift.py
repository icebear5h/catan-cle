import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from data_pipeline.board_recognition.replay_ms_swift import (
    SemanticExportError,
    ms_swift_row,
    prepare_output_dir,
    semantic_audit_row,
    validate_ms_swift_row,
)
from data_pipeline.board_recognition.semantics import semantic_candidates


def query(**overrides: object) -> dict:
    row = {
        "query_id": "state_e000_q000",
        "entity_type": "node",
        "entity_type_id": 1,
        "attribute": "occupancy",
        "attribute_id": 3,
        "head": "node.occupancy",
        "slot": "<N17>",
        "slot_index": 17,
        "class_name": "MYSTIC_BLUE_CITY",
        "class_index": 22,
        "class_vocabulary": ["EMPTY", "MYSTIC_BLUE_CITY"],
        "query_token": "<Q_NODE_OCCUPANCY>",
        "answer_token": "<A_NODE_OCCUPANCY_MYSTIC_BLUE_CITY>",
    }
    row.update(overrides)
    return row


def plan() -> dict:
    return {
        "state_id": "state",
        "split": "train",
        "epoch": 0,
        "queries": [query()],
    }


def state() -> dict:
    return {
        "image_path": "images/state.png",
        "label_path": "dense_labels/state.json",
        "sha256": {"image": "a" * 64},
        "source": {
            "kind": "engine_rollout",
            "game_id": None,
            "trajectory_id": "engine:train:0",
        },
        "density_bin": "sparse",
        "building_count": 9,
        "road_count": 12,
    }


def test_ms_swift_row_uses_standard_messages_images_and_semantic_text() -> None:
    row = ms_swift_row(query(), image_name="state.png")

    validate_ms_swift_row(row)
    assert row == {
        "messages": [
            {"role": "user", "content": "<image>\n<N17> building?"},
            {"role": "assistant", "content": "mystic blue city"},
        ],
        "images": ["state.png"],
    }
    assert "<Q_" not in json.dumps(row)
    assert "<A_" not in json.dumps(row)


def test_semantic_audit_preserves_source_tokens_only_as_provenance() -> None:
    audit: Any = semantic_audit_row(plan(), query(), state=state())

    assert audit["semantic_answer"] == "mystic blue city"
    assert audit["candidate_answers"] == list(semantic_candidates("node.occupancy"))
    assert audit["piece_count"] == 21
    assert audit["source_query_token"] == "<Q_NODE_OCCUPANCY>"
    assert audit["source_answer_token"] == "<A_NODE_OCCUPANCY_MYSTIC_BLUE_CITY>"
    assert len(audit["prompt_sha256"]) == len(audit["answer_sha256"]) == 64


def test_semantic_rows_and_audits_pass_draft_2020_12_schemas() -> None:
    schema_root = Path("data/curriculum/board_recognition/schemas")
    row_schema = json.loads((schema_root / "ms_swift_semantic_v1.schema.json").read_text())
    audit_schema = json.loads(
        (schema_root / "ms_swift_semantic_audit_v1.schema.json").read_text()
    )

    Draft202012Validator(row_schema).validate(ms_swift_row(query(), image_name="state.png"))
    Draft202012Validator(audit_schema).validate(semantic_audit_row(plan(), query(), state=state()))


def test_semantic_row_rejects_opaque_targets_and_escaping_images() -> None:
    opaque: Any = ms_swift_row(query(), image_name="state.png")
    opaque["messages"][1]["content"] = "<A_NODE_OCCUPANCY_EMPTY>"
    with pytest.raises(SemanticExportError, match="ordinary language"):
        validate_ms_swift_row(opaque)

    escaping = ms_swift_row(query(), image_name="state.png")
    escaping["images"] = ["../state.png"]
    with pytest.raises(SemanticExportError, match="relative to image_root"):
        validate_ms_swift_row(escaping)


def test_semantic_export_refuses_historical_projection_path(tmp_path: Path) -> None:
    historical = tmp_path / "qwen_sft"

    with pytest.raises(SemanticExportError, match="historical qwen_sft"):
        prepare_output_dir(historical, dataset_root=tmp_path, overwrite=True)
