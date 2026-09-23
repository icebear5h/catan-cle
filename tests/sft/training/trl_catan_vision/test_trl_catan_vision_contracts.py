"""Curriculum, image escape, and message pair contracts."""

import json
from pathlib import Path

import pytest

from sft.scripts.train.train_trl_catan_vision import (
    CURRICULUM_STAGES,
    _message_pair,
    inspect_jsonl_contract,
    validate_spatial_targets,
)

from .support import _row, _write_dataset


def test_contract_accepts_all_four_ordered_curriculum_stages(tmp_path: Path) -> None:
    train_jsonl, image_root = _write_dataset(tmp_path, list(CURRICULUM_STAGES))

    report = inspect_jsonl_contract(train_jsonl, image_root, require_curriculum=True)

    assert report["rows"] == 4
    assert [stage["name"] for stage in report["stages"]] == list(CURRICULUM_STAGES)
    assert [stage["start_row"] for stage in report["stages"]] == [0, 1, 2, 3]


def test_contract_rejects_missing_and_regressing_curriculum(tmp_path: Path) -> None:
    train_jsonl, image_root = _write_dataset(tmp_path / "missing", [None])
    with pytest.raises(ValueError, match="missing curriculum_stage"):
        inspect_jsonl_contract(train_jsonl, image_root, require_curriculum=True)

    train_jsonl, image_root = _write_dataset(
        tmp_path / "regression",
        [CURRICULUM_STAGES[1], CURRICULUM_STAGES[0]],
    )
    with pytest.raises(ValueError, match="order regressed"):
        inspect_jsonl_contract(train_jsonl, image_root, require_curriculum=True)


def test_contract_rejects_image_escape(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(json.dumps(_row("../outside.png", CURRICULUM_STAGES[0])) + "\n")

    with pytest.raises(ValueError, match="escapes image_root"):
        inspect_jsonl_contract(train_jsonl, image_root, require_curriculum=False)


def test_spatial_target_contract_rejects_bad_or_unnormalized_boxes() -> None:
    target = {
        "token": "<N00>",
        "entity_type": "node",
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "center": [0.2, 0.3],
        "control_bbox": [0.5, 0.6, 0.7, 0.8],
    }
    assert validate_spatial_targets({"spatial_targets": [target]}, line_number=1) == [target]

    bad = {**target, "bbox": [-0.1, 0.2, 0.3, 0.4]}
    with pytest.raises(ValueError, match="must be normalized"):
        validate_spatial_targets({"spatial_targets": [bad]}, line_number=2)


def test_message_pair_normalizes_sharegpt_and_short_answer() -> None:
    row = {
        "conversations": [
            {"from": "human", "value": "<image>\nWhere is the ore-wheat-sheep node?"},
            {"from": "gpt", "value": "<N17>"},
        ]
    }

    assert _message_pair(row, line_number=1) == (
        "Where is the ore-wheat-sheep node?",
        "<N17>",
    )


def test_message_pair_admits_full_board_readouts_and_rejects_runaway_answers() -> None:
    readout = "; ".join(f"<E{i:02d}_{i + 1:02d}> mystic blue road" for i in range(72))
    assert len(readout) > 512
    row = {"messages": [{"role": "user", "content": "<image>\nList every edge."}, {"role": "assistant", "content": readout}]}
    assert _message_pair(row, line_number=1) == ("List every edge.", readout)
    runaway = {"messages": [{"role": "user", "content": "<image>\nx"}, {"role": "assistant", "content": "a" * 2049}]}
    with pytest.raises(ValueError, match="answer-length contract"):
        _message_pair(runaway, line_number=2)
