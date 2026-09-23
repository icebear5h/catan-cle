"""Small CPU tests for admission, provenance and native completion boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sft.miles_sft.data import (
    STATIC_OPERATIONS,
    ChatMessage,
    encode_pair,
    inspect_corpus,
    prepare_dataset,
)


class NativeTokenizer:
    """Deterministic tokenizer double; IDs 1/2/3 are native chat controls."""

    chat_template = "test-native-template"
    name_or_path = "saved-test-tokenizer"
    all_special_ids = [1, 2, 3]

    def __init__(self, fault: str = "") -> None:
        self.fault = fault

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == "<|im_end|>":
            return [2]
        if text == "[MASK]":
            return [3]
        return [ord(c) + 1000 for c in text]

    def apply_chat_template(
        self, conversation: list[ChatMessage], *, tokenize: bool,
        add_generation_prompt: bool, return_dict: bool, enable_thinking: bool,
        preserve_thinking: bool, truncation: bool,
    ) -> list[int]:
        assert tokenize is True
        assert (return_dict, enable_thinking, preserve_thinking, truncation) == (False,) * 4
        prefix = [1] + self.encode(conversation[0]["content"], add_special_tokens=False) + [2, 3]
        if add_generation_prompt:
            assert len(conversation) == 1
            return prefix
        assert len(conversation) == 2
        answer = self.encode(conversation[1]["content"], add_special_tokens=False)
        ending = [2] + self.encode("\n", add_special_tokens=False)
        if self.fault == "prefix":
            prefix = prefix[:-1]
        elif self.fault == "missing_eot":
            ending = []
        elif self.fault == "double_eot":
            ending = [2, 2]
        elif self.fault == "answer":
            answer = answer[:-1]
        elif self.fault == "trailing":
            ending += [1]
        return prefix + answer + ending


def source_row(operation: str = "symbolic_neighbors", row_id: str = "r0") -> dict[str, object]:
    return {
        "id": row_id, "row_id": row_id, "split": "train", "task_type": operation,
        "messages": [
            {"role": "user", "content": "List all edge-connected node neighbors of <N11>."},
            {"role": "assistant", "content": "<N10> <N12> <N32>"},
        ],
        "metadata": {"split": "train", "task_type": operation, "task_role": "train"},
    }


def write_source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return source


def test_filter_order_hashes_limit_and_receipt(tmp_path: Path) -> None:
    operations = [
        "symbolic_direction", "symbolic_owned_incident_roads", "symbolic_direction_choice",
        "symbolic_reachable", "symbolic_shortest_route", "symbolic_neighbors",
        "node_pip_sum", "symbolic_incidence", "counterfactual", "symbolic_oriented_step",
        "tile_nodes", "port_direction",
    ]
    source = write_source(tmp_path, [source_row(op, str(i)) for i, op in enumerate(operations)])
    output = tmp_path / "prepared"
    receipt = prepare_dataset(source, output, NativeTokenizer(), limit=4)
    rows = [json.loads(line) for line in (output / "input.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["0", "2", "5", "7"]
    assert receipt["rows"] == 4 and receipt["limited_rows"] == 1
    assert receipt["corpus"]["admitted_rows"] == 5
    assert receipt["corpus"]["excluded_by_reason"] == {"operation_not_allowed": 7}
    assert set(receipt["corpus"]["admitted_by_operation"]) == STATIC_OPERATIONS
    assert receipt["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["input_sha256"] == hashlib.sha256((output / "input.jsonl").read_bytes()).hexdigest()
    assert json.loads((output / "metadata.json").read_text()) == receipt
    for row, index in zip(rows, (0, 2, 5, 7), strict=True):
        metadata = row["metadata"]
        assert row["row_id"] == row["id"]
        assert metadata["source_file"] == str(source.resolve())
        assert metadata["source_line"] == index + 1
        assert metadata["source_row_sha256"] == hashlib.sha256(source.read_bytes().splitlines()[index]).hexdigest()
        prefix = metadata["prefix_length"]
        suffix = metadata["tokens"][prefix:]
        assert suffix == NativeTokenizer().encode(row["messages"][1]["content"], add_special_tokens=False) + [2, 1010]
        assert metadata["loss_mask"] == [1] * len(suffix)
        assert metadata["response_length"] == len(suffix)
    with pytest.raises(FileExistsError, match="fresh output"):
        prepare_dataset(source, output, NativeTokenizer())


@pytest.mark.parametrize("split", [None, "test", "validation", "transfer_test"])
def test_requires_explicit_train_split(tmp_path: Path, split: str | None) -> None:
    row = source_row()
    row["metadata"] = {}
    if split is None:
        row.pop("split")
    else:
        row["split"] = split
    source = write_source(tmp_path, [row])
    report = inspect_corpus(source)
    reason = "missing_train_split" if split is None else "non_train_split"
    assert report["excluded_by_reason"] == {reason: 1}
    with pytest.raises(ValueError, match="no admitted train rows"):
        prepare_dataset(source, tmp_path / "empty", NativeTokenizer())
    assert not (tmp_path / "empty").exists()


@pytest.mark.parametrize("fault", ["prefix", "missing_eot", "double_eot", "answer", "trailing"])
def test_rejects_native_boundary_and_termination_faults(tmp_path: Path, fault: str) -> None:
    source = write_source(tmp_path, [source_row()])
    with pytest.raises(ValueError):
        prepare_dataset(source, tmp_path / "invalid", NativeTokenizer(fault))
    assert not (tmp_path / "invalid").exists()


def test_exact_budget_and_text_parts(tmp_path: Path) -> None:
    messages = [{"role": "user", "content": "Neighbors of <N11>?"},
                {"role": "assistant", "content": "<N10> <N12> <N32>"}]
    encoded = encode_pair(NativeTokenizer(), messages)
    length = len(encoded["tokens"])
    assert encode_pair(NativeTokenizer(), messages, length) == encoded
    with pytest.raises(ValueError, match="truncation forbidden"):
        encode_pair(NativeTokenizer(), messages, length - 1)
    row = source_row()
    row["messages"] = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                       for m in messages]
    source = write_source(tmp_path, [row])
    receipt = prepare_dataset(source, tmp_path / "parts", NativeTokenizer())
    assert receipt["longest_sequence"] == length
    with pytest.raises(ValueError, match="truncation forbidden"):
        prepare_dataset(source, tmp_path / "overlong", NativeTokenizer(), max_tokens=length - 1)
    assert not (tmp_path / "overlong").exists()


@pytest.mark.parametrize("content", [
    "<image>", "<|im_start|>assistant", "<think>x</think>", "[INST]x", "[MASK]",
    "<audio src='x'>", "bad\x00text", "bad\u202etext",
    [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
    [{"type": "text", "text": [{"type": "text", "text": "nested"}]}],
])
def test_rejects_media_and_controls(tmp_path: Path, content: object) -> None:
    row = source_row()
    row["messages"] = [{"role": "user", "content": content}, {"role": "assistant", "content": "no"}]
    source = write_source(tmp_path, [row])
    with pytest.raises(ValueError):
        prepare_dataset(source, tmp_path / "media", NativeTokenizer())
    assert not (tmp_path / "media").exists()


def test_rejects_ambiguous_metadata_duplicate_ids_and_nested_media(tmp_path: Path) -> None:
    for metadata in ({"split": "test"}, {"task_type": "symbolic_reachable"},
                     {"extra": {"images": []}}, {"extra": {"image_paths": []}},
                     {"target": {"state": None, "query": {"token": "<N11>", "color": "RED"}}}):
        row = source_row()
        row["metadata"] = metadata
        with pytest.raises(ValueError):
            inspect_corpus(write_source(tmp_path, [row]))
    with pytest.raises(ValueError, match="duplicate"):
        inspect_corpus(write_source(tmp_path, [source_row(), source_row()]))
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"x","id":"y"}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        inspect_corpus(source)


def test_state_and_training_provenance_cannot_bypass_allowlist(tmp_path: Path) -> None:
    rows = [source_row(row_id=str(i)) for i in range(4)]
    rows[0]["metadata"] = {"target": {"state": {"board": "occupied"}, "query": {}}}
    rows[1]["metadata"] = {"review_only": True}
    rows[2]["metadata"] = {"admitted_for_training": False}
    rows[3]["provenance"] = {"split": "test"}
    report = inspect_corpus(write_source(tmp_path, rows))
    assert report["admitted_rows"] == 0
    assert report["excluded_by_reason"] == {
        "state_conditioned": 1, "non_training_role": 2, "non_train_provenance": 1,
    }


def test_real_generated_corpus_labels() -> None:
    root = Path(__file__).resolve().parents[2] / "artifacts/generated/sft"
    source = root / "symbolic_board_v2/train.jsonl"
    if not source.is_file():
        pytest.skip("locally generated symbolic_board_v2 corpus is not present")
    report = inspect_corpus(source)
    assert report["total_rows"] == 3200
    assert report["admitted_by_operation"] == {
        "symbolic_direction": 400, "symbolic_direction_choice": 400,
        "symbolic_neighbors": 240, "symbolic_incidence": 400, "symbolic_oriented_step": 160,
    }
    assert report["excluded_rows"] == 1600
    assert report["excluded_by_reason"] == {"operation_not_allowed": 1600}
    assert report["excluded_by_operation"]["symbolic_reachable"] == 160
    assert report["excluded_by_operation"]["symbolic_shortest_route"] == 240
    atlas = root / "atlas_topology/catan_atlas_topology.jsonl"
    if atlas.is_file():
        legacy = inspect_corpus(atlas)
        assert legacy["admitted_rows"] == 0
        assert legacy["excluded_by_reason"] == {"missing_train_split": 390}
        assert legacy["excluded_by_operation"]["port_direction"] == 9
