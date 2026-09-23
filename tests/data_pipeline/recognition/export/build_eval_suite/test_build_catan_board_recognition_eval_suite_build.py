"""Suite build rows, alignment, and writer policy."""
from pathlib import Path
from typing import Any

import pytest

from scripts.board_recognition.build_catan_board_recognition_eval_suite import (
    build_rows,
    compact_metadata,
    write_jsonl,
)
from sft.analysis.behavior_diagnostics import records_fingerprint

from .support import CORRECTED_SOURCE, JsonRow


def test_spatial_only_uses_explicit_source_without_base_and_preserves_rows(tmp_path: Path, supplement: tuple[Path, list[JsonRow], list[JsonRow]]) -> None:
    source, rows, audits = supplement
    missing_root = tmp_path / "no-base-dataset"
    result = build_rows(missing_root, "validation", supplement_dir=source, spatial_only=True)

    assert not missing_root.exists()
    assert len(result) == 2
    assert result == build_rows(
        missing_root, "validation", supplement_dir=source, spatial_only=True
    )
    for enriched, index, entity in zip(result, [0, 2], ["node", "tile"], strict=True):
        assert enriched["images"] == rows[index]["images"]
        assert enriched["messages"] == rows[index]["messages"]
        assert enriched["id"] == f"{CORRECTED_SOURCE}:validation:{audits[index]['query_id']}"
        assert enriched["metadata"] == {
            **compact_metadata(audits[index], suite="spatial_robber", category="spatial_grounding"),
            "entity_type": entity,
            "source_supplement": str(source.resolve()),
            "source_split": "validation",
        }
    assert len({row["id"] for row in result}) == 2
    legacy = [{**row, "id": row["metadata"]["query_id"]} for row in result]
    assert records_fingerprint(result) != records_fingerprint(legacy)


def test_default_build_keeps_base_and_robber_rows_and_legacy_ids(tmp_path: Path, supplement: tuple[Path, list[JsonRow], list[JsonRow]]) -> None:
    source, rows, audits = supplement
    base: Any = {
        "images": ["base.png"],
        "messages": [
            {"role": "user", "content": "<image>\nWhat resource is on <T00>?"},
            {"role": "assistant", "content": "WOOD"},
        ],
    }
    audit: Any = {
        "query_id": "base-question",
        "entity_type": "tile",
        "head": "tile.resource",
        "semantic_prompt": "What resource is on <T00>?",
        "semantic_answer": "WOOD",
    }
    write_jsonl(tmp_path / "ms_swift_bidirectional_v1/mixed/validation.jsonl", [base])
    write_jsonl(
        tmp_path / "ms_swift_bidirectional_v1/mixed_index/validation.jsonl",
        [{"mixed_index": 0, "row_kind": "forward", "query_id": audit["query_id"]}],
    )
    write_jsonl(tmp_path / "ms_swift_semantic_v1/audit/validation.jsonl", [audit])
    write_jsonl(tmp_path / "ms_swift_bidirectional_v1/audit/validation.jsonl", [])
    write_jsonl(tmp_path / "spatial_robber_v1/validation.jsonl", rows)
    write_jsonl(tmp_path / "spatial_robber_v1/audit/validation.jsonl", audits)

    result: Any = build_rows(tmp_path, "validation")
    assert [row["id"] for row in result] == [audit["query_id"], *[a["query_id"] for a in audits]]
    assert result[0] == {
        **base,
        "id": audit["query_id"],
        "metadata": {
            **compact_metadata(audit, suite="bidirectional", category="tile.resource"),
            "row_kind": "forward",
        },
    }
    assert [row["messages"] for row in result[1:]] == [row["messages"] for row in rows]
    explicit: Any = build_rows(tmp_path, "validation", supplement_dir=source)
    assert len(explicit) == 4
    assert explicit[0] == result[0]
    assert explicit[2]["metadata"]["task_family"] == "robber"
    narrow: Any = build_rows(tmp_path, "validation", spatial_only=True)
    assert len(narrow) == 2
    assert all(row["id"].startswith("spatial_robber_v1:validation:") for row in narrow)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("prompt", "Wrong question", "prompt mismatch"),
        ("answer", "yes", "answer mismatch"),
        ("image_name", "wrong.png", "image mismatch"),
        ("split", "train", "split mismatch"),
        ("task_type", "edge_direction_token", "unknown spatial entity"),
    ],
)
def test_spatial_alignment_failures(tmp_path: Path, supplement: tuple[Path, list[JsonRow], list[JsonRow]], field: str, value: str, error: str) -> None:
    source, _, audits = supplement
    audits[0][field] = value
    write_jsonl(source / "audit/validation.jsonl", audits, overwrite=True)
    with pytest.raises(ValueError, match=error):
        build_rows(tmp_path, "validation", supplement_dir=source, spatial_only=True)


@pytest.mark.parametrize("duplicate", [False, True])
def test_spatial_rejects_length_mismatch_and_duplicate_query_ids(tmp_path: Path, supplement: tuple[Path, list[JsonRow], list[JsonRow]], duplicate: bool) -> None:
    source, rows, audits = supplement
    if duplicate:
        rows.append(rows[0])
        audits.append(audits[0])
    else:
        audits.pop()
    write_jsonl(source / "validation.jsonl", rows, overwrite=True)
    write_jsonl(source / "audit/validation.jsonl", audits, overwrite=True)
    with pytest.raises(
        ValueError, match="duplicate query_id" if duplicate else "different lengths"
    ):
        build_rows(tmp_path, "validation", supplement_dir=source, spatial_only=True)


def test_writer_is_deterministic_and_requires_explicit_overwrite(tmp_path: Path) -> None:
    first: Any = tmp_path / "first.jsonl"
    second: Any = tmp_path / "second.jsonl"
    rows: Any = [{"id": "one", "metadata": {"category": "spatial_grounding"}}]
    assert write_jsonl(first, iter(rows)) == write_jsonl(second, rows) == 1
    original = first.read_bytes()
    assert original == second.read_bytes()
    with pytest.raises(FileExistsError):
        write_jsonl(first, [])
    assert first.read_bytes() == original
    assert write_jsonl(first, [], overwrite=True) == 0
    assert first.read_bytes() == b""
