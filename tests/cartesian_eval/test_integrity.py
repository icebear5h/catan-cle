"""Source membership, immutable gold, and new-destination-only artifact guards."""
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from sft.cartesian_eval import score_response, validate_metadata, validate_rows
from sft.cartesian_eval.contracts import compact, object_list, object_map, parse_json, sha256, text
from sft.cartesian_eval.dataset import DEFAULT_SOURCE, build_dataset, build_rows


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return build_rows()


def test_gold_and_reference_cannot_be_rewritten(rows: list[dict[str, Any]]) -> None:
    metadata = object_map(rows[0]["metadata"])
    gold = text(metadata["answer"])
    wrong = "no" if gold == "yes" else "yes"
    with pytest.raises(ValueError, match="gold"):
        score_response(wrong, wrong, metadata)
    changed = deepcopy(metadata)
    changed["answer"] = wrong
    with pytest.raises(ValueError, match="gold"):
        score_response(wrong, wrong, changed)
    reference = object_map(metadata["atlas_reference"])
    atlas = object_map(reference["metadata"])
    changed_atlas = deepcopy(atlas)
    changed_atlas["answer"] = changed_atlas["canonical_answer"] = wrong
    # Even replacing the whole receipt and cached gold cannot replace the pinned case.
    changed["atlas_reference"] = {**reference, "metadata": changed_atlas}
    with pytest.raises(ValueError, match="immutable source"):
        validate_metadata(changed)
    returned = validate_metadata(metadata)
    returned["answer"] = "poisoned"
    assert object_map(score_response(gold, gold, metadata))["correct"] is True
    assert validate_metadata(metadata) == atlas


def test_panel_rejects_reordering_duplicates_prompt_and_mapping_changes(
    rows: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="200"):
        validate_rows(rows[:-1])
    for malformed in ([rows[1], rows[0], *rows[2:]], [rows[0], rows[0], *rows[2:]]):
        with pytest.raises(ValueError, match="order/duplicate"):
            validate_rows(malformed)
    changed = deepcopy(rows)
    messages = object_list(changed[0]["messages"])
    messages[0] = {"role": "user", "content": "Answer: no"}
    changed[0]["messages"] = messages
    with pytest.raises(ValueError, match="prompt/gold"):
        validate_rows(changed)
    metadata = object_map(rows[0]["metadata"])
    changed[0]["metadata"] = {**metadata, "mapping_sha256": "0" * 64}
    with pytest.raises(ValueError, match="metadata/gold"):
        validate_rows(changed)
    # Panel validation and scoring need no source-file reads on the remote runner.
    with patch.object(Path, "read_bytes", side_effect=AssertionError("source read")):
        assert validate_rows(rows)["valid"] is True


def test_builder_guards_precede_writes(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        build_dataset(destination)
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path / "missing" / "panel")
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("{}\n")
    destination = tmp_path / "panel"
    with pytest.raises(ValueError, match="source hash"):
        build_dataset(destination, source=corrupt)
    assert not destination.exists()
    assert len(sha256(DEFAULT_SOURCE.read_bytes())) == 64


def test_actual_manifest_when_present() -> None:
    # Artifact is built once after tests; the final targeted run exercises its receipts.
    destination = DEFAULT_SOURCE.parent.parent / "cartesian_eval_v2"
    if not destination.exists():
        pytest.skip("actual artifact is generated after initial tests")
    manifest = parse_json((destination / "manifest.json").read_text())
    rows = [parse_json(line) for line in (destination / "eval.jsonl").read_text().splitlines()]
    assert validate_rows(rows) == manifest["validation"]
    assert manifest["reference_source_ids"] == object_map(manifest["validation"])["source_ids"]
    assert manifest["intent"] == {"model": "Qwen3.8-27B", "weights": "stock", "finetuning": False,
                                   "inference_only": True, "model_run_performed": False}
    for name, receipt in object_map(manifest["files"]).items():
        assert sha256((destination / name).read_bytes()) == object_map(receipt)["sha256"]
    mapping = parse_json((destination / "mapping.json").read_text())
    assert sha256(compact(mapping).encode()) == manifest["mapping_sha256"]
    assert len(object_map(mapping["atlas_to_coordinates"])) == 154
    preview = (destination / "preview.md").read_text()
    assert preview.count("\n## symbolic_") == 8
