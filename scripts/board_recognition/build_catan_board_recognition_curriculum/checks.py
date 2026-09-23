"""Per-row, per-group, and split-file checks for the curriculum dataset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from scripts.board_recognition.build_catan_board_recognition_curriculum.jsonio import (
    canonical_json,
    file_sha256,
    read_json_object,
    read_jsonl,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.labels import (
    dense_label_differences,
    target_label_path,
    value_at_label_path,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.paths import (
    repository_relative,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import (
    SAMPLE_SCHEMA,
    JsonDict,
    integer,
    obj,
    text,
    values,
)

__all__ = ["check_group", "check_sample_row", "image_size", "validate_split_files"]


def image_size(row: JsonDict) -> tuple[int, int]:
    pair = values(row["image_size"], "image_size")
    if len(pair) != 2:
        raise ValueError("image_size must have two dimensions")
    return integer(pair[0], "image_size"), integer(pair[1], "image_size")


def check_sample_row(
    row: JsonDict,
    *,
    output_dir: Path,
    excluded_game_ids: set[str],
    style_path: Path,
) -> None:
    if row.get("schema") != SAMPLE_SCHEMA:
        raise ValueError(f"sample schema mismatch: {row.get('sample_id')}")
    render = obj(row["render"], "row render")
    if row.get("view") != "raw_full_board" or render.get("image_annotation") is not None:
        raise ValueError(f"non-raw image view in manifest: {row['sample_id']}")
    source = obj(row.get("source", {}), "row source")
    source_game_id = source.get("benchmark_game_id")
    if source_game_id is not None and str(source_game_id) in excluded_game_ids:
        raise ValueError(f"benchmark game leaked into curriculum: {source_game_id}")
    if source.get("kind") != "synthetic_engine_contract":
        raise ValueError(f"unsupported or unknown source provenance: {row['sample_id']}")
    if source_game_id is not None or not isinstance(source.get("seed"), int):
        raise ValueError(f"synthetic provenance is not fail-closed: {row['sample_id']}")
    counterfactual = obj(row["counterfactual"], "row counterfactual")
    if (
        source.get("curriculum_stage") != row["stage"]
        or source.get("counterfactual_group_id") != row["counterfactual_group_id"]
        or counterfactual.get("group_id") != row["counterfactual_group_id"]
    ):
        raise ValueError(f"state provenance/group mismatch: {row['sample_id']}")
    if render != {
        "renderer": "evals.catan_board_bench.render",
        "style_config": repository_relative(style_path),
        "image_annotation": None,
    }:
        raise ValueError(f"render policy mismatch: {row['sample_id']}")
    digests = obj(row["sha256"], "row sha256")
    for kind, path_key in (
        ("contract", "contract_path"),
        ("labels", "label_path"),
        ("image", "image_path"),
    ):
        artifact_path = output_dir / text(row[path_key], path_key)
        if not artifact_path.is_file():
            raise ValueError(f"missing {kind}: {artifact_path}")
        if digests[kind] != file_sha256(artifact_path):
            raise ValueError(f"{kind} hash mismatch: {artifact_path}")


def check_group(
    group_id: str,
    pair: list[JsonDict],
    *,
    output_dir: Path,
    target_counts: Counter[str],
) -> None:
    if len(pair) != 2 or {row["counterfactual_role"] for row in pair} != {
        "base",
        "counterfactual",
    }:
        raise ValueError(f"counterfactual group must contain one pair: {group_id}")
    if len({text(row["split"], "split") for row in pair}) != 1 or len(
        {text(row["stage"], "stage") for row in pair}
    ) != 1:
        raise ValueError(f"counterfactual group crossed split or stage: {group_id}")
    descriptors = {canonical_json(row["counterfactual"]) for row in pair}
    if len(descriptors) != 1:
        raise ValueError(f"counterfactual descriptor mismatch: {group_id}")
    base = next(row for row in pair if row["counterfactual_role"] == "base")
    changed = next(row for row in pair if row["counterfactual_role"] == "counterfactual")
    base_labels = read_json_object(output_dir / text(base["label_path"], "label_path"))
    changed_labels = read_json_object(output_dir / text(changed["label_path"], "label_path"))
    base_counterfactual = obj(base["counterfactual"], "counterfactual")
    target = obj(base_counterfactual["target"], "counterfactual target")
    differences = dense_label_differences(
        obj(base_labels["entities"], "label entities"),
        obj(changed_labels["entities"], "label entities"),
    )
    expected_path = target_label_path(target)
    if differences != [expected_path]:
        raise ValueError(
            f"counterfactual {group_id} changed {differences}, expected {[expected_path]}"
        )
    before = value_at_label_path(obj(base_labels["entities"], "label entities"), expected_path)
    after = value_at_label_path(obj(changed_labels["entities"], "label entities"), expected_path)
    if before != base_counterfactual["before"] or after != base_counterfactual["after"]:
        raise ValueError(f"counterfactual values disagree with labels: {group_id}")
    if obj(base["sha256"], "sha256")["image"] == obj(changed["sha256"], "sha256")["image"]:
        raise ValueError(f"counterfactual has no visible pixel change: {group_id}")
    target_counts[text(target["entity_type"], "target entity_type")] += 1


def validate_split_files(output_dir: Path, rows: Sequence[JsonDict]) -> None:
    expected = {
        split: sorted(
            text(row["sample_id"], "sample_id") for row in rows if row["split"] == split
        )
        for split in ("train", "validation", "test")
    }
    for split, sample_ids in expected.items():
        split_rows = read_jsonl(output_dir / "splits" / f"{split}.jsonl")
        if sorted(text(row["sample_id"], "sample_id") for row in split_rows) != sample_ids:
            raise ValueError(f"split manifest is stale: {split}")
        expected_rows = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: text(row["sample_id"], "sample_id"),
        )
        if sorted(split_rows, key=lambda row: text(row["sample_id"], "sample_id")) != expected_rows:
            raise ValueError(f"split manifest rows disagree with the main manifest: {split}")
