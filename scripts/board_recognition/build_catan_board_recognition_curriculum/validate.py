"""Fail-closed validation of a generated curriculum dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from cle.game_engine.models.enums import CITY, SETTLEMENT
from evals.catan_board_bench.render import render_contract_image
from evals.catan_board_bench.tokens import atlas_metadata_json
from scripts.board_recognition.build_catan_board_recognition_curriculum.checks import (
    check_group,
    check_sample_row,
    image_size,
    validate_split_files,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.jsonio import (
    file_sha256,
    load_leakage_ledger,
    read_json_object,
    read_jsonl,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.labels import (
    dense_labels,
    expected_entity_ids,
    validate_dense_labels,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.metadata import (
    collect_class_counts,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.paths import (
    DEFAULT_LEAKAGE_LEDGER,
    DEFAULT_SPEC_PATH,
    resolve_project_path,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import (
    DATASET_SCHEMA,
    ENTITY_TYPES,
    JsonDict,
    integer,
    obj,
    objs,
    strings,
    text,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.spec import (
    load_render_style,
    load_spec,
)

__all__ = ["validate_dataset", "validate_split_files"]


def validate_dataset(output_dir: Path, *, spec_path: Path = DEFAULT_SPEC_PATH) -> JsonDict:
    spec = load_spec(spec_path)
    metadata_path = output_dir / "metadata.json"
    manifest_path = output_dir / "manifest.jsonl"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"dataset is incomplete: {output_dir}")
    metadata = read_json_object(metadata_path)
    if metadata.get("schema") != DATASET_SCHEMA:
        raise ValueError("dataset metadata schema mismatch")
    if metadata.get("curriculum_sha256") != file_sha256(spec_path):
        raise ValueError("curriculum specification changed")
    if obj(metadata["leakage"], "metadata leakage")["ledger_sha256"] != file_sha256(
        DEFAULT_LEAKAGE_LEDGER
    ):
        raise ValueError("benchmark leakage ledger changed")
    _ledger, excluded_game_ids = load_leakage_ledger(DEFAULT_LEAKAGE_LEDGER)
    rows = read_jsonl(manifest_path)
    if metadata.get("sample_count") != len(rows):
        raise ValueError("metadata sample count is stale")
    if len(rows) != len({text(row["sample_id"], "sample_id") for row in rows}):
        raise ValueError("manifest sample IDs are not unique")
    groups: dict[str, list[JsonDict]] = {}
    target_counts: Counter[str] = Counter()
    atlas = atlas_metadata_json()
    expected_ids = expected_entity_ids(atlas)
    vocabularies = obj(spec["class_vocabularies"], "class_vocabularies")
    colors = tuple(strings(vocabularies["colors"], "colors"))
    style_path = resolve_project_path(
        text(obj(spec["render"], "spec render")["style_config"], "style_config")
    )
    style = load_render_style(style_path)
    node_classes = {
        "EMPTY",
        *[f"{color}_{building}" for color in colors for building in (SETTLEMENT, CITY)],
    }
    edge_classes = {"EMPTY", *colors}
    stages_by_id = {
        text(stage["id"], "stage id"): stage for stage in objs(spec["stages"], "spec stages")
    }

    for row in rows:
        check_sample_row(
            row,
            output_dir=output_dir,
            excluded_game_ids=excluded_game_ids,
            style_path=style_path,
        )
        contract = read_json_object(output_dir / text(row["contract_path"], "contract_path"))
        labels = read_json_object(output_dir / text(row["label_path"], "label_path"))
        if obj(contract.get("sample", {}), "contract sample").get("id") != row["sample_id"]:
            raise ValueError(f"contract identity mismatch: {row['sample_id']}")
        if row["counterfactual_role"] == "base":
            stage = stages_by_id[text(row["stage"], "row stage")]
            expected_buildings = integer(
                stage["distractor_buildings"], "stage distractor_buildings"
            ) + int(stage["target_occupancy"] != "EMPTY")
            actual_buildings = sum(
                node["building"] is not None
                for node in objs(contract["nodes"], "contract nodes")
            )
            actual_roads = sum(
                edge["road_color"] is not None
                for edge in objs(contract["edges"], "contract edges")
            )
            if (actual_buildings, actual_roads) != (
                expected_buildings,
                integer(stage["roads"], "stage roads"),
            ):
                raise ValueError(f"base state does not match curriculum stage: {row['sample_id']}")
        if labels != dense_labels(contract, stage_id=text(row["stage"], "row stage")):
            raise ValueError(f"dense labels disagree with engine contract: {row['sample_id']}")
        size = image_size(row)
        with Image.open(output_dir / text(row["image_path"], "image_path")) as actual_image:
            actual_rgb = actual_image.convert("RGB")
            expected_image = render_contract_image(
                contract,
                image_size=size[0],
                style=style,
            )
            if actual_rgb.size != size or actual_rgb.tobytes() != expected_image.tobytes():
                raise ValueError(f"image disagrees with engine contract: {row['sample_id']}")
        validate_dense_labels(
            labels,
            expected_ids=expected_ids,
            node_classes=node_classes,
            edge_classes=edge_classes,
        )
        if labels["sample_id"] != row["sample_id"] or labels["stage"] != row["stage"]:
            raise ValueError(f"label identity mismatch: {row['sample_id']}")
        groups.setdefault(
            text(row["counterfactual_group_id"], "counterfactual_group_id"), []
        ).append(row)

    for group_id, pair in groups.items():
        check_group(group_id, pair, output_dir=output_dir, target_counts=target_counts)

    if set(target_counts) != set(ENTITY_TYPES) or len(set(target_counts.values())) != 1:
        raise ValueError(f"counterfactual entity types are not balanced: {dict(target_counts)}")
    expected_stage_counts = {
        key: count
        for key, count in sorted(Counter(text(row["stage"], "stage") for row in rows).items())
    }
    expected_split_counts = {
        key: count
        for key, count in sorted(Counter(text(row["split"], "split") for row in rows).items())
    }
    if metadata.get("stage_counts") != expected_stage_counts:
        raise ValueError("metadata stage counts are stale")
    if metadata.get("split_counts") != expected_split_counts:
        raise ValueError("metadata split counts are stale")
    if metadata.get("counterfactual_target_counts") != {
        key: count for key, count in sorted(target_counts.items())
    }:
        raise ValueError("metadata counterfactual target counts are stale")
    if metadata.get("class_counts") != collect_class_counts(output_dir, rows):
        raise ValueError("metadata class counts are stale")
    validate_split_files(output_dir, rows)
    return {
        "valid": True,
        "samples": len(rows),
        "counterfactual_groups": len(groups),
        "splits": {
            key: count
            for key, count in sorted(Counter(text(row["split"], "split") for row in rows).items())
        },
        "stages": {
            key: count
            for key, count in sorted(Counter(text(row["stage"], "stage") for row in rows).items())
        },
        "targets": {key: count for key, count in sorted(target_counts.items())},
        "entity_counts_per_sample": {"tiles": 19, "nodes": 54, "edges": 72, "ports": 9},
    }


