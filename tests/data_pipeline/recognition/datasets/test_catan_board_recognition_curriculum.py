import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.board_recognition.build_catan_board_recognition_curriculum import (
    DEFAULT_SPEC_PATH,
    build_dataset,
    dense_label_differences,
    file_sha256,
    load_spec,
    read_jsonl,
    validate_dataset,
)


def test_curriculum_spec_uses_dense_raw_board_contract() -> None:
    spec: Any = load_spec(DEFAULT_SPEC_PATH)

    assert spec["views"] == ["raw_full_board"]
    assert spec["render"]["image_size"] == 1024
    assert spec["render"]["image_annotation"] is None
    assert not spec["render"]["allow_crops"]
    assert not spec["render"]["allow_shifts"]
    assert not spec["render"]["allow_overlays"]
    assert spec["topology"] == {"tiles": 19, "nodes": 54, "edges": 72, "ports": 9}
    assert spec["sampling"]["entity_type_order"] == ["tile", "node", "edge", "port"]
    assert [stage["id"] for stage in spec["stages"]] == [
        "empty_setup",
        "initial_placements",
        "sparse_midgame",
        "dense_endgame",
    ]


def test_builder_emits_balanced_dense_counterfactual_pilot(tmp_path: Path) -> None:
    output_dir: Any = tmp_path / "curriculum"

    report = build_dataset(
        output_dir=output_dir,
        image_size=64,
        pairs_per_entity_type=1,
        seed=1234,
    )

    assert report == {
        "valid": True,
        "samples": 32,
        "counterfactual_groups": 16,
        "splits": {"test": 2, "train": 28, "validation": 2},
        "stages": {
            "dense_endgame": 8,
            "empty_setup": 8,
            "initial_placements": 8,
            "sparse_midgame": 8,
        },
        "targets": {"edge": 4, "node": 4, "port": 4, "tile": 4},
        "entity_counts_per_sample": {"tiles": 19, "nodes": 54, "edges": 72, "ports": 9},
        "output_dir": str(output_dir),
    }
    rows: Any = read_jsonl(output_dir / "manifest.jsonl")
    assert len(rows) == len({row["sample_id"] for row in rows}) == 32
    assert {row["view"] for row in rows} == {"raw_full_board"}
    assert {row["render"]["image_annotation"] for row in rows} == {None}
    assert {row["source"]["benchmark_game_id"] for row in rows} == {None}

    for row in rows:
        labels: Any = json.loads((output_dir / row["label_path"]).read_text())
        assert len(labels["entities"]["tiles"]) == 19
        assert len(labels["entities"]["nodes"]) == 54
        assert len(labels["entities"]["edges"]) == 72
        assert len(labels["entities"]["ports"]) == 9
        with Image.open(output_dir / row["image_path"]) as image:
            assert image.size == (64, 64)

    base_piece_counts: Any = {}
    for row in rows:
        if row["counterfactual_role"] != "base":
            continue
        contract: Any = json.loads((output_dir / row["contract_path"]).read_text())
        base_piece_counts.setdefault(
            row["stage"],
            (
                sum(node["building"] is not None for node in contract["nodes"]),
                sum(edge["road_color"] is not None for edge in contract["edges"]),
            ),
        )
    assert base_piece_counts == {
        "empty_setup": (0, 0),
        "initial_placements": (4, 4),
        "sparse_midgame": (9, 12),
        "dense_endgame": (15, 28),
    }


def test_build_is_deterministic_except_metadata_timestamp(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output_dir in (first, second):
        build_dataset(
            output_dir=output_dir,
            image_size=64,
            pairs_per_entity_type=1,
            seed=9876,
        )

    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    for relative_path in first_files:
        if relative_path == Path("metadata.json"):
            first_metadata = json.loads((first / relative_path).read_text())
            second_metadata = json.loads((second / relative_path).read_text())
            first_metadata.pop("generated_at")
            second_metadata.pop("generated_at")
            assert first_metadata == second_metadata
        else:
            assert (first / relative_path).read_bytes() == (second / relative_path).read_bytes()


def test_validator_rejects_a_second_counterfactual_label_change(tmp_path: Path) -> None:
    output_dir = tmp_path / "curriculum"
    build_dataset(output_dir=output_dir, image_size=64, seed=2026)
    rows: Any = read_jsonl(output_dir / "manifest.jsonl")
    changed: Any = next(row for row in rows if row["counterfactual_role"] == "counterfactual")
    label_path: Any = output_dir / changed["label_path"]
    labels = json.loads(label_path.read_text())
    tile = labels["entities"]["tiles"][0]
    tile["resource"] = "ORE" if tile["resource"] != "ORE" else "WOOD"
    label_path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
    for row in rows:
        if row["label_path"] == changed["label_path"]:
            row["sha256"]["labels"] = file_sha256(label_path)
    with (output_dir / "manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="disagree|changed"):
        validate_dataset(output_dir)


def test_validator_fails_closed_on_benchmark_game_id(tmp_path: Path) -> None:
    output_dir = tmp_path / "curriculum"
    build_dataset(output_dir=output_dir, image_size=64, seed=44)
    rows: Any = read_jsonl(output_dir / "manifest.jsonl")
    rows[0]["source"]["benchmark_game_id"] = "191035308"
    with (output_dir / "manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="benchmark game leaked"):
        validate_dataset(output_dir)


def test_dense_diff_names_the_single_symbolic_field() -> None:
    first: Any = {
        "tiles": [{"id": "T00", "resource": "WOOD", "number": 6, "robber": False}],
        "nodes": [],
        "edges": [],
        "ports": [],
    }
    second: Any = {
        "tiles": [{"id": "T00", "resource": "ORE", "number": 6, "robber": False}],
        "nodes": [],
        "edges": [],
        "ports": [],
    }

    assert dense_label_differences(first, second) == ["tiles/T00/resource"]
