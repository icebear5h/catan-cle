import json

import pytest

from data_pipeline.board_recognition.density_curriculum import (
    DensityCurriculumError,
    ordered_curriculum_records,
)
from sft.density_curriculum import (
    SequentialCurriculumSampler,
    curriculum_stage_for_step,
    load_curriculum_manifest,
)


def record(index: int, *, density: str, pieces: int, head: str, class_name: str):
    annotation = {"messages": [{"content": str(index)}], "images": [f"{index}.png"]}
    audit = {
        "query_id": f"state_e000_q{index:03d}",
        "state_id": f"state-{index}",
        "density_bin": density,
        "piece_count": pieces,
        "head": head,
        "class_name": class_name,
    }
    return annotation, audit


def test_curriculum_orders_stages_and_preserves_each_row_once():
    pairs = [
        record(0, density="dense", pieces=40, head="edge.owner", class_name="EMPTY"),
        record(1, density="setup", pieces=8, head="edge.owner", class_name="EMPTY"),
        record(2, density="sparse", pieces=20, head="tile.resource", class_name="WOOD"),
        record(3, density="empty", pieces=0, head="tile.resource", class_name="WOOD"),
        record(4, density="setup", pieces=16, head="edge.owner", class_name="RED"),
        record(5, density="dense", pieces=32, head="edge.owner", class_name="RED"),
    ]
    annotations = [pair[0] for pair in pairs]
    audits = [pair[1] for pair in pairs]

    ordered = ordered_curriculum_records(annotations, audits)

    densities = [row["audit"]["density_bin"] for row in ordered]
    assert set(densities[:3]) == {"empty", "setup"}
    assert densities[3] == "sparse"
    assert densities[4:] == ["dense", "dense"]
    assert {row["audit"]["query_id"] for row in ordered} == {
        row["query_id"] for row in audits
    }
    assert sorted(row["original_index"] for row in ordered) == list(range(len(pairs)))


def test_curriculum_rejects_piece_count_outside_density_boundary():
    annotation, audit = record(
        0,
        density="setup",
        pieces=17,
        head="edge.owner",
        class_name="EMPTY",
    )

    with pytest.raises(DensityCurriculumError, match="outside early"):
        ordered_curriculum_records([annotation], [audit])


def test_sequential_sampler_cannot_shuffle_curriculum_rows():
    data = list(range(8))
    sampler = SequentialCurriculumSampler(data, expected_rows=8)

    assert list(sampler) == list(range(8))
    assert len(sampler) == 8
    with pytest.raises(ValueError, match="expected 9"):
        SequentialCurriculumSampler(data, expected_rows=9)


def test_manifest_stage_lookup_and_strict_row_count(tmp_path):
    manifest = {
        "schema": "catan_board_recognition_density_curriculum/v1",
        "total_rows": 8192,
        "unique_query_ids": 8192,
        "stages": [
            {"stage": "early", "start_index": 0, "end_index_exclusive": 2000},
            {"stage": "mid", "start_index": 2000, "end_index_exclusive": 4000},
            {"stage": "late", "start_index": 4000, "end_index_exclusive": 8192},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    loaded = load_curriculum_manifest(path)

    assert curriculum_stage_for_step(loaded, 0) == "early"
    assert curriculum_stage_for_step(loaded, 3999) == "mid"
    assert curriculum_stage_for_step(loaded, 8191) == "late"
    with pytest.raises(IndexError):
        curriculum_stage_for_step(loaded, 8192)
