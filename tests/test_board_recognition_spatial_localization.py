import json
from pathlib import Path

from PIL import Image

from data_pipeline.board_recognition.spatial_localization import (
    PROBE_DOT_SCALES,
    _deterministic_shuffle,
    _relation_rows,
    atlas_regions,
    marker_rows_for_board,
    nearby_marker_groups,
    neutral_probe_rows,
)
from data_pipeline.board_recognition.spatial_robber import spatial_query_bank


FIXTURE_ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")


def _fixture_contract():
    manifest = [json.loads(line) for line in (FIXTURE_ROOT / "manifest.jsonl").read_text().splitlines()]
    state = next(row for row in manifest if row["sample_id"] == "empty_setup_node_p000_base")
    return state, json.loads((FIXTURE_ROOT / state["contract_path"]).read_text())


def test_renderer_aligned_regions_cover_the_full_atlas():
    _, contract = _fixture_contract()

    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)

    assert len(regions) == 154
    assert sum(row["entity_type"] == "node" for row in regions.values()) == 54
    assert sum(row["entity_type"] == "edge" for row in regions.values()) == 72
    assert sum(row["entity_type"] == "tile" for row in regions.values()) == 19
    assert sum(row["entity_type"] == "port" for row in regions.values()) == 9
    assert all(0 <= value <= 1 for row in regions.values() for value in row["bbox"])


def test_marker_groups_are_local_same_type_and_never_singletons():
    _, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)

    groups = nearby_marker_groups(regions)

    assert len(groups) == 40
    assert sum(map(len, groups)) == 154
    assert all(2 <= len(group) <= 4 for group in groups)
    assert all(len({regions[token]["entity_type"] for token in group}) == 1 for group in groups)


def test_marker_rows_are_bidirectional_and_have_control_bboxes(tmp_path):
    fixture_state, contract = _fixture_contract()
    state = {
        **fixture_state,
        "split": "validation",
        "sample_id": "fixture_empty",
        "image_size": [256, 256],
    }
    image = Image.new("RGB", (256, 256), (9, 103, 165))

    rows = marker_rows_for_board(
        state=state,
        contract=contract,
        base_image=image,
        output_images=tmp_path,
        board_index=0,
        view_padding_factor=1.2,
    )

    assert len(rows) == 308
    assert sum(row["task_type"] == "marker_to_token" for row in rows) == 154
    assert sum(row["task_type"] == "token_to_marker" for row in rows) == 154
    assert len(list(tmp_path.glob("*.png"))) == 40
    assert {row["marker_style"] for row in rows} == {"validation_diamond"}
    assert all(row["spatial_targets"][0]["bbox"] != row["spatial_targets"][0]["control_bbox"] for row in rows)


def test_relation_stage_expands_every_canonical_fact_eight_times():
    states = [
        {"sample_id": f"empty_{index}", "split": "train"}
        for index in range(8)
    ]
    fact_count = sum(len(rows) for rows in spatial_query_bank().values())

    rows = _relation_rows(states, repetitions=8)

    assert fact_count == 1179
    assert len(rows) == 9432
    assert {row["grounding_stage"] for row in rows} == {"unmarked_orientation"}
    assert not any("spatial_targets" in row for row in rows)


def test_balanced_relation_stage_equalizes_yes_and_no_per_relationship():
    states = [
        {"sample_id": f"empty_{index}", "split": "train"}
        for index in range(8)
    ]

    base = _relation_rows(states, repetitions=8)
    balanced = _relation_rows(states, repetitions=8, balance_polarity=True)

    assert len(balanced) > len(base)
    assert {row["row_id"] for row in base} <= {row["row_id"] for row in balanced}
    assert len({row["row_id"] for row in balanced}) == len(balanced)
    counts: dict[tuple[str, str, str], int] = {}
    for row in balanced:
        if row["polarity"] in {"positive", "hard_negative"}:
            key = (row["entity_type"], row["relationship"], row["polarity"])
            counts[key] = counts.get(key, 0) + 1
    for entity, relationship in {(k[0], k[1]) for k in counts}:
        assert counts[(entity, relationship, "positive")] == counts[
            (entity, relationship, "hard_negative")
        ]


def test_deterministic_shuffle_is_a_stable_permutation():
    rows = [{"row_id": f"row_{index:03d}", "sampling_repeat": index % 2} for index in range(64)]

    first = _deterministic_shuffle(rows, "stage1")
    second = _deterministic_shuffle(rows, "stage1")

    assert first == second
    assert sorted(first, key=lambda row: (row["row_id"], row["sampling_repeat"])) == rows
    assert [row["row_id"] for row in first] != [row["row_id"] for row in rows]
    assert _deterministic_shuffle(rows, "stage2") != first


def test_neutral_probe_covers_every_node_and_edge_once(tmp_path):
    fixture_state, contract = _fixture_contract()
    state = {
        **fixture_state,
        "split": "test",
        "sample_id": "fixture_empty",
        "image_size": [512, 512],
    }

    rows = neutral_probe_rows(
        state=state,
        contract=contract,
        base_image=Image.new("RGB", (512, 512), (9, 103, 165)),
        output_images=tmp_path,
        view_padding_factor=1.2,
    )

    assert len(rows) == 252
    assert len(list(tmp_path.glob("probe_*.png"))) == 252
    assert sum(row["entity_type"] == "node" for row in rows) == 108
    assert sum(row["entity_type"] == "edge" for row in rows) == 144
    assert {row["probe_style"] for row in rows} == set(PROBE_DOT_SCALES)
    assert len({row["row_id"] for row in rows}) == 252
    small = [row for row in rows if row["probe_style"] == "heldout_gray_dot_small"]
    large = [row for row in rows if row["probe_style"] == "heldout_gray_dot_large"]
    assert small[0]["probe_dot_radius_px"] < large[0]["probe_dot_radius_px"]
