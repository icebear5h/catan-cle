"""Prompt rows, tokens, and novel-pair filtering."""
from pathlib import Path
from typing import Any

from data_pipeline.board_recognition.adjacent_pair_localization import (
    FAR_KINDS,
    SINGLE_KINDS,
    build_pair_board,
    rows_for_pair,
)
from data_pipeline.board_recognition.replay_dataset import DEFAULT_STYLE_PATH, load_render_style
from data_pipeline.board_recognition.single_piece_localization import (
    COLORS,
)
from data_pipeline.board_recognition.spatial_localization import (
    _control_regions,
    atlas_regions,
)

from .support import _fixture_contract, _touch


def test_rows_for_pair_prompts_and_metadata() -> None:
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    controls = _control_regions(regions)
    rows: Any = rows_for_pair(
        state={**state, "split": "validation", "sample_id": "fixture_empty"},
        regions=regions,
        controls=controls,
        first=("<N17>", "SETTLEMENT", "RED"),
        second=("<E17_18>", "ROAD", "BLUE"),
        pair_kind="node_edge",
        image_name="x.png",
        empties=[
            {"token": "<N18>", "anchor": "<N17>", "negative_distance": 1, "kind": "adjacent"},
            {"token": "<E30_31>", "anchor": "<E17_18>", "negative_distance": "far", "kind": "far"},
        ],
    )
    prompts = {row["row_id"].split("fixture_empty_node_edge_N17_SETTLEMENT_RED_E17_18_ROAD_BLUE_")[1]: (row["messages"][0]["content"], row["messages"][1]["content"]) for row in rows}
    assert prompts["N17_occupancy_positive"] == ("<image>\n<N17> building?", "red settlement")
    assert prompts["N17_colored_piece_to_token"] == ("<image>\nWhich node has the red settlement?", "<N17>")
    assert prompts["N17_piece_to_token"] == ("<image>\nWhich node has the settlement?", "<N17>")
    assert prompts["E17_18_occupancy_positive"] == ("<image>\n<E17_18> road?", "blue road")
    assert prompts["E17_18_piece_to_token"] == ("<image>\nWhich edge has the road?", "<E17_18>")
    assert prompts["occupancy_negative_adjacent_0"] == ("<image>\n<N18> building?", "empty")
    assert prompts["occupancy_negative_far_1"] == ("<image>\n<E30_31> road?", "empty")
    assert len(rows) == 8 and len({row["row_id"] for row in rows}) == 8
    positive: Any = next(row for row in rows if row["row_id"].endswith("N17_occupancy_positive"))
    assert positive["partner_token"] == "<E17_18>" and positive["partner_piece"] == "ROAD"
    assert positive["partner_color"] == "blue" and positive["partner_distance"] == 1
    assert positive["pair_kind"] == "node_edge" and positive["same_color"] is False
    assert positive["schema"] == "catan_adjacent_pair_localization_row/v1"
    assert positive["grounding_stage"] == "adjacent_pair" and positive["task_family"] == "adjacent_pair_localization"
    assert positive["spatial_targets"][0]["token"] == "<N17>"
    negative = next(row for row in rows if row["task_type"] == "occupancy_negative_adjacent")
    assert negative["queried_token"] == "<N18>" and negative["target_token"] == "<N17>"
    assert negative["negative_distance"] == 1 and negative["category"] == "node.occupancy" and "spatial_targets" not in negative
    far = next(row for row in rows if row["task_type"] == "occupancy_negative_far")
    assert far["target_token"] == "<E17_18>" and far["category"] == "edge.owner" and far["negative_distance"] == "far"


def test_piece_to_token_only_when_the_type_is_unambiguous() -> None:
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    controls = _control_regions(regions)
    base: Any = dict(state={**state, "split": "train", "sample_id": "fixture_empty"}, regions=regions, controls=controls, image_name="x.png", empties=[])
    two_settlements = rows_for_pair(first=("<N17>", "SETTLEMENT", "RED"), second=("<N18>", "SETTLEMENT", "BLUE"), pair_kind="node_node", **base)
    assert [row["task_type"] for row in two_settlements].count("piece_to_token") == 0
    settlement_city = rows_for_pair(first=("<N17>", "SETTLEMENT", "RED"), second=("<N18>", "CITY", "BLUE"), pair_kind="node_node", **base)
    assert [row["task_type"] for row in settlement_city].count("piece_to_token") == 2
    two_roads = rows_for_pair(first=("<E17_18>", "ROAD", "RED"), second=("<E18_40>", "ROAD", "BLUE"), pair_kind="edge_edge", **base)
    assert [row["task_type"] for row in two_roads].count("piece_to_token") == 0


def test_novel_pair_drops_only_rows_that_name_the_color() -> None:
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    rows: Any = rows_for_pair(
        state={**state, "split": "validation", "sample_id": "fixture_empty"},
        regions=regions,
        controls=_control_regions(regions),
        first=("<N17>", "CITY", "NOVEL_VALIDATION_H165"),
        second=("<E17_18>", "ROAD", "BLUE"),
        pair_kind="node_edge",
        image_name="x.png",
        empties=[{"token": "<N18>", "anchor": "<N17>", "negative_distance": 1, "kind": "adjacent"}],
    )
    kinds = [(row["target_token"], row["task_type"]) for row in rows]
    assert ("<N17>", "occupancy_positive") not in kinds and ("<N17>", "colored_piece_to_token") not in kinds
    assert ("<N17>", "piece_to_token") in kinds
    assert ("<E17_18>", "occupancy_positive") in kinds and ("<E17_18>", "colored_piece_to_token") in kinds
    assert all("novel" not in row["messages"][0]["content"] for row in rows)
    assert next(row for row in rows if row["target_token"] == "<N17>")["color_heldout"] is True
    assert next(row for row in rows if row["target_token"] == "<E17_18>")["color_heldout"] is False


def test_far_pair_control_rows(tmp_path: Path) -> None:
    state, contract = _fixture_contract()
    style = load_render_style(Path(DEFAULT_STYLE_PATH))
    rows: Any = build_pair_board(
        state={**state, "split": "validation", "sample_id": "fixture_empty", "image_size": [256, 256]},
        contract=contract,
        output_images=tmp_path,
        style=style,
        images_per_kind={"node_node_far": 1, "edge_edge_far": 1, "node_edge_far": 1},
        colors=COLORS,
    )
    assert {row["pair_kind"] for row in rows} == set(FAR_KINDS)
    assert len(list(tmp_path.glob("validation_fixture_empty_*_far_*.png"))) == 3
    for row in rows:
        assert row["partner_distance"] == "far"
        if row["task_type"] == "occupancy_positive":
            assert not _touch(contract, row["target_token"], row["partner_token"])
    node_edge = [row for row in rows if row["pair_kind"] == "node_edge_far"]
    assert sum(row["task_type"] == "piece_to_token" for row in node_edge) == 2


def test_single_kinds_reuse_the_single_piece_rows(tmp_path: Path) -> None:
    state, contract = _fixture_contract()
    style = load_render_style(Path(DEFAULT_STYLE_PATH))
    rows = build_pair_board(
        state={**state, "split": "validation", "sample_id": "fixture_empty", "image_size": [256, 256]},
        contract=contract,
        output_images=tmp_path,
        style=style,
        images_per_kind={"single_node": 2, "single_edge": 1, "node_edge": 1},
        colors=COLORS,
        novel_color="NOVEL_VALIDATION_H165",
        asset_root=None,
    )
    images = sorted(path.name for path in tmp_path.glob("validation_fixture_empty_*.png"))
    assert len(images) == 4 and sum("_single_" in name for name in images) == 3
    singles: Any = [row for row in rows if row["pair_kind"] in SINGLE_KINDS]
    assert {row["pair_kind"] for row in singles} == set(SINGLE_KINDS)
    assert all(row["partner_distance"] == "none" and "partner_token" not in row for row in singles)
    per_image: Any = {}
    for row in singles:
        per_image.setdefault(row["images"][0], []).append(row["task_type"])
    for task_types in per_image.values():
        assert task_types.count("occupancy_negative_adjacent") == 1 and task_types.count("occupancy_negative_far") == 1
        assert task_types.count("occupancy_positive") <= 1
    node_positive: Any = next(row for row in singles if row["pair_kind"] == "single_node" and row["task_type"] == "occupancy_positive")
    assert node_positive["messages"][0]["content"].endswith("building?") and node_positive["schema"].startswith("catan_single_piece")
    pair_rows = [row for row in rows if row["pair_kind"] == "node_edge"]
    assert pair_rows and all(row["partner_distance"] == 1 for row in pair_rows)
