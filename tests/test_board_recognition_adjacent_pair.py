import json
from pathlib import Path

import pytest

from data_pipeline.board_recognition.adjacent_pair_localization import (
    PAIR_KINDS,
    build_pair_board,
    cross_touching,
    location_pairs,
    parse_kind_counts,
    place_pair,
    rows_for_pair,
    sample_pair_empty_tokens,
    sample_pairs,
)
from data_pipeline.board_recognition.replay_dataset import DEFAULT_STYLE_PATH, load_render_style
from data_pipeline.board_recognition.single_piece_localization import (
    COLORS,
    NEAR_MAX_HOPS,
    neighbor_distances,
    neighbor_tokens,
    place_piece,
    render_contract,
    render_placement,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _control_regions,
    atlas_regions,
)


FIXTURE_ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")


def _fixture_contract():
    manifest = [json.loads(line) for line in (FIXTURE_ROOT / "manifest.jsonl").read_text().splitlines()]
    state = next(row for row in manifest if row["sample_id"] == "empty_setup_node_p000_base")
    return state, json.loads((FIXTURE_ROOT / state["contract_path"]).read_text())


def _touch(contract, first, second):
    neighbors = neighbor_tokens(contract)
    touching = cross_touching(contract)
    return second in neighbors[first] or second in touching[first]


def test_location_pairs_touch_per_kind():
    _, contract = _fixture_contract()
    counts = {kind: len(location_pairs(contract, kind)) for kind in PAIR_KINDS}
    assert counts == {"node_node": 72, "edge_edge": 126, "node_edge": 144}
    for kind in PAIR_KINDS:
        for first, second in location_pairs(contract, kind):
            assert first != second and _touch(contract, first, second)
    with pytest.raises(SpatialLocalizationError):
        location_pairs(contract, "tile_tile")


def test_sample_pairs_is_deterministic_and_follows_the_color_rules():
    _, contract = _fixture_contract()
    for kind in PAIR_KINDS:
        pairs = location_pairs(contract, kind)
        first = sample_pairs(sample_id="board_a", pair_kind=kind, pairs=pairs, colors=COLORS, count=40)
        again = sample_pairs(sample_id="board_a", pair_kind=kind, pairs=pairs, colors=COLORS, count=40)
        other = sample_pairs(sample_id="board_b", pair_kind=kind, pairs=pairs, colors=COLORS, count=40)
        assert first == again and first != other
        assert len({(a[0], b[0]) for a, b in first}) == 40
        same = [a[2] == b[2] for a, b in first]
        if kind == "node_edge":
            assert 8 <= sum(same) <= 32
        else:
            assert not any(same)
        novel = sample_pairs(
            sample_id="board_a", pair_kind=kind, pairs=pairs, colors=COLORS, count=10, novel_color="NOVEL_VALIDATION_H165"
        )
        for a, b in novel:
            assert (a[2] == "NOVEL_VALIDATION_H165") != (b[2] == "NOVEL_VALIDATION_H165")
            assert a[2] != b[2]
    with pytest.raises(SpatialLocalizationError):
        sample_pairs(sample_id="x", pair_kind="node_node", pairs=location_pairs(contract, "node_node"), colors=COLORS, count=73)


def test_place_pair_keeps_both_pieces_and_refuses_overlap():
    _, contract = _fixture_contract()
    placed = place_pair(contract, ("<N17>", "CITY", "RED"), ("<E17_18>", "ROAD", "BLUE"))
    node = next(entry for entry in placed["nodes"] if entry["token"] == "<N17>")
    edge = next(entry for entry in placed["edges"] if entry["token"] == "<E17_18>")
    assert (node["building"], node["color"], edge["road_color"]) == ("CITY", "RED", "BLUE")
    assert sum(entry["building"] is not None for entry in placed["nodes"]) == 1
    assert sum(entry["road_color"] is not None for entry in placed["edges"]) == 1
    with pytest.raises(SpatialLocalizationError):
        place_pair(contract, ("<N17>", "CITY", "RED"), ("<N17>", "SETTLEMENT", "BLUE"))


def test_pair_negatives_touch_a_piece_and_far_ones_touch_nothing():
    state, contract = _fixture_contract()
    neighbors = neighbor_tokens(contract)
    touching = cross_touching(contract)
    tokens = [n["token"] for n in contract["nodes"]] + [e["token"] for e in contract["edges"]]
    first, second = ("<N17>", "SETTLEMENT", "RED"), ("<E17_18>", "ROAD", "BLUE")
    empties = sample_pair_empty_tokens(
        sample_id=state["sample_id"], first=first, second=second, tokens=tokens,
        neighbors=neighbors, touching=touching, counts={"adjacent": 4, "far": 3},
    )
    assert [e["kind"] for e in empties] == ["adjacent"] * 4 + ["far"] * 3
    assert len({e["token"] for e in empties}) == 7
    for empty in empties:
        token = empty["token"]
        assert token not in ("<N17>", "<E17_18>")
        touches = _touch(contract, "<N17>", token) or _touch(contract, "<E17_18>", token)
        assert touches is (empty["kind"] == "adjacent")
        assert empty["negative_distance"] == (1 if touches else "far")
        if empty["kind"] == "far":
            for anchor in ("<N17>", "<E17_18>"):
                if anchor[1] == token[1]:
                    assert token not in neighbor_distances(neighbors, anchor, NEAR_MAX_HOPS)
    again = sample_pair_empty_tokens(
        sample_id=state["sample_id"], first=first, second=second, tokens=tokens,
        neighbors=neighbors, touching=touching, counts={"adjacent": 4, "far": 3},
    )
    assert again == empties


def test_rows_for_pair_prompts_and_metadata():
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    controls = _control_regions(regions)
    rows = rows_for_pair(
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
    positive = next(row for row in rows if row["row_id"].endswith("N17_occupancy_positive"))
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


def test_piece_to_token_only_when_the_type_is_unambiguous():
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    controls = _control_regions(regions)
    base = dict(state={**state, "split": "train", "sample_id": "fixture_empty"}, regions=regions, controls=controls, image_name="x.png", empties=[])
    two_settlements = rows_for_pair(first=("<N17>", "SETTLEMENT", "RED"), second=("<N18>", "SETTLEMENT", "BLUE"), pair_kind="node_node", **base)
    assert [row["task_type"] for row in two_settlements].count("piece_to_token") == 0
    settlement_city = rows_for_pair(first=("<N17>", "SETTLEMENT", "RED"), second=("<N18>", "CITY", "BLUE"), pair_kind="node_node", **base)
    assert [row["task_type"] for row in settlement_city].count("piece_to_token") == 2
    two_roads = rows_for_pair(first=("<E17_18>", "ROAD", "RED"), second=("<E18_40>", "ROAD", "BLUE"), pair_kind="edge_edge", **base)
    assert [row["task_type"] for row in two_roads].count("piece_to_token") == 0


def test_novel_pair_drops_only_rows_that_name_the_color():
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    rows = rows_for_pair(
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


def test_build_pair_board_renders_one_image_per_pair(tmp_path):
    state, contract = _fixture_contract()
    style = load_render_style(Path(DEFAULT_STYLE_PATH))
    rows = build_pair_board(
        state={**state, "split": "test", "sample_id": "fixture_empty", "image_size": [256, 256]},
        contract=contract,
        output_images=tmp_path,
        style=style,
        images_per_kind=1,
        colors=COLORS,
    )
    images = sorted(path.name for path in tmp_path.glob("test_fixture_empty_*.png"))
    assert len(images) == 3 and {name.split("_")[3] + "_" + name.split("_")[4] for name in images} == set(PAIR_KINDS)
    assert {row["pair_kind"] for row in rows} == set(PAIR_KINDS)
    per_image = {}
    for row in rows:
        per_image.setdefault(row["images"][0], []).append(row["task_type"])
    for name, task_types in per_image.items():
        assert task_types.count("occupancy_positive") == 2
        assert task_types.count("occupancy_negative_adjacent") == 1 and task_types.count("occupancy_negative_far") == 1
    neighbors = neighbor_tokens(contract)
    touching = cross_touching(contract)
    for row in rows:
        if row["polarity"] != "hard_negative":
            continue
        queried, target, partner = row["queried_token"], row["target_token"], row["partner_token"]
        touches = queried in neighbors[target] or queried in touching[target] or queried in neighbors[partner] or queried in touching[partner]
        assert touches is (row["negative_kind"] == "adjacent")


def test_per_kind_image_counts(tmp_path):
    assert parse_kind_counts("40", 0) == {"node_node": 40, "edge_edge": 40, "node_edge": 40}
    assert parse_kind_counts(7, 0) == {"node_node": 7, "edge_edge": 7, "node_edge": 7}
    assert parse_kind_counts("edge_edge=80,node_edge=50", 40) == {"node_node": 40, "edge_edge": 80, "node_edge": 50}
    with pytest.raises(SpatialLocalizationError):
        parse_kind_counts("tile_tile=3", 40)
    state, contract = _fixture_contract()
    style = load_render_style(Path(DEFAULT_STYLE_PATH))
    rows = build_pair_board(
        state={**state, "split": "test", "sample_id": "fixture_empty", "image_size": [256, 256]},
        contract=contract,
        output_images=tmp_path,
        style=style,
        images_per_kind={"node_node": 0, "edge_edge": 2, "node_edge": 1},
        colors=COLORS,
    )
    assert len(list(tmp_path.glob("test_fixture_empty_*.png"))) == 3
    assert {row["pair_kind"] for row in rows} == {"edge_edge", "node_edge"}
    assert sum(row["task_type"] == "occupancy_positive" and row["piece"] == "ROAD" for row in rows) == 5


def test_render_placement_matches_render_contract(tmp_path):
    state, contract = _fixture_contract()
    style = load_render_style(Path(DEFAULT_STYLE_PATH))
    render_placement(contract, "<N17>", "CITY", "RED", 256, style, tmp_path / "a.png")
    render_contract(place_piece(contract, "<N17>", "CITY", "RED"), 256, style, tmp_path / "b.png")
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
