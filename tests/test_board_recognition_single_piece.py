import json
from pathlib import Path

import pytest

from data_pipeline.board_recognition.single_piece_localization import (
    COLORS,
    NOVEL_HUE_INTERVALS,
    build_board,
    forward_answer,
    novel_hue,
    place_piece,
    recolor_svg,
    rows_for_placement,
    sample_placements,
    write_novel_sprites,
)
from data_pipeline.board_recognition.replay_dataset import DEFAULT_STYLE_PATH, load_render_style
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


def test_place_piece_sets_exactly_one_piece_and_refuses_occupied_locations():
    _, contract = _fixture_contract()

    with_city = place_piece(contract, "<N17>", "CITY", "RED")
    with_road = place_piece(contract, "<E00_01>", "ROAD", "MYSTIC_BLUE")

    node = next(entry for entry in with_city["nodes"] if entry["token"] == "<N17>")
    assert (node["building"], node["color"], node["color_token"]) == ("CITY", "RED", "<RED>")
    assert sum(entry["building"] is not None for entry in with_city["nodes"]) == 1
    edge = next(entry for entry in with_road["edges"] if entry["token"] == "<E00_01>")
    assert edge["road_color"] == "MYSTIC_BLUE"
    assert sum(entry["road_color"] is not None for entry in with_road["edges"]) == 1
    assert all(entry["building"] is None for entry in contract["nodes"])
    with pytest.raises(SpatialLocalizationError):
        place_piece(with_city, "<N17>", "SETTLEMENT", "BLUE")


def test_sampling_is_deterministic_and_covers_every_color():
    tokens = [f"<N{i:02d}>" for i in range(54)]

    first = sample_placements(sample_id="board_a", entity_type="node", tokens=tokens, colors=COLORS, count=70)
    second = sample_placements(sample_id="board_a", entity_type="node", tokens=tokens, colors=COLORS, count=70)
    other = sample_placements(sample_id="board_b", entity_type="node", tokens=tokens, colors=COLORS, count=70)

    assert first == second and first != other
    assert len(set(first)) == 70
    assert {piece for _, piece, _ in first} == {"SETTLEMENT", "CITY"}
    assert len({color for _, _, color in first}) >= 9


def test_novel_hue_lands_in_a_gap_and_recolor_keeps_shading(tmp_path):
    hue = novel_hue("validation:abc")
    assert any(low <= hue < high for low, high in NOVEL_HUE_INTERVALS)
    assert novel_hue("validation:abc") == hue and novel_hue("test:abc") != hue

    svg = '<stop stop-color="#FF0000"/><stop stop-color="#B30000"/><path fill="#8C0039"/><g fill="#CCCCCC"/>'
    rotated = recolor_svg(svg, 165)
    assert "#CCCCCC" in rotated and "#FF0000" not in rotated
    colors = [c for c in rotated.split('"') if c.startswith("#")]
    assert len(set(colors)) == 4  # three shaded stops stay distinct, gray untouched

    written = write_novel_sprites(tmp_path / "assets", "NOVEL_VALIDATION_H165", 165)
    assert [path.name for path in written] == [
        "settlement_novel_validation_h165.svg",
        "city_novel_validation_h165.svg",
        "road_novel_validation_h165.svg",
    ]
    assert (tmp_path / "assets" / "pieces" / "settlement_red.svg").is_file()


def test_novel_color_rows_never_name_the_color():
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    rows = rows_for_placement(
        state={**state, "split": "validation", "sample_id": "fixture_empty"},
        regions=regions,
        controls=_control_regions(regions),
        token="<N17>",
        piece="CITY",
        color="NOVEL_VALIDATION_H165",
        image_name="x.png",
        empty_token="<N20>",
    )
    assert [row["task_type"] for row in rows] == ["piece_to_token", "occupancy_negative"]
    assert all(row["color_heldout"] is True for row in rows)
    assert "novel" not in rows[0]["messages"][0]["content"]


def test_rows_mirror_production_forward_prompts():
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    controls = _control_regions(regions)

    rows = rows_for_placement(
        state={**state, "split": "validation", "sample_id": "fixture_empty"},
        regions=regions,
        controls=controls,
        token="<E00_01>",
        piece="ROAD",
        color="MYSTIC_BLUE",
        image_name="x.png",
        empty_token="<E00_05>",
    )

    prompts = {row["task_type"]: (row["messages"][0]["content"], row["messages"][1]["content"]) for row in rows}
    assert prompts["piece_to_token"] == ("<image>\nWhich edge has the road?", "<E00_01>")
    assert prompts["colored_piece_to_token"] == ("<image>\nWhich edge has the mystic blue road?", "<E00_01>")
    assert prompts["occupancy_positive"] == ("<image>\n<E00_01> road?", "mystic blue road")
    assert prompts["occupancy_negative"] == ("<image>\n<E00_05> road?", "empty")
    assert all(row["category"] in {"localization", "edge.owner"} for row in rows)
    assert all(row["color_heldout"] is False for row in rows)
    assert len(rows) == 4 and "spatial_targets" not in rows[3] and rows[0]["spatial_targets"][0]["token"] == "<E00_01>"
    assert forward_answer("PINK", "SETTLEMENT") == "pink settlement"


def test_build_board_renders_one_image_per_placement(tmp_path):
    state, contract = _fixture_contract()
    style = load_render_style(Path(DEFAULT_STYLE_PATH))
    rows = build_board(
        state={**state, "split": "test", "sample_id": "fixture_empty", "image_size": [256, 256]},
        contract=contract,
        output_images=tmp_path,
        style=style,
        images_per_entity=1,
        colors=COLORS,
    )

    assert len(rows) == 8
    assert len(list(tmp_path.glob("test_fixture_empty_*.png"))) == 2
    assert {row["entity_type"] for row in rows} == {"node", "edge"}


def test_tile_rows_use_production_prompts_and_unique_descriptions():
    from data_pipeline.board_recognition.single_piece_localization import tile_facts, tile_rows_for_image

    state, contract = _fixture_contract()
    tiles = tile_facts(contract)
    assert len(tiles) == 19
    assert sum(tile["resource"] == "desert" for tile in tiles) == 1
    desert = next(tile for tile in tiles if tile["resource"] == "desert")
    assert desert["number"] == "none" and desert["description"] == "the desert tile"
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    rows = tile_rows_for_image(
        state={**state, "split": "train", "sample_id": "fixture_empty"},
        tiles=tiles,
        regions=regions,
        controls=_control_regions(regions),
        image_name="x.png",
        salt="N17_CITY_RED",
    )
    prompts = {row["task_type"]: (row["messages"][0]["content"], row["messages"][1]["content"]) for row in rows}
    token = rows[0]["target_token"]
    assert prompts["tile_resource"][0] == f"<image>\n{token} resource?"
    assert prompts["tile_number"][0] == f"<image>\n{token} number?"
    assert prompts["tile_to_token"][1] == token and prompts["tile_to_token"][0].startswith("<image>\nWhere is the ")
    assert all(row["entity_type"] == "tile" for row in rows)
