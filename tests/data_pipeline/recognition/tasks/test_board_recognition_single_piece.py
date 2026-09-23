import json
from pathlib import Path
from typing import Any

import pytest

from data_pipeline.board_recognition.replay_dataset import DEFAULT_STYLE_PATH, load_render_style
from data_pipeline.board_recognition.single_piece_localization import (
    COLORS,
    NOVEL_HUE_INTERVALS,
    build_board,
    forward_answer,
    neighbor_distances,
    neighbor_tokens,
    novel_hue,
    parse_negatives,
    place_piece,
    recolor_svg,
    rows_for_placement,
    sample_empty_tokens,
    sample_placements,
    tile_facts,
    tile_rows_for_image,
    write_novel_sprites,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _control_regions,
    atlas_regions,
)

Contract = dict[str, Any]

FIXTURE_ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")


def _fixture_contract() -> tuple[Contract, Contract]:
    manifest = [json.loads(line) for line in (FIXTURE_ROOT / "manifest.jsonl").read_text().splitlines()]
    state = next(row for row in manifest if row["sample_id"] == "empty_setup_node_p000_base")
    return state, json.loads((FIXTURE_ROOT / state["contract_path"]).read_text())


def test_place_piece_sets_exactly_one_piece_and_refuses_occupied_locations() -> None:
    _, contract = _fixture_contract()

    with_city: Any = place_piece(contract, "<N17>", "CITY", "RED")
    with_road: Any = place_piece(contract, "<E00_01>", "ROAD", "MYSTIC_BLUE")

    node: Any = next(entry for entry in with_city["nodes"] if entry["token"] == "<N17>")
    assert (node["building"], node["color"], node["color_token"]) == ("CITY", "RED", "<RED>")
    assert sum(entry["building"] is not None for entry in with_city["nodes"]) == 1
    edge: Any = next(entry for entry in with_road["edges"] if entry["token"] == "<E00_01>")
    assert edge["road_color"] == "MYSTIC_BLUE"
    assert sum(entry["road_color"] is not None for entry in with_road["edges"]) == 1
    assert all(entry["building"] is None for entry in contract["nodes"])
    with pytest.raises(SpatialLocalizationError):
        place_piece(with_city, "<N17>", "SETTLEMENT", "BLUE")


def test_sampling_is_deterministic_and_covers_every_color() -> None:
    tokens = [f"<N{i:02d}>" for i in range(54)]

    first = sample_placements(sample_id="board_a", entity_type="node", tokens=tokens, colors=COLORS, count=70)
    second = sample_placements(sample_id="board_a", entity_type="node", tokens=tokens, colors=COLORS, count=70)
    other = sample_placements(sample_id="board_b", entity_type="node", tokens=tokens, colors=COLORS, count=70)

    assert first == second and first != other
    assert len(set(first)) == 70
    assert {piece for _, piece, _ in first} == {"SETTLEMENT", "CITY"}
    assert len({color for _, _, color in first}) >= 9


def test_novel_hue_lands_in_a_gap_and_recolor_keeps_shading(tmp_path: Path) -> None:
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


def test_novel_color_rows_never_name_the_color() -> None:
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    rows: Any = rows_for_placement(
        state={**state, "split": "validation", "sample_id": "fixture_empty"},
        regions=regions,
        controls=_control_regions(regions),
        token="<N17>",
        piece="CITY",
        color="NOVEL_VALIDATION_H165",
        image_name="x.png",
        empty_tokens={"adjacent": ["<N16>"], "far": ["<N40>"]},
    )
    assert [row["task_type"] for row in rows] == [
        "piece_to_token",
        "occupancy_negative_adjacent",
        "occupancy_negative_far",
    ]
    assert all(row["color_heldout"] is True for row in rows)
    assert "novel" not in rows[0]["messages"][0]["content"]


def test_rows_mirror_production_forward_prompts() -> None:
    state, contract = _fixture_contract()
    regions = atlas_regions(contract, image_size=1024, view_padding_factor=1.2)
    controls = _control_regions(regions)

    rows: Any = rows_for_placement(
        state={**state, "split": "validation", "sample_id": "fixture_empty"},
        regions=regions,
        controls=controls,
        token="<E00_01>",
        piece="ROAD",
        color="MYSTIC_BLUE",
        image_name="x.png",
        empty_tokens={"adjacent": ["<E00_05>", "<E05_16>"], "far": ["<E30_31>"]},
        distances={"<E00_05>": 1, "<E05_16>": 2},
    )

    prompts = {row["task_type"]: (row["messages"][0]["content"], row["messages"][1]["content"]) for row in rows}
    assert prompts["piece_to_token"] == ("<image>\nWhich edge has the road?", "<E00_01>")
    assert prompts["colored_piece_to_token"] == ("<image>\nWhich edge has the mystic blue road?", "<E00_01>")
    assert prompts["occupancy_positive"] == ("<image>\n<E00_01> road?", "mystic blue road")
    negatives: Any = [row for row in rows if row["polarity"] == "hard_negative"]
    assert [(row["queried_token"], row["messages"][0]["content"], row["category"]) for row in negatives] == [
        ("<E00_05>", "<image>\n<E00_05> road?", "edge.owner"),
        ("<E05_16>", "<image>\n<E05_16> road?", "edge.owner"),
        ("<E30_31>", "<image>\n<E30_31> road?", "edge.owner"),
    ]
    assert all(row["messages"][1]["content"] == "empty" for row in negatives)
    assert [row["negative_kind"] for row in negatives] == ["adjacent", "adjacent", "far"]
    assert [row["negative_distance"] for row in negatives] == [1, 2, "far"]
    assert len({row["row_id"] for row in rows}) == len(rows) == 6
    assert all(row["color_heldout"] is False for row in rows)
    assert "spatial_targets" not in rows[3] and rows[0]["spatial_targets"][0]["token"] == "<E00_01>"
    assert forward_answer("PINK", "SETTLEMENT") == "pink settlement"


def test_build_board_renders_one_image_per_placement(tmp_path: Path) -> None:
    state, contract = _fixture_contract()
    style = load_render_style(Path(DEFAULT_STYLE_PATH))
    rows: Any = build_board(
        state={**state, "split": "test", "sample_id": "fixture_empty", "image_size": [256, 256]},
        contract=contract,
        output_images=tmp_path,
        style=style,
        images_per_entity=1,
        colors=COLORS,
    )

    assert len(rows) == 10  # 3 named rows + 1 adjacent + 1 far, for one node and one edge image
    assert len(list(tmp_path.glob("test_fixture_empty_*.png"))) == 2
    assert {row["entity_type"] for row in rows} == {"node", "edge"}
    neighbors: Any = neighbor_tokens(contract)
    for row in rows:
        if row["polarity"] != "hard_negative":
            continue
        queried, target = row["queried_token"], row["target_token"]
        hops: Any = neighbor_distances(neighbors, target).get(queried)
        assert queried != target
        assert (hops is not None) is (row["negative_kind"] == "adjacent")
        assert row["negative_distance"] == (hops if hops else "far")
    adjacent = [row for row in rows if row.get("negative_kind") == "adjacent"]
    assert all(row["negative_distance"] == 1 for row in adjacent)  # touching locations rank first
    fewer = build_board(
        state={**state, "split": "test", "sample_id": "fixture_empty", "image_size": [256, 256]},
        contract=contract,
        output_images=tmp_path,
        style=style,
        images_per_entity=1,
        colors=COLORS,
        negatives={"adjacent": 1, "far": 3},
    )
    assert len(fewer) == 14
    assert parse_negatives("far=3") == {"adjacent": 1, "far": 3}
    with pytest.raises(SpatialLocalizationError):
        parse_negatives("cross=2")


def test_neighbor_tokens_follow_the_contract_graph_and_empty_sampling_splits_them() -> None:
    _, contract = _fixture_contract()
    neighbors = neighbor_tokens(contract)

    assert neighbors["<N00>"] == ["<N01>", "<N05>", "<N20>"]
    assert neighbors["<E00_01>"] == ["<E00_05>", "<E00_20>", "<E01_02>", "<E01_06>"]
    assert all(2 <= len(neighbors[t]) <= 3 for t in neighbors if t.startswith("<N"))
    assert all(2 <= len(neighbors[t]) <= 4 for t in neighbors if t.startswith("<E"))
    distances = neighbor_distances(neighbors, "<N00>")
    assert {t for t, hops in distances.items() if hops == 1} == {"<N01>", "<N05>", "<N20>"}
    assert {t for t, hops in distances.items() if hops == 2} == {"<N02>", "<N06>", "<N04>", "<N16>", "<N19>", "<N22>"}

    tokens = [node["token"] for node in contract["nodes"]]
    counts = {"adjacent": 7, "far": 7}
    first = sample_empty_tokens(
        sample_id="board_a", token="<N00>", piece="CITY", color="RED", tokens=tokens, neighbors=neighbors, counts=counts
    )
    again = sample_empty_tokens(
        sample_id="board_a", token="<N00>", piece="CITY", color="RED", tokens=tokens, neighbors=neighbors, counts=counts
    )
    assert first == again
    assert set(first["adjacent"][:3]) == {"<N01>", "<N05>", "<N20>"}  # touching first
    assert len(first["adjacent"]) == 7 and all(distances[t] == 2 for t in first["adjacent"][3:])
    assert len(first["far"]) == 7 and not set(first["far"]) & set(distances)
    capped = sample_empty_tokens(
        sample_id="board_a", token="<N00>", piece="CITY", color="RED", tokens=tokens, neighbors=neighbors, counts={"adjacent": 99, "far": 0}
    )
    assert len(capped["adjacent"]) == len(distances) - 1 and capped["far"] == []


def test_tile_rows_use_production_prompts_and_unique_descriptions() -> None:
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
    prompts: Any = {row["task_type"]: (row["messages"][0]["content"], row["messages"][1]["content"]) for row in rows}
    token: Any = rows[0]["target_token"]
    assert prompts["tile_resource"][0] == f"<image>\n{token} resource?"
    assert prompts["tile_number"][0] == f"<image>\n{token} number?"
    assert prompts["tile_to_token"][1] == token and prompts["tile_to_token"][0].startswith("<image>\nWhere is the ")
    assert all(row["entity_type"] == "tile" for row in rows)
