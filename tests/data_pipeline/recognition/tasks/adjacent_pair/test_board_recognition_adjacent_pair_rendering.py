"""Pair board rendering and per-kind image counts."""
from pathlib import Path
from typing import Any

import pytest

from data_pipeline.board_recognition.adjacent_pair_localization import (
    FAR_KINDS,
    SINGLE_KINDS,
    TOUCHING_KINDS,
    build_pair_board,
    cross_touching,
    parse_kind_counts,
)
from data_pipeline.board_recognition.replay_dataset import DEFAULT_STYLE_PATH, load_render_style
from data_pipeline.board_recognition.single_piece_localization import (
    COLORS,
    neighbor_tokens,
    place_piece,
    render_contract,
    render_placement,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
)

from .support import _fixture_contract


def test_build_pair_board_renders_one_image_per_pair(tmp_path: Path) -> None:
    state, contract = _fixture_contract()
    style = load_render_style(Path(DEFAULT_STYLE_PATH))
    rows: Any = build_pair_board(
        state={**state, "split": "test", "sample_id": "fixture_empty", "image_size": [256, 256]},
        contract=contract,
        output_images=tmp_path,
        style=style,
        images_per_kind=1,
        colors=COLORS,
    )
    images = sorted(path.name for path in tmp_path.glob("test_fixture_empty_*.png"))
    assert len(images) == 3 and {name.split("_")[3] + "_" + name.split("_")[4] for name in images} == set(TOUCHING_KINDS)
    assert {row["pair_kind"] for row in rows} == set(TOUCHING_KINDS)
    per_image: Any = {}
    for row in rows:
        per_image.setdefault(row["images"][0], []).append(row["task_type"])
    for name, task_types in per_image.items():
        assert task_types.count("occupancy_positive") == 2
        assert task_types.count("occupancy_negative_adjacent") == 1 and task_types.count("occupancy_negative_far") == 1
    neighbors: Any = neighbor_tokens(contract)
    touching: Any = cross_touching(contract)
    for row in rows:
        if row["polarity"] != "hard_negative":
            continue
        queried, target, partner = row["queried_token"], row["target_token"], row["partner_token"]
        touches: Any = queried in neighbors[target] or queried in touching[target] or queried in neighbors[partner] or queried in touching[partner]
        assert touches is (row["negative_kind"] == "adjacent")


def test_per_kind_image_counts(tmp_path: Path) -> None:
    zero_far = {kind: 0 for kind in FAR_KINDS + SINGLE_KINDS}
    assert parse_kind_counts("40", 0) == {"node_node": 40, "edge_edge": 40, "node_edge": 40, **zero_far}
    assert parse_kind_counts(7, 0) == {"node_node": 7, "edge_edge": 7, "node_edge": 7, **zero_far}
    assert parse_kind_counts("edge_edge=80,node_edge=50", 40) == {"node_node": 0, "edge_edge": 80, "node_edge": 50, **zero_far}
    far_only = parse_kind_counts("node_node_far=10,edge_edge_far=20,node_edge_far=15", 30)
    assert far_only["node_node_far"] == 10 and far_only["node_node"] == 0 and far_only["single_node"] == 0
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


def test_render_placement_matches_render_contract(tmp_path: Path) -> None:
    state, contract = _fixture_contract()
    style = load_render_style(Path(DEFAULT_STYLE_PATH))
    render_placement(contract, "<N17>", "CITY", "RED", 256, style, tmp_path / "a.png")
    render_contract(place_piece(contract, "<N17>", "CITY", "RED"), 256, style, tmp_path / "b.png")
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
