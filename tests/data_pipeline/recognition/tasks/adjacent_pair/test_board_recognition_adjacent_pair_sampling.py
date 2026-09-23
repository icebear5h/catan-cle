"""Pair location, sampling, placement, and negatives."""
from typing import Any

import pytest

from data_pipeline.board_recognition.adjacent_pair_localization import (
    FAR_KINDS,
    TOUCHING_KINDS,
    cross_touching,
    location_pairs,
    place_pair,
    sample_pair_empty_tokens,
    sample_pairs,
)
from data_pipeline.board_recognition.single_piece_localization import (
    COLORS,
    NEAR_MAX_HOPS,
    neighbor_distances,
    neighbor_tokens,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
)

from .support import _fixture_contract, _touch


def test_location_pairs_touch_per_kind() -> None:
    _, contract = _fixture_contract()
    counts = {kind: len(location_pairs(contract, kind)) for kind in TOUCHING_KINDS}
    assert counts == {"node_node": 72, "edge_edge": 126, "node_edge": 144}
    for kind in TOUCHING_KINDS:
        for first, second in location_pairs(contract, kind):
            assert first != second and _touch(contract, first, second)
    neighbors = neighbor_tokens(contract)
    touching = cross_touching(contract)
    for kind in FAR_KINDS:
        pairs = location_pairs(contract, kind)
        assert len(pairs) > 100
        for first, second in pairs:
            assert not _touch(contract, first, second)
            if first[1] == second[1]:
                assert second not in neighbor_distances(neighbors, first, NEAR_MAX_HOPS)
            else:
                near = neighbor_distances(neighbors, first, NEAR_MAX_HOPS)
                assert not any(endpoint in near for endpoint in touching[second])
    with pytest.raises(SpatialLocalizationError):
        location_pairs(contract, "tile_tile")


def test_sample_pairs_is_deterministic_and_follows_the_color_rules() -> None:
    _, contract = _fixture_contract()
    for kind in TOUCHING_KINDS:
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


def test_place_pair_keeps_both_pieces_and_refuses_overlap() -> None:
    _, contract = _fixture_contract()
    placed: Any = place_pair(contract, ("<N17>", "CITY", "RED"), ("<E17_18>", "ROAD", "BLUE"))
    node: Any = next(entry for entry in placed["nodes"] if entry["token"] == "<N17>")
    edge: Any = next(entry for entry in placed["edges"] if entry["token"] == "<E17_18>")
    assert (node["building"], node["color"], edge["road_color"]) == ("CITY", "RED", "BLUE")
    assert sum(entry["building"] is not None for entry in placed["nodes"]) == 1
    assert sum(entry["road_color"] is not None for entry in placed["edges"]) == 1
    with pytest.raises(SpatialLocalizationError):
        place_pair(contract, ("<N17>", "CITY", "RED"), ("<N17>", "SETTLEMENT", "BLUE"))


def test_pair_negatives_touch_a_piece_and_far_ones_touch_nothing() -> None:
    state, contract = _fixture_contract()
    neighbors: Any = neighbor_tokens(contract)
    touching = cross_touching(contract)
    tokens = [n["token"] for n in contract["nodes"]] + [e["token"] for e in contract["edges"]]
    first, second = ("<N17>", "SETTLEMENT", "RED"), ("<E17_18>", "ROAD", "BLUE")
    empties: Any = sample_pair_empty_tokens(
        sample_id=state["sample_id"], first=first, second=second, tokens=tokens,
        neighbors=neighbors, touching=touching, counts={"adjacent": 4, "far": 3},
    )
    assert [e["kind"] for e in empties] == ["adjacent"] * 4 + ["far"] * 3
    assert len({e["token"] for e in empties}) == 7
    for empty in empties:
        token: Any = empty["token"]
        assert token not in ("<N17>", "<E17_18>")
        touches: Any = _touch(contract, "<N17>", token) or _touch(contract, "<E17_18>", token)
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
