"""Tile, path, and readout scoring strictness."""

import json
from itertools import permutations

import pytest

import sft.scripts.eval.eval_qwen_vl_adapter as evaluator
from sft.board.spatial_tasks import (
    score_spatial_task,
    shortest_node_path,
)

from .support import PATH, PATH_METADATA, TILE_METADATA, ZERO


def test_tiles_are_an_exact_unordered_set_with_whitespace_tolerance() -> None:
    gold = "<T00> <T05> <T06>"
    for order in permutations(gold.split()):
        result = score_spatial_task(gold, " \n\t".join(order), TILE_METADATA)
        assert result["correct"] and result["scoring"] == "node_tiles"
        assert result["response_normalized"] == gold
    boundary = {"task_type": "node_tiles", "target": {"node": "<N53>"}}
    assert score_spatial_task("<T18>", " \n<T18>\t", boundary)["correct"]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "<T00> <T05>",
        "<T00> <T05> <T06> <T06>",
        "<T00> <T05> <T06> <T07>",
        "<T00> <T05> <T07>",
        "<T00> <T05> <T99>",
        "<N00> <T05> <T06>",
        "Tiles: <T00> <T05> <T06>",
        "<T00> <T05> <T06> because they touch.",
        "<T00>, <T05>, <T06>",
        "<T00><T05><T06>",
        '["<T00>", "<T05>", "<T06>"]',
    ],
)
def test_tile_duplicates_missing_extras_wrong_types_and_prose_fail(response: str) -> None:
    assert not score_spatial_task("<T00> <T05> <T06>", response, TILE_METADATA)["correct"]


def test_shortest_ties_are_accepted_without_sorting_routes() -> None:
    alternative = ["<N00>", "<N05>", "<N04>", "<N03>"]
    for gold in (PATH, alternative):
        for path in (PATH, alternative):
            result = score_spatial_task(" ".join(gold), "\n".join(path), PATH_METADATA)
            assert result["correct"]
            assert result["response_normalized"] == " ".join(path)
    singleton = {"task_type": "shortest_node_path", "target": {"start": "<N53>", "end": "<N53>"}}
    assert score_spatial_task("<N53>", "<N53>", singleton)["correct"]
    assert not score_spatial_task("<N53>", "<N53> <N24> <N53>", singleton)["correct"]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "<N03> <N02> <N01> <N00>",
        "<N01> <N00> <N02> <N03>",
        "<N00> <N02> <N01> <N03>",
        "<N00> <N01> <N02> <N09>",
        "<N00> <N03>",
        "<N00> <N01> <N00> <N01> <N02> <N03>",
        "<N00> <N01> <N06> <N07> <N08> <N09> <N02> <N03>",
        "<N00> <T01> <N02> <N03>",
        "<N00> <N99> <N02> <N03>",
        "<N00> -> <N01> -> <N02> -> <N03>",
        "<N00> <N01> <N02> <N03> is shortest",
    ],
)
def test_path_endpoints_nonedges_loops_detours_and_prose_fail(response: str) -> None:
    assert not score_spatial_task(" ".join(PATH), response, PATH_METADATA)["correct"]


def test_reordering_four_or_more_nodes_is_not_readout_scoring() -> None:
    for end in ("<N03>", "<N53>"):
        path = shortest_node_path("<N00>", end)
        assert len(path) >= 4
        gold = " ".join(path)
        response = " ".join([path[0], path[2], path[1], *path[3:]])
        metadata = {"task_type": "shortest_node_path", "target": {"start": path[0], "end": end}}
        assert evaluator.score_response(gold, response)[
            "correct"
        ]  # Preserve legacy two-arg behavior.
        result = evaluator.score_response(gold, response, metadata=metadata)
        assert not result["correct"] and result["scoring"] == "shortest_node_path"


@pytest.mark.parametrize(
    "metadata,gold",
    [
        ({"task_type": "node_tiles"}, "<T00> <T05> <T06>"),
        ({"task_type": "node_tiles", "target": None}, "<T00> <T05> <T06>"),
        ({"task_type": "node_tiles", "target": {"node": "<N54>"}}, "<T00>"),
        ({"task_type": "node_tiles", "target": {"node": "<N00>", "extra": 1}}, "<T00>"),
        (TILE_METADATA, "<T00> <T05>"),
        (TILE_METADATA, "<T00> <T05> <T06> <T06>"),
        (PATH_METADATA, "<N00> <N02> <N01> <N03>"),
        (PATH_METADATA, "<N03> <N02> <N01> <N00>"),
        ({"task_type": "shortest_node_path", "target": {"start": "<N00>"}}, " ".join(PATH)),
        (
            {"task_type": "shortest_node_path", "target": {"start": "<N00>", "end": "<N54>"}},
            " ".join(PATH),
        ),
        ({"task_type": "local_node_tiles", "target": {"node": 0}}, "{}"),
        ({"task_type": "dice_production", "target": {"color": "red", "roll": 8}}, json.dumps(ZERO)),
        (
            {"task_type": "dice_production", "target": {"color": "PURPLE", "roll": 8}},
            json.dumps(ZERO),
        ),
        (
            {"task_type": "dice_production", "target": {"color": "RED", "roll": True}},
            json.dumps(ZERO),
        ),
        (
            {"task_type": "dice_production", "target": {"color": "RED", "roll": 8.0}},
            json.dumps(ZERO),
        ),
        (
            {"task_type": "dice_production", "target": {"color": "RED", "roll": 13}},
            json.dumps(ZERO),
        ),
        (
            {"task_type": "dice_production", "target": {"color": "RED", "roll": 7}},
            json.dumps({**ZERO, "ore": 1}),
        ),
    ],
)
def test_bad_gold_and_metadata_raise_instead_of_legacy_fallback(
    metadata: dict[str, object], gold: str
) -> None:
    with pytest.raises(ValueError):
        evaluator.score_response(gold, gold, metadata=metadata)
