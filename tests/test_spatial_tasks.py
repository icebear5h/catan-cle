import argparse
import copy
import json
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

import sft.scripts.eval_qwen_vl_adapter as evaluator
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import snapshot_public_board
from evals.catan_board_bench.builder import CatanObservationSuite
from evals.catan_board_bench.tokens import atlas_metadata
from sft.spatial_tasks import (
    atlas_node_graph,
    dice_production,
    local_node_tiles,
    node_tile_tokens,
    score_spatial_task,
    shortest_node_path,
)


ROOT = Path(__file__).resolve().parents[1]
ZERO = {"wood": 0, "brick": 0, "sheep": 0, "wheat": 0, "ore": 0}
PATH = ["<N00>", "<N01>", "<N02>", "<N03>"]
PATH_METADATA = {"task_type": "shortest_node_path", "target": {"start": PATH[0], "end": PATH[-1]}}
TILE_METADATA = {"task_type": "node_tiles", "target": {"node": "<N00>"}}
LOCAL_METADATA = {"task_type": "local_node_tiles", "target": {"node": "<N00>"}}
DICE_METADATA = {"task_type": "dice_production", "target": {"color": "RED", "roll": 8}}


@pytest.fixture
def game():
    return GameEngine(
        [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE],
        seed=123,
        shuffle_players=False,
    )


@pytest.fixture
def contract(game):
    return snapshot_public_board(game.observe(Color.RED)).contract()


def test_canonical_graph_and_all_node_tile_boundaries():
    atlas = atlas_metadata()
    graph = atlas_node_graph()
    assert set(graph) == {node["token"] for node in atlas["nodes"]}
    assert len(graph) == 54 and sum(map(len, graph.values())) == 144
    assert {tuple(sorted((a, b))) for a in graph for b in graph[a]} == {
        tuple(f"<N{node:02d}>" for node in edge["id"]) for edge in atlas["edges"]
    }
    assert graph["<N00>"] == {"<N01>", "<N05>", "<N20>"}
    assert graph["<N53>"] == {"<N24>", "<N52>"}
    assert node_tile_tokens("<N00>") == ["<T00>", "<T05>", "<T06>"]
    assert node_tile_tokens("<N53>") == ["<T18>"]
    counts = set()
    for node in atlas["nodes"]:
        expected = sorted(
            tile["token"] for tile in atlas["tiles"] if node["id"] in tile["nodes"].values()
        )
        assert node_tile_tokens(node["token"]) == expected
        counts.add(len(expected))
    assert counts == {1, 2, 3}
    graph["<N00>"].clear()
    node_tile_tokens("<N00>").clear()
    assert atlas_node_graph()["<N00>"] == {"<N01>", "<N05>", "<N20>"}
    assert len(node_tile_tokens("<N00>")) == 3


def test_bfs_is_deterministic_and_minimal_for_every_node_pair():
    atlas = atlas_metadata()
    graph = nx.Graph(tuple(f"<N{node:02d}>" for node in edge["id"]) for edge in atlas["edges"])
    assert shortest_node_path(*PATH_METADATA["target"].values()) == PATH
    for start, distances in nx.all_pairs_shortest_path_length(graph):
        for end, distance in distances.items():
            path = shortest_node_path(start, end)
            assert path == shortest_node_path(start, end)
            assert path[0] == start and path[-1] == end
            assert len(path) == distance + 1
            assert all(graph.has_edge(a, b) for a, b in zip(path, path[1:]))
            if start == end:
                assert path == [start]


@pytest.mark.parametrize(
    "node",
    ["<N54>", "<N99>", "<N-1>", "N00", "<N0>", "<n00>", "<T00>", " <N00>", 0, True, None, []],
)
def test_invalid_nodes_raise(node, contract):
    for call in (
        lambda: node_tile_tokens(node),
        lambda: shortest_node_path(node, "<N00>"),
        lambda: shortest_node_path("<N00>", node),
        lambda: local_node_tiles(contract, node),
    ):
        with pytest.raises(ValueError):
            call()


def test_tiles_are_an_exact_unordered_set_with_whitespace_tolerance():
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
def test_tile_duplicates_missing_extras_wrong_types_and_prose_fail(response):
    assert not score_spatial_task("<T00> <T05> <T06>", response, TILE_METADATA)["correct"]


def test_shortest_ties_are_accepted_without_sorting_routes():
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
def test_path_endpoints_nonedges_loops_detours_and_prose_fail(response):
    assert not score_spatial_task(" ".join(PATH), response, PATH_METADATA)["correct"]


def test_reordering_four_or_more_nodes_is_not_readout_scoring():
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
def test_bad_gold_and_metadata_raise_instead_of_legacy_fallback(metadata, gold):
    with pytest.raises(ValueError):
        evaluator.score_response(gold, gold, metadata=metadata)


def test_local_tiles_match_both_real_engine_contract_emitters(game, contract):
    expanded = CatanObservationSuite().public_board_contract(game)
    for token in atlas_node_graph():
        expected = {
            f"<T{tile['id']:02d}>": {
                "resource": tile["resource"].lower() if tile["resource"] is not None else "desert",
                "number": tile["number"],
            }
            for tile in contract["tiles"]
            if int(token[2:-1]) in tile["nodes"]
        }
        actual = local_node_tiles(contract, token)
        assert actual == expected == local_node_tiles(expanded, token)
        assert list(actual) == sorted(actual)
    assert local_node_tiles(contract, "<N03>")["<T03>"] == {"resource": "desert", "number": None}


def test_production_counts_cities_settlements_robber_and_ignores_bank(game):
    board = game.state.board
    for color, node in ((Color.RED, 0), (Color.RED, 2), (Color.RED, 18), (Color.BLUE, 4)):
        board.build_settlement(color, node, initial_build_phase=True)
    before_city = snapshot_public_board(game.observe(Color.RED)).contract()
    assert dice_production(before_city, "RED", 8) == {**ZERO, "ore": 3}
    board.build_city(Color.RED, 0)
    contract = snapshot_public_board(game.observe(Color.RED)).contract()
    assert dice_production(contract, "RED", 8) == {**ZERO, "ore": 4}
    assert dice_production(contract, "BLUE", 8) == {**ZERO, "ore": 1}
    assert dice_production(contract, "RED", 11) == {**ZERO, "sheep": 2}
    assert dice_production(contract, "RED", 7) == ZERO
    assert dice_production(contract, "WHITE", 8) == ZERO
    assert dice_production(contract, "RED", 2) == ZERO
    for roll in range(2, 13):
        assert dice_production(contract, "RED", roll) == dice_production(
            CatanObservationSuite().public_board_contract(game),
            "RED",
            roll,
        )
    game.state.resource_freqdeck = [0] * 5
    empty_bank = snapshot_public_board(game.observe(Color.RED)).contract()
    assert dice_production(empty_bank, "RED", 8) == {**ZERO, "ore": 4}
    board.robber_coordinate = next(
        coord for coord, tile in board.map.land_tiles.items() if tile.id == 0
    )
    blocked = snapshot_public_board(game.observe(Color.RED)).contract()
    assert dice_production(blocked, "RED", 8) == {**ZERO, "ore": 1}
    assert dice_production(blocked, "BLUE", 8) == ZERO


@pytest.mark.parametrize(
    "color,roll",
    [
        ("RED", 1),
        ("RED", 13),
        ("RED", True),
        ("RED", 8.0),
        ("RED", "8"),
        ("red", 8),
        ("<RED>", 8),
        ("PURPLE", 8),
        ("GREEN", 8),
        (None, 8),
        ([], 8),
    ],
)
def test_production_rejects_invalid_targets_and_absent_players(contract, color, roll):
    with pytest.raises(ValueError):
        dice_production(contract, color, roll)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema",), "public_board_contract/v0"),
        (("tiles", 0, "resource"), "wood"),
        (("tiles", 0, "resource"), "GOLD"),
        (("tiles", 0, "number"), True),
        (("tiles", 0, "number"), 8.0),
        (("tiles", 0, "number"), 7),
        (("tiles", 0, "number"), None),
        (("tiles", 3, "number"), 8),
        (("tiles", 0, "id"), True),
        (("tiles", 0, "id"), 1),
        (("tiles", 0, "token"), "<T01>"),
        (("tiles", 0, "nodes"), [0, 1, 2, 3, 4, 4]),
        (("tiles", 0, "coord"), [1, 0, -1]),
        (("tiles", 0, "has_robber"), 1),
        (("tiles", 0, "has_robber"), True),
        (("robber", "tile_id"), 0),
        (("robber", "tile_id"), True),
        (("robber", "coord"), [0, 0, 0]),
        (("robber", "tile_token"), "<T00>"),
        (("tiles",), []),
        (("robber",), None),
    ],
)
def test_invalid_contract_tile_and_robber_facts_raise(contract, path, value):
    container = contract
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    for call in (
        lambda: local_node_tiles(contract, "<N00>"),
        lambda: dice_production(contract, "RED", 8),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize(
    "path,value",
    [
        (("nodes", 0, "building"), "ROAD"),
        (("nodes", 0, "color"), "RED"),
        (("nodes", 0, "building"), "CITY"),
        (("nodes", 0, "color"), "PURPLE"),
        (("nodes", 0, "adjacent_tiles"), [0, 5, 5]),
        (("nodes", 0, "adjacent_tiles"), [False, 5, 6]),
        (("nodes", 0, "id"), 1),
        (("nodes", 0, "id"), 0.0),
        (("nodes", 0, "token"), "<N01>"),
        (("players", 0, "color"), "ORANGE"),
        (("players", 0, "color"), "red"),
        (("nodes",), []),
        (("players",), []),
    ],
)
def test_invalid_contract_piece_and_player_facts_raise_even_on_seven(contract, path, value):
    container = contract
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    with pytest.raises(ValueError):
        dice_production(contract, "RED", 7)


@pytest.mark.parametrize("task", ["local_node_tiles", "dice_production"])
def test_dynamic_json_is_order_insensitive_but_strict_and_never_extracts_prose(contract, task):
    metadata = LOCAL_METADATA if task == "local_node_tiles" else DICE_METADATA
    value = (
        local_node_tiles(contract, "<N00>") if task == "local_node_tiles" else {**ZERO, "ore": 1}
    )
    gold = json.dumps(value)
    reordered = {
        key: dict(reversed(list(item.items()))) if isinstance(item, dict) else item
        for key, item in reversed(list(value.items()))
    }
    assert score_spatial_task(gold, json.dumps(reordered, indent=2), metadata)["correct"]
    key = next(iter(value))
    invalid = [
        "",
        "null",
        "[]",
        "{}",
        json.dumps(list(value)),
        gold + " trailing prose",
        "Here is the answer: " + gold,
        gold + gold,
        json.dumps({k: v for k, v in value.items() if k != key}),
        json.dumps({**value, "extra": 0}),
        gold[:-1] + ", " + json.dumps(key) + ": " + json.dumps(value[key]) + "}",
    ]
    if task == "local_node_tiles":
        for replacement in (
            None,
            [],
            {"resource": "wood"},
            {"resource": "wood", "number": 8, "extra": 0},
            {"resource": "WOOD", "number": 8},
            {"resource": "desert", "number": 8},
            {"resource": "wood", "number": None},
            {"resource": "wood", "number": "8"},
            {"resource": "wood", "number": True},
            {"resource": "wood", "number": 8.0},
            {"resource": "wood", "number": 7},
            {"resource": "gold", "number": 8},
        ):
            invalid.append(json.dumps({**value, key: replacement}))
        invalid.append(gold.replace('"resource": "ore"', '"resource": "ore", "resource": "ore"', 1))
    else:
        for replacement in (True, False, 0.0, -1, "0", None, [], {}, float("nan"), float("inf")):
            invalid.append(json.dumps({**value, key: replacement}))
        invalid.append(gold.replace('"ore": 1', '"ore": 1.0'))
        invalid.append(gold.replace('"ore": 1', '"ore": true'))
        invalid.append(gold.replace('"ore": 1', '"ore": 1e0'))
    for response in invalid:
        assert not score_spatial_task(gold, response, metadata)["correct"], response
        with pytest.raises(ValueError):
            score_spatial_task(response, gold, metadata)
    changed = copy.deepcopy(value)
    if task == "local_node_tiles":
        changed[key]["number"] = 6
    else:
        changed["ore"] = 2
    assert not score_spatial_task(gold, json.dumps(changed), metadata)["correct"]


def test_normalization_is_shared_by_gold_and_response_and_legacy_is_untouched(contract):
    cases = [
        ("<T00> <T05> <T06>", "<T06> <T00> <T05>", TILE_METADATA),
        (" ".join(PATH), "<N00> <N05> <N04> <N03>", PATH_METADATA),
        (
            json.dumps(local_node_tiles(contract, "<N00>")),
            json.dumps(local_node_tiles(contract, "<N00>")),
            LOCAL_METADATA,
        ),
        (json.dumps(ZERO), json.dumps(ZERO), DICE_METADATA),
    ]
    for gold, response, metadata in cases:
        for wrapper in (" {} ", "```json\n{}\n```", "Answer: {}", "Assistant: {}<|im_end|>"):
            assert evaluator.score_response(
                wrapper.format(gold), wrapper.format(response), metadata=metadata
            )["correct"]
    for metadata in ({}, {"task_type": "unrelated", "target": None}):
        assert score_spatial_task("not gold", "anything", metadata) is None
        assert evaluator.score_response(
            "yes", "yes, because", metadata=metadata
        ) == evaluator.score_response("yes", "yes, because")
        assert evaluator.score_response('{"x": 1}', 'prose {"x": true}', metadata=metadata)[
            "correct"
        ]


@pytest.mark.parametrize("task_type", [None, "", "node_tiles", "full_board_readout"])
def test_conflicting_task_declarations_cannot_bypass_ordered_path_scoring(task_type):
    row = {"task_type": task_type, "metadata": PATH_METADATA}
    with pytest.raises(ValueError, match="conflicting spatial task declarations"):
        evaluator.evaluation_metadata(row, image_variant="original")


def test_nested_path_declaration_survives_without_top_level_override():
    for row in ({"metadata": PATH_METADATA}, {"task_type": "shortest_node_path", "metadata": PATH_METADATA}):
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        assert not evaluator.score_response(
            " ".join(PATH), "<N00> <N02> <N01> <N03>", metadata=metadata
        )["correct"]
    with pytest.raises(ValueError, match="conflicting spatial task declarations"):
        evaluator.evaluation_metadata(
            {"task_type": "unrelated", "metadata": {"category": "shortest_node_path"}},
            image_variant="original",
        )


def test_eval_job_dispatches_nested_target_before_readout(tmp_path, monkeypatch):
    row = {
        "id": "path-dispatch",
        "image": "unused.png",
        "task_type": "shortest_node_path",
        "metadata": {"target": PATH_METADATA["target"]},
        "messages": [
            {"role": "user", "content": "Give a shortest path from <N00> to <N03>."},
            {"role": "assistant", "content": " ".join(PATH)},
        ],
    }
    source = tmp_path / "eval.jsonl"
    source.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(evaluator, "resolve_dataset_asset", lambda *args: tmp_path / "unused.png")
    # Only inference is stubbed: exercise the real row, metadata, score, and receipt pipeline offline.
    monkeypatch.setattr(
        evaluator, "generate_responses", lambda **kwargs: (["<N00> <N02> <N01> <N03>"], None)
    )
    args = argparse.Namespace(
        image_root=None,
        limit=None,
        batch_size=1,
        long_batch_size=1,
        max_new_tokens=16,
        long_max_new_tokens=128,
        occlusion_margin=0.03,
        candidate_scoring=False,
        model_id="offline",
        adapter_dir=None,
        bits=16,
        token_inventory=None,
    )
    summary = evaluator.run_eval_job(
        model=None,
        processor=SimpleNamespace(tokenizer=None),
        args=args,
        adapter_evidence={"semantic_tokens": {"tokens": []}},
        eval_jsonl=str(source),
        image_variant="original",
        output_dir=tmp_path / "out",
    )
    record = json.loads((tmp_path / "out" / "records.jsonl").read_text())
    assert record["metadata"]["target"] == PATH_METADATA["target"]
    assert record["score"]["scoring"] == "shortest_node_path"
    assert not record["score"]["correct"] and summary["exact_accuracy"] == 0


@pytest.mark.parametrize(
    "relative_path,total,correct",
    [
        ("full-board-new-layouts-ck128-spatial-answer-only-v1-20260908-r01/records.jsonl", 120, 57),
        ("full-board-new-layouts-20260907/heldout-records.jsonl", 64, 64),
    ],
)
def test_archived_matched_spatial_and_full_board_scores_are_unchanged(
    relative_path, total, correct
):
    path = ROOT / "artifacts" / "runs" / "sft" / relative_path
    if not path.is_file():
        pytest.skip("local archived eval records are not included in a clean checkout")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(records) == total
    scores = [
        evaluator.score_response(row["expected"], row["response"], metadata=row["metadata"])
        for row in records
    ]
    assert sum(score["correct"] for score in scores) == correct
    for row, score in zip(records, scores):
        assert score == row["score"]
        assert score == evaluator.score_response(row["expected"], row["response"])
