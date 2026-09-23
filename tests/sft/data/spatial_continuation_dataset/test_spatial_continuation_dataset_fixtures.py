"""Engine fixture labels, partitioning, and bank balance."""
import copy
import json
import random
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

from cle.game_engine.models.board import Board
from cle.game_engine.models.enums import RESOURCES
from cle.game_engine.models.player import Color
from cle.game_engine.state import yield_resources
from data_pipeline.board_recognition.spatial_robber import spatial_query_bank
from sft.board.spatial_tasks import (
    atlas_node_graph,
    dice_production,
    local_node_tiles,
    node_tile_tokens,
    shortest_node_path,
)
from sft.scripts.builders.build_spatial_continuation_dataset import (
    BANK_QUOTAS,
    _bank_queries,
    _path_pairs,
    _production_options,
    _query,
    _select_paths,
    build_dataset,
)

from .support import JsonDict


def test_engine_fixture_labels_and_output_guidance(engine_board: tuple[Board, JsonDict]) -> None:
    board, contract = engine_board
    original = copy.deepcopy(contract)
    for node in sorted(atlas_node_graph()):
        expected = {
            tile["token"]: {
                "resource": (tile["resource"] or "desert").lower(),
                "number": tile["number"],
            }
            for tile in contract["tiles"]
            if int(node[2:-1]) in tile["nodes"]
        }
        assert node_tile_tokens(node) == sorted(expected)
        query = _query("node_tiles", {"node": node}, contract)
        assert query["answer"].split() == sorted(expected)
        assert "any order" in query["prompt"]
        assert "Output only" in query["prompt"]
        assert (
            json.loads(_query("local_node_tiles", {"node": node}, contract)["answer"]) == expected
        )
        assert local_node_tiles(contract, node) == expected
    for roll in range(2, 13):
        payouts, _ = yield_resources(board, [19] * 5, roll)
        for color in (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE):
            target = {"color": color.value, "roll": roll}
            query = _query("dice_production", target, contract)
            expected = dict(zip((r.lower() for r in RESOURCES), payouts.get(color, [0] * 5)))
            assert json.loads(query["answer"]) == expected
            assert dice_production(contract, **target) == expected
            assert "two per city" in query["prompt"]
            assert "robber blocks" in query["prompt"]
            assert "Ignore bank shortages" in query["prompt"]
            assert "including zeros" in query["prompt"]
    options = _production_options(contract, random.Random(45))
    assert any(q["production"]["city_relevant"] for q in options)
    assert any(q["production"]["robber_relevant"] for q in options)
    assert {q["production"]["zero"] for q in options} == {True, False}
    assert contract == original


def test_path_partition_precedes_sampling_and_groups_reversals() -> None:
    membership, info = _path_pairs(45)
    again, _ = _path_pairs(45)
    assert membership == again
    assert membership != _path_pairs(46)[0]
    all_pairs = set(combinations(sorted(atlas_node_graph()), 2))
    sets = {split: set(pairs) for split, pairs in membership.items()}
    assert set.union(*sets.values()) == all_pairs
    for left, right in combinations(sets.values(), 2):
        assert not left & right
    strata = {(i["distance"], i["shortest_path_count"] > 1) for i in info.values()}
    for split, count in (("train", 128), ("validation", 64), ("test", 64)):
        targets = _select_paths(membership[split], info, count, random.Random(45))
        pairs = [tuple(sorted((t["start"], t["end"]))) for t in targets]
        assert len(set(pairs)) == count
        assert set(pairs) <= sets[split]
        assert {(info[p]["distance"], info[p]["shortest_path_count"] > 1) for p in pairs} == strata
        for target, pair in zip(targets, pairs, strict=True):
            assert tuple(sorted((target["end"], target["start"]))) == pair
            assert len(shortest_node_path(**target)) - 1 == info[pair]["distance"]


def test_corrected_bank_balances_entities_yes_no_and_rendered_choices() -> None:
    queries = _bank_queries(random.Random(45))
    assert queries == _bank_queries(random.Random(45))
    assert queries != _bank_queries(random.Random(46))
    flat = [q for pool in queries.values() for q in pool]
    assert Counter(q["task_type"] for q in flat) == BANK_QUOTAS
    positions = Counter()
    bank = spatial_query_bank()
    for query in flat:
        task = query["task_type"]
        assert any(
            q["prompt"] == query["bank_prompt"] and q["answer"] == query["answer"]
            for q in bank[task]
        )
        if task.endswith("_token"):
            choices = re.search(r": (<[NT]\d+>) or (<[NT]\d+>)\?", query["prompt"])
            assert choices is not None
            positions[(task, choices.groups().index(query["answer"]))] += 1
            assert "Output only one of the two listed tokens" in query["prompt"]
        else:
            assert "Answer only yes or no" in query["prompt"]
    assert positions == {
        (f"{entity}_direction_token", pos): 12 for entity in ("node", "tile") for pos in (0, 1)
    }


def test_existing_empty_output_is_refused_before_reading_sources(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_dataset(tmp_path, root=tmp_path / "absent")
    assert not list(tmp_path.iterdir())


def test_missing_sources_fail_without_creating_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path / "output", root=tmp_path / "absent")
    assert not (tmp_path / "output").exists()
