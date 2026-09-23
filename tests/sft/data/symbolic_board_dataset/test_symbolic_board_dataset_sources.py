"""Source contract auditing, physical-state admission and directional partitioning."""

import copy
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

from data_pipeline.board_recognition.sources import file_sha256
from sft.board.symbolic_board_tasks import (
    PhysicalStateError,
    atlas_geometry,
    detached_board,
    symbolic_answer,
    validate_contract,
)
from sft.json_types import JsonValue
from sft.scripts.builders.build_symbolic_board_dataset import (
    DEFAULT_OUTPUT,
    DEFAULT_ROOT,
    audit_source_state,
    build_dataset,
    directional_pair_manifest,
    static_queries,
)

from .support import RealExample


def test_real_contract_readout_dense_smoke_and_no_source_mutation(real_example: RealExample) -> None:
    source, readout, contract = real_example
    before = copy.deepcopy((source, readout, contract))
    digest = file_sha256(DEFAULT_ROOT / source["contract_path"])
    state, provenance = audit_source_state(DEFAULT_ROOT, source, readout)
    assert state == validate_contract(contract)
    assert len(state["board"].split(";")) == 155
    assert len(state["colors"]) == 4
    assert set(provenance["paths"]) == {"contract", "labels"}
    board = detached_board(state)
    assert len(board.map.land_nodes) == 54
    assert state["board"] == readout["messages"][-1]["content"]
    assert (source, readout, contract) == before
    assert file_sha256(DEFAULT_ROOT / source["contract_path"]) == digest


@pytest.mark.parametrize("family,field,value", [
    ("tiles", "coord", [True, 0, 0]),
    ("tiles", "number", "6"),
    ("tiles", "node_tokens", ["<N00>"] * 6),
    ("tiles", "edges", [[0, 2]] * 6),
    ("nodes", "token", "<N01>"),
    ("nodes", "adjacent_edges", [[0, 2]]),
    ("nodes", "port_ids", [8]),
    ("edges", "nodes", [0, 2]),
    ("edges", "road_color", "NOT_A_PLAYER"),
    ("ports", "attached_nodes", [0, 1]),
    ("ports", "direction", "NORTH"),
])
def test_contract_exact_topology_and_identity_rejected(real_example: RealExample, family: str, field: str,
                                              value: JsonValue) -> None:
    contract = copy.deepcopy(real_example[2])
    contract[family][0][field] = value
    with pytest.raises(ValueError):
        validate_contract(contract)


def test_duplicate_ownership_and_source_hash_checks(real_example: RealExample) -> None:
    source, readout, original = real_example
    contract = copy.deepcopy(original)
    contract["nodes"][1] = copy.deepcopy(contract["nodes"][0])
    with pytest.raises(ValueError):
        validate_contract(contract)
    contract = copy.deepcopy(original)
    contract["nodes"][0]["color"] = contract["players"][0]["color"]
    with pytest.raises(ValueError, match="ownership"):
        validate_contract(contract)
    changed = copy.deepcopy(source)
    changed["sha256"]["contract"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        audit_source_state(DEFAULT_ROOT, changed, readout)
    changed = copy.deepcopy(readout)
    changed["messages"][-1]["content"] += " junk"
    with pytest.raises(ValueError, match="answer"):
        audit_source_state(DEFAULT_ROOT, source, changed)
    # Derived source road caches may be historically stale; never use them as gold.
    changed = copy.deepcopy(original)
    changed["achievements"]["longest_road"]["length"] = 999
    changed["players"][0]["longest_road_length"] = 999
    assert validate_contract(changed) == validate_contract(original)


def test_source_physical_supply_and_distance_are_not_silently_repaired(real_example: RealExample) -> None:
    original = real_example[2]
    contract = copy.deepcopy(original)
    color = contract["players"][0]["color"]
    for node in contract["nodes"][:2]:
        node.update(building="SETTLEMENT", color=color,
                    building_token="<SETTLEMENT>", color_token=f"<{color}>")
    with pytest.raises(PhysicalStateError, match="distance"):
        validate_contract(contract)
    contract = copy.deepcopy(original)
    for edge in contract["edges"][:16]:
        edge.update(road_color=color, road_color_token=f"<{color}>")
    with pytest.raises(PhysicalStateError, match="supply"):
        validate_contract(contract)


def test_direction_pair_partition_covers_all_and_balances_choices_and_inverses() -> None:
    atlas = atlas_geometry()
    pairs = directional_pair_manifest(46)
    sets = {s: {tuple(p) for p in values} for s, values in pairs.items()}
    all_pairs = {p for family in "NT" for p in combinations(sorted(t for t in atlas["positions"] if t[1] == family), 2)}
    assert set.union(*sets.values()) == all_pairs
    for a, b in combinations(sets, 2):
        assert not sets[a] & sets[b]
    for split, assigned in pairs.items():
        for task, count in (("symbolic_direction", 400), ("symbolic_direction_choice", 400)):
            queries = static_queries(task, assigned, count, random.Random(46))
            assert queries == static_queries(task, assigned, count, random.Random(46))
            bins = Counter()
            for query in queries:
                assert tuple(sorted((query["a"], query["b"]))) in sets[split]
                gold = symbolic_answer(task, {"state": None, "query": query})
                bins[(query["a"][1], query["choices"].index(gold) if task.endswith("choice") else gold)] += 1
                inverse = dict(query, a=query["b"], b=query["a"])
                inverse["direction"] = {"left": "right", "right": "left", "above": "below", "below": "above"}[query["direction"]]
                inverse_gold = symbolic_answer(task, {"state": None, "query": inverse})
                assert inverse_gold == (gold if not task.endswith("choice") else query["b"] if gold == query["a"] else query["a"])
            assert set(bins.values()) == {100}


def test_immutable_output_and_parent_precondition(tmp_path: Path) -> None:
    assert DEFAULT_OUTPUT.name == "symbolic_board_v2"
    with pytest.raises(FileExistsError):
        build_dataset(tmp_path)
    destination = tmp_path / "missing-parent" / "dataset"
    with pytest.raises(FileNotFoundError, match="parent"):
        build_dataset(destination, dry_run=True)
    assert not destination.parent.exists()
