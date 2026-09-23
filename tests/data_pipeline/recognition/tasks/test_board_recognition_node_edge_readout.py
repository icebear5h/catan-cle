import json
import re
from pathlib import Path
from typing import Any

from data_pipeline.board_recognition.node_edge_readout import (
    EDGE_READOUT_PROMPT,
    EMPTY_SHARES,
    NODE_READOUT_PROMPT,
    board_pieces,
    empty_candidates,
    empty_quotas,
    export_node_edge_readout,
    family_tokens,
    readout_answer,
    rows_for_state,
    sample_empties,
)
from data_pipeline.board_recognition.semantics import semantic_answer, semantic_prompt

Contract = dict[str, Any]

FIXTURE_ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")
ATLAS_TOKEN = re.compile(r"^<[NETP][0-9_]+>$")
REQUIRED_KEYS = {"schema", "row_id", "curriculum_stage", "grounding_stage", "task_family", "task_type", "category", "polarity", "images", "messages", "split", "state_id", "layout_id", "density_bin", "piece_count", "entity_type", "target_token", "piece", "color", "color_heldout"}


def _fixture_state(sample_id: str) -> tuple[Contract, Contract]:
    manifest = [json.loads(line) for line in (FIXTURE_ROOT / "manifest.jsonl").read_text().splitlines()]
    state = next(row for row in manifest if row["sample_id"] == sample_id)
    return state, json.loads((FIXTURE_ROOT / state["contract_path"]).read_text())


def _prompt(row: Contract) -> str:
    return row["messages"][0]["content"].removeprefix("<image>\n")


def _answer(row: Contract) -> str:
    return row["messages"][1]["content"]


def test_readouts_walk_every_token_in_order_with_explicit_empties() -> None:
    state, contract = _fixture_state("sparse_midgame_edge_p000_base")
    tokens = family_tokens(contract)
    pieces = board_pieces(contract)
    rows = rows_for_state(state, contract, coverage="full")
    node_readout = next(row for row in rows if row["task_type"] == "node_readout")
    edge_readout = next(row for row in rows if row["task_type"] == "edge_readout")
    assert _prompt(node_readout) == NODE_READOUT_PROMPT and _prompt(edge_readout) == EDGE_READOUT_PROMPT
    node_items = _answer(node_readout).split("; ")
    edge_items = _answer(edge_readout).split("; ")
    assert [item.split(" ", 1)[0] for item in node_items] == tokens["node"] and len(node_items) == 54
    assert [item.split(" ", 1)[0] for item in edge_items] == tokens["edge"] and len(edge_items) == 72
    assert tokens["node"][0] == "<N00>" and tokens["edge"][0] == "<E00_01>" and tokens["edge"][-1] == "<E52_53>"
    for item in node_items + edge_items:
        token, value = item.split(" ", 1)
        family = "node" if token.startswith("<N") else "edge"
        assert value == (pieces[family][token]["answer"] if token in pieces[family] else "empty")
    assert sum(value != "empty" for value in (item.split(" ", 1)[1] for item in node_items)) == len(pieces["node"]) == 9
    assert sum(value != "empty" for value in (item.split(" ", 1)[1] for item in edge_items)) == len(pieces["edge"]) == 12
    assert node_readout["item_count"] == 54 and node_readout["occupied_count"] == 9 and edge_readout["item_count"] == 72
    assert readout_answer("node", ["<N00>", "<N01>"], {"<N01>": {"answer": "mystic blue city"}}) == "<N00> empty; <N01> mystic blue city"


def test_full_coverage_rows_match_the_semantic_contract() -> None:
    state, contract = _fixture_state("sparse_midgame_tile_p000_base")
    rows = rows_for_state({**state, "split": "validation"}, contract, coverage="full")
    short: Any = [row for row in rows if row["entity_type"] != "board"]
    assert len(short) == 54 + 72 and len(rows) == 54 + 72 + 2
    nodes: Any = {node["token"]: node for node in contract["nodes"]}
    edges: Any = {edge["token"]: edge for edge in contract["edges"]}
    for row in short:
        token: Any = row["target_token"]
        if row["entity_type"] == "node":
            node: Any = nodes[token]
            class_name: Any = "EMPTY" if node["building"] is None else f"{node['color']}_{node['building']}"
            assert _prompt(row) == semantic_prompt("node.occupancy", token)
            assert _answer(row) == semantic_answer("node.occupancy", class_name)
            assert row["category"] == "node.occupancy" and row["task_type"] == "node_occupancy"
        else:
            edge: Any = edges[token]
            class_name = "EMPTY" if edge["road_color"] is None else edge["road_color"]
            assert _prompt(row) == semantic_prompt("edge.owner", token)
            assert _answer(row) == semantic_answer("edge.owner", class_name)
            assert row["category"] == "edge.owner" and row["task_type"] == "edge_owner"
        assert row["polarity"] == ("hard_negative" if _answer(row) == "empty" else "positive")
        assert row["queried_token"] == token and row["piece"] in {"SETTLEMENT", "CITY", "ROAD", "EMPTY"}
    assert all(REQUIRED_KEYS <= set(row) and "spatial_targets" not in row for row in rows)
    assert all(row["schema"] == "catan_node_edge_readout_row/v1" and row["split"] == "validation" for row in rows)
    assert all(not ATLAS_TOKEN.match(_answer(row)) for row in rows)
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert {row["density_bin"] for row in rows} == {"sparse"} and {row["piece_count"] for row in rows} == {21}


def test_train_sampling_is_capped_deterministic_and_hard_first() -> None:
    state, contract = _fixture_state("dense_endgame_node_p000_base")
    rows = rows_for_state(state, contract, coverage="capped")
    again = rows_for_state(state, contract, coverage="capped")
    assert rows == again
    for family in ("node", "edge"):
        occupied = [row for row in rows if row["entity_type"] == family and row["polarity"] == "positive"]
        empties = [row for row in rows if row["entity_type"] == family and row["polarity"] == "hard_negative"]
        assert len(occupied) == 4 and len(empties) == 4
        kinds = [row["negative_kind"] for row in empties]
        # A dense board has no far empties, so the far quota falls through to the ranked leftovers.
        assert kinds.count("adjacent") >= 2 and "cross_type" in kinds and "far" not in kinds
        assert all(row["negative_distance"] == 1 for row in empties if row["negative_kind"] == "adjacent")
    assert sum(row["entity_type"] == "board" for row in rows) == 2
    assert len(rows) == 8 + 8 + 2
    smaller = rows_for_state(state, contract, coverage="capped", rows_per_family=2, readouts_per_family=2)
    assert len(smaller) == 4 + 4 + 4


def test_empty_quotas_follow_the_shares() -> None:
    assert empty_quotas(4) == {"adjacent": 2, "cross_type": 1, "hop2": 0, "hop3": 0, "far": 1}
    assert empty_quotas(20) == {"adjacent": 10, "cross_type": 3, "hop2": 2, "hop3": 1, "far": 4}
    assert sum(empty_quotas(37).values()) == 37 and abs(sum(EMPTY_SHARES.values()) - 1) < 1e-9


def test_empty_kinds_rank_touching_before_far() -> None:
    state, contract = _fixture_state("initial_placements_edge_p000_base")
    pieces = board_pieces(contract)
    node_candidates = empty_candidates(contract, "node")
    edge_candidates = empty_candidates(contract, "edge")
    assert len(node_candidates) == 54 - len(pieces["node"]) and len(edge_candidates) == 72 - len(pieces["edge"])
    node_kinds = {item["token"]: item for item in node_candidates}
    neighbors_of_building = {other for node in contract["nodes"] if node["building"] for edge in contract["edges"] if node["token"] in edge["node_tokens"] for other in edge["node_tokens"] if other != node["token"]}
    for token, item in node_kinds.items():
        if token in neighbors_of_building:
            assert item["kind"] == "adjacent" and item["distance"] == 1
    road_ends = {node for edge in contract["edges"] if edge["road_color"] for node in edge["node_tokens"]}
    assert any(node_kinds[token]["kind"] == "cross_type" for token in road_ends if token in node_kinds and node_kinds[token]["distance"] != 1)
    far = [item for item in node_candidates if item["kind"] == "far"]
    assert far and all(item["distance"] == "far" for item in far)
    chosen = sample_empties(state["sample_id"], "node", node_candidates, 4)
    assert [item["kind"] for item in chosen][:2] == ["adjacent", "adjacent"] and chosen[2]["kind"] == "cross_type" and chosen[3]["kind"] == "far"


def test_empty_board_yields_far_empties_and_all_empty_readouts() -> None:
    state, contract = _fixture_state("empty_setup_node_p000_base")
    rows = rows_for_state(state, contract, coverage="capped")
    assert not [row for row in rows if row["polarity"] == "positive" and row["entity_type"] != "board"]
    empties = [row for row in rows if row["polarity"] == "hard_negative"]
    assert len(empties) == 8 and all(row["negative_kind"] == "far" and row["negative_distance"] == "far" for row in empties)
    readouts = [row for row in rows if row["entity_type"] == "board"]
    assert all(set(item.split(" ", 1)[1] for item in _answer(row).split("; ")) == {"empty"} for row in readouts)
    assert {row["density_bin"] for row in rows} == {"empty"} and {row["piece_count"] for row in rows} == {0}


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    root.rmdir()


def test_export_splits_by_layout_with_full_coverage_evals() -> None:
    output = FIXTURE_ROOT / "node_edge_readout_test_output"
    _remove_tree(output)
    try:
        metadata: Any = export_node_edge_readout(FIXTURE_ROOT, output_dir=output, overwrite=True, validate_dataset=False)
        assert metadata["rows_per_family"] == 4 and metadata["readouts_per_family"] == 1
        assert metadata["layouts_by_split"]["train"] > 0 and metadata["layouts_by_split"]["color_diagnostic"] == 0
        assert "stage1/color_diagnostic.jsonl" not in metadata["files"]
        train = [json.loads(line) for line in (output / "stage1" / "train.jsonl").read_text().splitlines()]
        validation = [json.loads(line) for line in (output / "stage1" / "validation.jsonl").read_text().splitlines()]
        assert train and all(row["split"] == "train" for row in train)
        assert not ({row["layout_id"] for row in train} & {row["layout_id"] for row in validation})
        per_image = {}
        for row in validation:
            per_image[row["images"][0]] = per_image.get(row["images"][0], 0) + 1
        assert all(count <= 54 + 72 + 2 for count in per_image.values())
        occupied = sum(1 for row in validation if row["polarity"] == "positive" and row["entity_type"] != "board")
        empties = sum(1 for row in validation if row["polarity"] == "hard_negative")
        assert occupied and empties >= occupied and empties <= max(occupied, 8 * len(per_image))
        assert max(sum(1 for row in train if row["images"][0] == image and row["entity_type"] == "node") for image in {row["images"][0] for row in train}) <= 8
        assert all((output / "images" / row["images"][0]).is_file() for row in train[:20])
        assert metadata["files"]["stage1/validation.jsonl"]["coverage"] == "balanced" and metadata["files"]["stage1/train.jsonl"]["coverage"] == "capped"
    finally:
        _remove_tree(output)


def test_balanced_coverage_keeps_every_piece_and_matches_it_with_empties() -> None:
    state, contract = _fixture_state("sparse_midgame_edge_p000_base")
    rows = rows_for_state({**state, "split": "validation"}, contract, coverage="balanced")
    pieces = board_pieces(contract)
    for family in ("node", "edge"):
        occupied = [row for row in rows if row["entity_type"] == family and row["polarity"] == "positive"]
        empties = [row for row in rows if row["entity_type"] == family and row["polarity"] == "hard_negative"]
        assert len(occupied) == len(pieces[family]) and len(empties) == len(pieces[family])
        assert empties[0]["negative_kind"] == "adjacent"
    assert sum(row["entity_type"] == "board" for row in rows) == 2
    full = rows_for_state({**state, "split": "validation"}, contract, coverage="full")
    assert len(full) == 54 + 72 + 2 and len(rows) < len(full)
