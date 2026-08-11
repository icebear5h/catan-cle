import json

from catanbench.builder import CatanObservationSuite
from catanbench.scoring import (
    PROBE_CATEGORIES,
    normalize_text,
    score_answer,
    select_questions,
)
from engine.game import Game
from engine.models.map import NUM_EDGES, NUM_NODES, NUM_TILES
from engine.models.player import Color, SimplePlayer


def test_public_board_contract_has_complete_atlas_and_no_hidden_hands():
    game = Game(
        [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ],
        seed=123,
        shuffle_players=False,
    )

    contract = CatanObservationSuite().public_board_contract(
        game,
        sample={"id": "sample_test", "index": 0},
        source={"kind": "unit_test"},
    )

    assert contract["schema"] == "catan_public_board_contract/v0"
    assert len(contract["tiles"]) == NUM_TILES
    assert len(contract["nodes"]) == NUM_NODES
    assert len(contract["edges"]) == NUM_EDGES
    assert len(contract["ports"]) == 9

    robber_tiles = [tile for tile in contract["tiles"] if tile["has_robber"]]
    assert len(robber_tiles) == 1
    assert robber_tiles[0]["id"] == contract["robber"]["tile_id"]

    for player in contract["players"]:
        assert "resource_freqdeck" not in player
        assert "development_cards" not in player


def test_public_board_qa_rows_are_engine_scored_and_promptable():
    game = Game(
        [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ],
        seed=123,
        shuffle_players=False,
    )
    suite = CatanObservationSuite()
    contract = suite.public_board_contract(
        game,
        sample={
            "id": "sample_test",
            "index": 0,
            "contract_path": "contracts/sample_test.json",
            "image_path": "images/sample_test.png",
        },
        source={"kind": "unit_test"},
    )

    qas = suite.qa_pairs(contract)

    assert len(qas) == 30
    assert {qa["category"] for qa in qas}.issuperset(
        {
            "robber_tile",
            "robber_resource_number",
            "tile_has_robber",
            "tile_resource_number",
            "node_occupancy",
            "edge_road_owner",
            "color_road_locations",
            "port_trade_type",
            "port_type_nodes",
            "port_occupancy",
            "nodes_connected",
            "edge_connects_nodes",
            "color_building_counts",
            "color_road_count",
        }
    )
    assert all(qa["question"] for qa in qas)
    assert all(qa["answer"] for qa in qas)
    assert all(qa["target"] for qa in qas)


def test_probe_rows_select_without_contracts(tmp_path):
    question_dir = tmp_path / "questions"
    question_dir.mkdir()
    rows = [
        {
            "id": "green_city_probe",
            "sample_id": "green_city_probe",
            "category": "isolated_node_occupancy",
            "question": "What building, if any, is shown on this node?",
            "answer": "<GREEN> <CITY>",
            "image_path": "images/green_city_probe.png",
            "contract_path": None,
            "target": {"color": "GREEN", "building": "CITY"},
        },
        {
            "id": "mystic_road_probe",
            "sample_id": "mystic_road_probe",
            "category": "isolated_road_owner",
            "question": "Who owns the road shown?",
            "answer": "<MYSTIC_BLUE>",
            "image_path": "images/mystic_road_probe.png",
            "contract_path": None,
            "target": {"color": "MYSTIC_BLUE"},
        },
    ]
    (question_dir / "qa.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    selected = select_questions(
        tmp_path,
        question_dir=question_dir,
        categories=PROBE_CATEGORIES,
        limit_samples=10,
        questions_per_sample=len(PROBE_CATEGORIES),
        max_requests=None,
        selection_mode="flat",
    )

    assert [row["category"] for row in selected] == ["isolated_road_owner", "isolated_node_occupancy"]
    assert {row["id"] for row in selected} == {"green_city_probe", "mystic_road_probe"}
    assert all("contract" not in row for row in selected)


def test_probe_scoring_accepts_extended_colonist_colors():
    green_city = {
        "category": "isolated_node_occupancy",
        "answer": "<GREEN> <CITY>",
        "target": {"color": "GREEN", "building": "CITY"},
    }
    mystic_road = {
        "category": "isolated_road_owner",
        "answer": "<MYSTIC_BLUE>",
        "target": {"color": "MYSTIC_BLUE"},
    }

    assert normalize_text("green city") == "<GREEN> <CITY>"
    assert normalize_text("mystic blue") == "<MYSTIC_BLUE>"
    assert score_answer(green_city, "green city")["correct"]
    assert score_answer(mystic_road, "mystic blue")["correct"]
