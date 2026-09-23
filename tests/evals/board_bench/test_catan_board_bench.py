import json
from pathlib import Path

from cle.game_engine.game import GameEngine
from cle.game_engine.models.map import NUM_EDGES, NUM_NODES, NUM_TILES
from cle.game_engine.models.player import Color
from evals.catan_board_bench.builder import CatanObservationSuite
from evals.catan_board_bench.hex_direction_probe import (
    DIRECTION_ORDER,
    SCREEN_DIRECTIONS,
    UNIT_VECTORS,
    build_probe,
    cube_to_pixel,
)
from evals.catan_board_bench.scoring import (
    PROBE_CATEGORIES,
    normalize_text,
    score_answer,
    select_questions,
)
from evals.catan_board_bench.text_representations import (
    REPRESENTATION_NAMES,
    fact_digest,
    parse_representation,
    public_board_facts,
    render_representation,
)


def test_public_board_contract_has_complete_atlas_and_no_hidden_hands() -> None:
    game = GameEngine(
        [
            Color.RED,
            Color.BLUE,
            Color.WHITE,
            Color.ORANGE,
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


def test_public_board_qa_rows_are_engine_scored_and_promptable() -> None:
    game = GameEngine(
        [
            Color.RED,
            Color.BLUE,
            Color.WHITE,
            Color.ORANGE,
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


def test_text_representations_round_trip_one_target_neutral_fact_set() -> None:
    contract_dir = Path("evals/catan_board_bench/datasets/catan_board_bench_100/contracts")
    for sample_index in range(10):
        contract = json.loads((contract_dir / f"sample_{sample_index:03d}.json").read_text())
        facts = public_board_facts(contract)
        outputs = {name: render_representation(name, facts) for name in REPRESENTATION_NAMES}

        assert len(fact_digest(facts)) == 64
        assert set(outputs) == {
            "verbose_json",
            "compact_json",
            "graph_dsl",
            "spatial_ascii",
        }
        assert "players" not in facts
        assert "settlement_count" not in json.dumps(facts)
        assert "road_count" not in json.dumps(facts)
        assert sum(tile["has_robber"] for tile in facts["tiles"]) == 1
        assert all(node["building"] for node in facts["nodes"])
        assert all(edge["road_color"] for edge in facts["edges"])

        for name, rendered in outputs.items():
            parsed = parse_representation(name, rendered)
            assert parsed == facts
            assert fact_digest(parsed) == fact_digest(facts)
            for tile in facts["tiles"]:
                assert tile["token"] in rendered
            for node in facts["nodes"]:
                assert node["token"] in rendered
            for edge in facts["edges"]:
                assert edge["token"] in rendered
            for port in facts["ports"]:
                assert port["token"] in rendered

        assert len(outputs["compact_json"]) < len(outputs["verbose_json"])
        assert len(outputs["graph_dsl"]) < len(outputs["verbose_json"])


def test_probe_rows_select_without_contracts(tmp_path: Path) -> None:
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

    assert [row["category"] for row in selected] == [
        "isolated_road_owner",
        "isolated_node_occupancy",
    ]
    assert {row["id"] for row in selected} == {"green_city_probe", "mystic_road_probe"}
    assert all("contract" not in row for row in selected)


def test_hex_direction_probe_is_balanced_and_uses_engine_projection(tmp_path: Path) -> None:
    metadata = build_probe(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "questions/qa.jsonl").read_text().splitlines()]

    assert metadata["qa_count"] == 72
    assert metadata["chance_accuracy"] == 1 / 6
    assert {row["category"] for row in rows} == {
        "isolated_hex_direction_to_label",
        "isolated_hex_label_to_direction",
    }
    assert all(row["contract_path"] is None for row in rows)

    selected = select_questions(
        tmp_path,
        question_dir=tmp_path / "questions",
        categories=[
            "isolated_hex_direction_to_label",
            "isolated_hex_label_to_direction",
        ],
        limit_samples=0,
        questions_per_sample=0,
        max_requests=None,
        selection_mode="flat",
    )
    assert len(selected) == 72
    assert {
        category: sum(row["category"] == category for row in selected)
        for category in {row["category"] for row in selected}
    } == {
        "isolated_hex_direction_to_label": 36,
        "isolated_hex_label_to_direction": 36,
    }

    direction_counts = {}
    label_direction_pairs = set()
    for row in rows:
        target = row["target"]
        direction_counts[target["direction"]] = direction_counts.get(target["direction"], 0) + 1
        label_direction_pairs.add((target["neighbor_label"], target["direction"]))
    assert set(direction_counts.values()) == {12}
    assert len(label_direction_pairs) == 36

    center = (256.0, 256.0)
    pixels = {
        SCREEN_DIRECTIONS[direction]: cube_to_pixel(
            UNIT_VECTORS[direction], center=center, radius=72
        )
        for direction in DIRECTION_ORDER
    }
    assert pixels["LEFT"][0] < center[0] and pixels["LEFT"][1] == center[1]
    assert pixels["RIGHT"][0] > center[0] and pixels["RIGHT"][1] == center[1]
    assert pixels["UP-LEFT"][0] < center[0] and pixels["UP-LEFT"][1] < center[1]
    assert pixels["UP-RIGHT"][0] > center[0] and pixels["UP-RIGHT"][1] < center[1]
    assert pixels["DOWN-LEFT"][0] < center[0] and pixels["DOWN-LEFT"][1] > center[1]
    assert pixels["DOWN-RIGHT"][0] > center[0] and pixels["DOWN-RIGHT"][1] > center[1]


def test_hex_direction_scoring_accepts_semantic_aliases_only() -> None:
    direction_qa = {
        "category": "isolated_hex_label_to_direction",
        "answer": "UP-LEFT",
        "target": {"direction": "UP-LEFT", "neighbor_label": "4"},
    }
    label_qa = {
        "category": "isolated_hex_direction_to_label",
        "answer": "4",
        "target": {"direction": "UP-LEFT", "neighbor_label": "4"},
    }

    assert score_answer(direction_qa, "north-west")["correct"]
    assert score_answer(direction_qa, "upper left")["correct"]
    assert not score_answer(direction_qa, "left")["correct"]
    assert not score_answer(direction_qa, "not upper left")["correct"]
    assert not score_answer(direction_qa, "no, up-left")["correct"]
    assert not score_answer(direction_qa, "never up-left")["correct"]
    assert not score_answer(direction_qa, "can't be up-left")["correct"]
    assert not score_answer(direction_qa, "wasn't up-left")["correct"]
    assert not score_answer(direction_qa, "up-left or up-right")["correct"]
    assert not score_answer(direction_qa, "left or right")["correct"]
    assert score_answer(label_qa, "Hex 4")["correct"]
    assert not score_answer(label_qa, "not 4")["correct"]
    assert not score_answer(label_qa, "no, 4")["correct"]
    assert not score_answer(label_qa, "can't be 4")["correct"]
    assert not score_answer(label_qa, "anything except 4")["correct"]
    assert not score_answer(label_qa, "4 or 5")["correct"]


def test_probe_scoring_accepts_extended_colonist_colors() -> None:
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
