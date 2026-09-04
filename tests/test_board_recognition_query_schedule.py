import json
from collections import Counter
from pathlib import Path

from PIL import Image

from data_pipeline.board_recognition.dataset import BoardRecognitionStateDataset
from data_pipeline.board_recognition.query_schedule import (
    build_split_query_plan,
    query_plan_metrics,
    validate_split_balance,
)
from data_pipeline.board_recognition.replay_dataset import DATASET_SCHEMA
from data_pipeline.board_recognition.replay_sft import atomic_prompt, qwen_row, validate_qwen_row
from evals.catan_board_bench.tokens import base_edges, edge_token
from cle.game_engine.models.player import Color


COLORS = [color.value for color in Color]
NODE_CLASSES = [f"{color}_{building}" for color in COLORS for building in ("SETTLEMENT", "CITY")]
EDGE_CLASSES = COLORS


def dense_labels(state_id: str) -> dict:
    resources = ["DESERT", "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    numbers = [None, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    port_types = [
        "THREE_TO_ONE",
        "TWO_TO_ONE_WOOD",
        "TWO_TO_ONE_BRICK",
        "TWO_TO_ONE_SHEEP",
        "TWO_TO_ONE_WHEAT",
        "TWO_TO_ONE_ORE",
    ]
    return {
        "schema": "catan_board_recognition_dense_labels/v2",
        "sample_id": state_id,
        "entities": {
            "tiles": [
                {
                    "id": f"<T{index:02d}>",
                    "resource": resources[index % len(resources)],
                    "number": numbers[index % len(numbers)],
                    "robber": index == 0,
                }
                for index in range(19)
            ],
            "nodes": [
                {
                    "id": f"<N{index:02d}>",
                    "occupancy": NODE_CLASSES[index] if index < len(NODE_CLASSES) else "EMPTY",
                }
                for index in range(54)
            ],
            "edges": [
                {
                    "id": edge_token(edge),
                    "owner": EDGE_CLASSES[index] if index < len(EDGE_CLASSES) else "EMPTY",
                }
                for index, edge in enumerate(base_edges())
            ],
            "ports": [
                {
                    "id": f"<P{index:02d}>",
                    "port_type": port_types[index % len(port_types)],
                }
                for index in range(9)
            ],
        },
    }


def make_schedule_fixture(root: Path, *, states: int = 88) -> list[dict]:
    (root / "dense_labels").mkdir(parents=True)
    (root / "images").mkdir()
    Image.new("RGB", (64, 64), "white").save(root / "images" / "board.png")
    rows = []
    for index in range(states):
        state_id = f"state_{index:03d}"
        label_path = f"dense_labels/{state_id}.json"
        (root / label_path).write_text(json.dumps(dense_labels(state_id)))
        rows.append(
            {
                "sample_id": state_id,
                "split": "train",
                "image_path": "images/board.png",
                "label_path": label_path,
                "density_bin": "dense",
                "source": {
                    "kind": "engine_rollout",
                    "game_id": None,
                    "trajectory_id": f"engine:train:{index}",
                },
            }
        )
    return rows


def test_split_scheduler_is_atomic_balanced_complete_and_deterministic(tmp_path: Path):
    states = make_schedule_fixture(tmp_path)

    first = build_split_query_plan(
        states,
        dataset_dir=tmp_path,
        split="train",
        seed=77,
        queries_per_state=8,
        epoch_index=0,
    )
    second = build_split_query_plan(
        states,
        dataset_dir=tmp_path,
        split="train",
        seed=77,
        queries_per_state=8,
        epoch_index=0,
    )
    metrics = validate_split_balance(first, split="train")

    assert first == second
    assert metrics["rows"] == 88 * 8
    assert metrics["duplicates"] == 0
    assert metrics["entity_counts"] == {
        "edge": 176,
        "node": 176,
        "port": 176,
        "tile": 176,
    }
    assert metrics["polarity_counts"] == {
        "edge.owner.empty": 88,
        "edge.owner.positive": 88,
        "node.occupancy.empty": 88,
        "node.occupancy.positive": 88,
    }
    assert {head: len(counts) for head, counts in metrics["slot_counts"].items()} == {
        "edge.owner": 72,
        "node.occupancy": 54,
        "port.port_type": 9,
        "tile.number": 19,
        "tile.resource": 19,
        "tile.robber": 19,
    }
    node_counts = [
        metrics["class_counts"][f"node.occupancy.{class_name}"] for class_name in NODE_CLASSES
    ]
    edge_counts = [
        metrics["class_counts"][f"edge.owner.{class_name}"] for class_name in EDGE_CLASSES
    ]
    assert set(node_counts) == {4}
    assert set(edge_counts) == {8}


def test_epoch_identity_changes_the_plan_without_changing_gates(tmp_path: Path):
    states = make_schedule_fixture(tmp_path)
    epoch_zero = build_split_query_plan(
        states,
        dataset_dir=tmp_path,
        split="train",
        seed=77,
        queries_per_state=8,
        epoch_index=0,
    )
    epoch_one = build_split_query_plan(
        states,
        dataset_dir=tmp_path,
        split="train",
        seed=77,
        queries_per_state=8,
        epoch_index=1,
    )

    assert epoch_zero != epoch_one
    assert query_plan_metrics(epoch_zero)["rows"] == query_plan_metrics(epoch_one)["rows"]
    validate_split_balance(epoch_one, split="train")


def test_state_dataset_loads_one_image_with_eight_scheduled_queries(tmp_path: Path):
    rows = make_schedule_fixture(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps({"schema": DATASET_SCHEMA}))
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = BoardRecognitionStateDataset(tmp_path, split="train")
    item = dataset[0]

    assert len(dataset) == 88
    assert item["image"].size == (64, 64)
    assert len(item["queries"]) == 8
    assert Counter(query["entity_type"] for query in item["queries"]) == {
        "tile": 2,
        "node": 2,
        "edge": 2,
        "port": 2,
    }


def test_qwen_projection_has_one_image_two_query_tokens_and_one_answer_token(tmp_path: Path):
    states = make_schedule_fixture(tmp_path)
    plan = build_split_query_plan(
        states,
        dataset_dir=tmp_path,
        split="train",
        seed=77,
        queries_per_state=8,
    )[0]
    query = plan["queries"][0]

    row = qwen_row(query, image_name="board.png")
    validate_qwen_row(row)

    assert row["conversations"][0]["value"] == f"<image>\n{atomic_prompt(query)}"
    assert atomic_prompt(query) == f"{query['slot']}{query['query_token']}"
    assert row["conversations"][1]["value"] == query["answer_token"]
