import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from data_pipeline.board_recognition.replay_dataset import (
    BoardStateCandidate,
    board_density,
    dense_labels,
    generate_engine_trajectory_candidates,
    select_split_candidates,
    split_replay_games,
)
from evals.catan_board_bench.builder import CatanObservationSuite

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def candidate(trajectory: str, index: int, density: str) -> BoardStateCandidate:
    return BoardStateCandidate(
        trajectory_id=trajectory,
        board_fact_sha256=f"{index:064x}",
        board_map_sha256=f"{index // 100:064x}",
        density_bin=density,
        building_count=0,
        road_count=0,
        contract={},
        source={},
    )


def test_game_split_is_deterministic_complete_and_disjoint() -> None:
    sources: Any = [{"game_id": str(1000 + index)} for index in range(53)]

    first = split_replay_games(sources, seed=44)
    second = split_replay_games(sources, seed=44)

    assert first == second
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 43,
        "validation": 5,
        "test": 5,
    }
    split_ids = [{row["game_id"] for row in first[name]} for name in first]
    assert len(set.union(*split_ids)) == 53
    assert all(
        not left & right for index, left in enumerate(split_ids) for right in split_ids[index + 1 :]
    )


def test_state_selection_caps_trajectories_and_spans_density_bins() -> None:
    rows = [
        candidate(f"game:{game}", game * 100 + index, density)
        for game in range(4)
        for index, density in enumerate(("empty", "setup", "sparse", "dense") * 5)
    ]

    selected = select_split_candidates(
        rows,
        target=32,
        per_trajectory_cap=8,
        seed=91,
    )

    assert len(selected) == 32
    assert len({row.board_fact_sha256 for row in selected}) == 32
    assert Counter(row.trajectory_id for row in selected) == {
        f"game:{game}": 8 for game in range(4)
    }
    assert set(row.density_bin for row in selected) == {
        "empty",
        "setup",
        "sparse",
        "dense",
    }


def test_dense_labels_are_complete_without_stage_or_counterfactual_fields() -> None:
    engine = GameEngine(COLORS, seed=12, shuffle_players=False)
    contract = CatanObservationSuite().public_board_contract(engine)

    labels: Any = dense_labels(contract, sample_id="engine-state")

    assert set(labels) == {"schema", "sample_id", "entities"}
    assert len(labels["entities"]["tiles"]) == 19
    assert len(labels["entities"]["nodes"]) == 54
    assert len(labels["entities"]["edges"]) == 72
    assert len(labels["entities"]["ports"]) == 9
    assert board_density(contract) == (0, 0, "empty")


def test_engine_trajectory_uses_unique_legal_board_changes() -> None:
    rows: Any = generate_engine_trajectory_candidates(
        trajectory_index=0,
        seed=712,
        split="train",
        max_actions=2_000,
    )

    assert len(rows) > 20
    assert len({row.board_fact_sha256 for row in rows}) == len(rows)
    assert all(row.source["legal_actions_only"] for row in rows)
    assert all(row.source["kind"] == "engine_rollout" for row in rows)
    assert {row.density_bin for row in rows}.issuperset({"empty", "setup", "sparse"})
    assert len(rows[0].contract["players"]) == 4


def test_engine_trajectory_is_stable_across_process_hash_seeds() -> None:
    script = """
import json
from data_pipeline.board_recognition.replay_dataset import generate_engine_trajectory_candidates
rows = generate_engine_trajectory_candidates(
    trajectory_index=7,
    seed=381427,
    split='train',
    max_actions=300,
)
print(json.dumps([(row.board_fact_sha256, row.source['engine_action_count']) for row in rows]))
"""
    outputs = []
    for hash_seed in ("1", "777"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]


def test_v2_schemas_are_valid_json_documents() -> None:
    schema_dir = Path("data/curriculum/board_recognition/schemas")
    for name in ("manifest_v2.schema.json", "dense_labels_v2.schema.json"):
        schema = json.loads((schema_dir / name).read_text())
        assert schema["$schema"].endswith("2020-12/schema")
