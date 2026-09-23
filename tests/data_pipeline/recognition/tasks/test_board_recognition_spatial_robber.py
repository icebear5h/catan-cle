import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from data_pipeline.board_recognition.spatial_robber import (
    CURRICULUM_STAGES,
    build_curriculum_smoke_rows,
    robber_queries_for_state,
    spatial_queries_for_state,
    spatial_query_bank,
)

Contract = dict[str, Any]

FIXTURE_ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")


def _fixture_state_and_contract() -> tuple[Contract, Contract]:
    manifest = [
        json.loads(line) for line in (FIXTURE_ROOT / "manifest.jsonl").read_text().splitlines()
    ]
    state = dict(next(row for row in manifest if row["stage"] == "empty_setup"))
    state["density_bin"] = "empty"
    contract = json.loads((FIXTURE_ROOT / state["contract_path"]).read_text())
    return state, contract


def test_spatial_bank_covers_directions_topology_and_hard_negatives() -> None:
    bank = spatial_query_bank()
    rows = [row for pool in bank.values() for row in pool]
    relationships = {row["relationship"] for row in rows}
    tokens: Any = {token for row in rows for token in row["tokens"]}

    assert {"above", "below", "left_of", "right_of", "adjacent", "connected"} <= relationships
    assert {row["polarity"] for row in rows} == {"positive", "hard_negative", "token_return"}
    assert sum(token.startswith("<N") for token in tokens) == 54
    assert sum(token.startswith("<T") for token in tokens) == 19


@pytest.mark.parametrize("entity", ["node", "tile"])
def test_direction_choices_balance_inverse_pairs_and_cyclic_samples(entity: str) -> None:
    rows: Any = spatial_query_bank()[f"{entity}_direction_token"]
    answer_positions: Any = []
    positions_by_relation: Any = defaultdict(set)
    for row in rows:
        choices: Any = row["prompt"].rsplit(": ", 1)[1].removesuffix("?").split(" or ")
        assert choices == row["tokens"]
        assert len(set(choices)) == 2
        position: Any = choices.index(row["answer"])
        answer_positions.append(position)
        positions_by_relation[row["relationship"]].add(position)

    assert len(rows) % 2 == 0
    for forward, inverse in zip(rows[::2], rows[1::2], strict=True):
        assert forward["tokens"] == inverse["tokens"]
        assert forward["answer"] != inverse["answer"]
    assert set(positions_by_relation) == {"above", "below", "left_of", "right_of"}
    assert all(positions == {0, 1} for positions in positions_by_relation.values())
    # State sampling takes two consecutive rows, including cross-pair/wraparound starts.
    for start, position in enumerate(answer_positions):
        assert {position, answer_positions[(start + 1) % len(rows)]} == {0, 1}


@pytest.mark.parametrize("state_index", [0, 1, 17])
def test_empty_state_gets_balanced_24_row_spatial_block(state_index: int) -> None:
    state, _ = _fixture_state_and_contract()
    rows: Any = spatial_queries_for_state(state, state_index=state_index)

    assert rows == spatial_queries_for_state(state, state_index=state_index)
    assert len(rows) == 24
    assert sum(row["answer"] == "yes" for row in rows) == 10
    assert sum(row["answer"] == "no" for row in rows) == 10
    assert sum(row["answer"].startswith("<") for row in rows) == 4
    assert {row["curriculum_stage"] for row in rows} == {"spatial_grounding"}
    for entity in ("node", "tile"):
        choices: Any = [row for row in rows if row["task_type"] == f"{entity}_direction_token"]
        assert sorted(row["tokens"].index(row["answer"]) for row in choices) == [0, 1]


def test_robber_rows_include_positive_negative_and_token_localization() -> None:
    state, contract = _fixture_state_and_contract()
    rows = robber_queries_for_state(state, contract, state_index=0)
    robber = next(tile["token"] for tile in contract["tiles"] if tile["has_robber"])

    assert [row["answer"] for row in rows] == ["yes", "no", robber]
    assert rows[1]["tokens"] != [robber]
    assert rows[2]["prompt"] == "Where is the robber? Answer with one tile token."
    assert {row["curriculum_stage"] for row in rows} == {"clean_board_grounding"}


def test_generated_curriculum_smoke_has_eight_ordered_rows_per_stage() -> None:
    dataset_root = Path("artifacts/generated/board_recognition/replay_v1")
    supplement_root = dataset_root / "spatial_robber_v1"
    if not supplement_root.is_dir():
        return
    rows = [json.loads(line) for line in (supplement_root / "train.jsonl").read_text().splitlines()]
    audits = [
        json.loads(line)
        for line in (supplement_root / "audit" / "train.jsonl").read_text().splitlines()
    ]

    smoke = build_curriculum_smoke_rows(dataset_root, rows, audits)

    assert len(smoke) == 32
    assert [row["curriculum_stage"] for row in smoke[::8]] == list(CURRICULUM_STAGES)
