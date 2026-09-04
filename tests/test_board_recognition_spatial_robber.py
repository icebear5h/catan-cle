import json
from pathlib import Path

from data_pipeline.board_recognition.spatial_robber import (
    CURRICULUM_STAGES,
    build_curriculum_smoke_rows,
    robber_queries_for_state,
    spatial_queries_for_state,
    spatial_query_bank,
)


FIXTURE_ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")


def _fixture_state_and_contract():
    manifest = [
        json.loads(line) for line in (FIXTURE_ROOT / "manifest.jsonl").read_text().splitlines()
    ]
    state = dict(next(row for row in manifest if row["stage"] == "empty_setup"))
    state["density_bin"] = "empty"
    contract = json.loads((FIXTURE_ROOT / state["contract_path"]).read_text())
    return state, contract


def test_spatial_bank_covers_directions_topology_and_hard_negatives():
    bank = spatial_query_bank()
    rows = [row for pool in bank.values() for row in pool]
    relationships = {row["relationship"] for row in rows}
    tokens = {token for row in rows for token in row["tokens"]}

    assert {"above", "below", "left_of", "right_of", "adjacent", "connected"} <= relationships
    assert {row["polarity"] for row in rows} == {"positive", "hard_negative", "token_return"}
    assert sum(token.startswith("<N") for token in tokens) == 54
    assert sum(token.startswith("<T") for token in tokens) == 19


def test_empty_state_gets_balanced_24_row_spatial_block():
    state, _ = _fixture_state_and_contract()
    rows = spatial_queries_for_state(state, state_index=0)

    assert len(rows) == 24
    assert sum(row["answer"] == "yes" for row in rows) == 10
    assert sum(row["answer"] == "no" for row in rows) == 10
    assert sum(row["answer"].startswith("<") for row in rows) == 4
    assert {row["curriculum_stage"] for row in rows} == {"spatial_grounding"}


def test_robber_rows_include_positive_negative_and_token_localization():
    state, contract = _fixture_state_and_contract()
    rows = robber_queries_for_state(state, contract, state_index=0)
    robber = next(tile["token"] for tile in contract["tiles"] if tile["has_robber"])

    assert [row["answer"] for row in rows] == ["yes", "no", robber]
    assert rows[1]["tokens"] != [robber]
    assert rows[2]["prompt"] == "Where is the robber? Answer with one tile token."
    assert {row["curriculum_stage"] for row in rows} == {"clean_board_grounding"}


def test_generated_curriculum_smoke_has_eight_ordered_rows_per_stage():
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
