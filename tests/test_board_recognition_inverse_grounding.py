import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from data_pipeline.board_recognition.inverse_grounding import (
    InverseGroundingError,
    inverse_audit_row,
    inverse_grounding_contract,
    inverse_ms_swift_row,
    inverse_queries_for_state,
    validate_inverse_ms_swift_row,
)
from data_pipeline.board_recognition.location_descriptions import (
    compact_node_signatures,
    inverse_location_descriptions,
)


FIXTURE_CONTRACT = Path(
    "artifacts/fixtures/board_recognition/curriculum_smoke/contracts/"
    "dense_endgame_node_p000_base.json"
)


def contract() -> dict:
    return json.loads(FIXTURE_CONTRACT.read_text())


def state() -> dict:
    return {
        "sample_id": "fixture_state",
        "split": "train",
        "image_path": "images/fixture_state.png",
        "contract_path": "contracts/fixture_state.json",
        "sha256": {"image": "a" * 64, "contract": "b" * 64},
        "source": {
            "kind": "engine_rollout",
            "game_id": None,
            "trajectory_id": "engine:train:1",
        },
        "density_bin": "dense",
        "building_count": 12,
        "road_count": 30,
    }


def test_compact_node_signatures_include_codes_words_and_uniqueness():
    signatures = compact_node_signatures(contract())

    assert len(signatures) == 54
    assert set(signatures["<N00>"]) == {
        "full",
        "resources",
        "full_words",
        "resource_words",
        "full_unique",
        "resources_unique",
    }
    assert signatures["<N00>"]["full"].count("/") == 2
    assert signatures["<N00>"]["full_words"].count("/") == 2
    assert any(
        word.startswith("wheat ")
        for row in signatures.values()
        for word in row["full_words"].split("/")
    )
    assert all("WHEAT" not in row["full"] for row in signatures.values())
    assert any("WO" in row["full"] for row in signatures.values())
    assert any("WH" in row["full"] for row in signatures.values())
    assert all("G" not in row["full"] for row in signatures.values())


def test_inverse_descriptions_do_not_leak_atlas_tokens_or_keep_collisions():
    descriptions = inverse_location_descriptions(contract())

    assert descriptions
    assert all("<" not in description and ">" not in description for description in descriptions.values())
    for prefix in "TNEP":
        values = [description for token, description in descriptions.items() if token[1] == prefix]
        assert len(values) == len(set(values))


def test_inverse_state_has_four_token_targets_and_piece_qualified_rows():
    queries = inverse_queries_for_state(state(), contract(), state_index=73)

    assert [query["entity_type"] for query in queries] == ["tile", "node", "edge", "port"]
    assert len({query["query_id"] for query in queries}) == 4
    assert all(query["prompt"].startswith("<image>\nWhere is ") for query in queries)
    assert all(query["answer"] == query["target_token"] for query in queries)
    assert queries[1]["visual_qualifier"] is not None
    assert queries[2]["visual_qualifier"] is not None


def test_inverse_rows_and_audits_pass_schemas():
    query = inverse_queries_for_state(state(), contract(), state_index=73)[1]
    row = inverse_ms_swift_row(query, image_name="fixture_state.png")
    audit = inverse_audit_row(state(), query, state_index=73)
    schema_root = Path("data/curriculum/board_recognition/schemas")

    validate_inverse_ms_swift_row(row)
    Draft202012Validator(
        json.loads((schema_root / "ms_swift_inverse_v1.schema.json").read_text())
    ).validate(row)
    Draft202012Validator(
        json.loads((schema_root / "ms_swift_inverse_audit_v1.schema.json").read_text())
    ).validate(audit)


def test_inverse_row_rejects_token_leakage_and_multi_token_answers():
    query = inverse_queries_for_state(state(), contract(), state_index=73)[1]
    row = inverse_ms_swift_row(query, image_name="fixture_state.png")
    row["messages"][0]["content"] = "<image>\nWhere is <N17>?"
    with pytest.raises(InverseGroundingError, match="leaks an atlas token"):
        validate_inverse_ms_swift_row(row)

    row = inverse_ms_swift_row(query, image_name="fixture_state.png")
    row["messages"][1]["content"] = "<N17> <N18>"
    with pytest.raises(InverseGroundingError, match="exactly one"):
        validate_inverse_ms_swift_row(row)


def test_inverse_contract_records_alias_and_mix_policy():
    payload = inverse_grounding_contract()

    assert "O10/WH5/S6" in payload["node_surface_forms"]
    assert "ore 10/wheat 5/sheep 6" in payload["node_surface_forms"]
    assert payload["mixed_projection"]["ratio"] == "2:1 forward:inverse"
    assert payload["answer_format"].startswith("exactly one")
