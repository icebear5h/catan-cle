"""Full-build counts, emitted row schema and whole-dataset validation."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from evals.catan_board_bench.tokens import atlas_tokens
from sft.board.symbolic_board_tasks import (
    STATIC_TASKS,
    TRAIN_TASKS,
    TRANSFER_TASKS,
    score_symbolic_task,
)
from sft.scripts.builders.build_symbolic_board_dataset import (
    DEFAULT_ROOT,
    SOURCE_COUNTS,
    TRAIN_QUOTAS,
    directional_pair_manifest,
    read_json,
    read_jsonl,
    validate_rows,
)

from .support import BuiltDataset


def test_full_real_build_counts_split_exclusions_and_transfer_only(built_dataset: BuiltDataset) -> None:
    output, result, verified, metadata, files = built_dataset
    assert verified["valid"] and result["counts"] == metadata["counts"]
    assert len(files["train"]) == 3200
    assert len(files["validation"]) == len(files["test"]) == 480
    assert Counter(r["task_type"] for r in files["train"]) == TRAIN_QUOTAS
    assert metadata["static_training_presentations"] == metadata["state_training_presentations"] == 1600
    audit = metadata["source_audit"]
    assert audit["source_counts"] == SOURCE_COUNTS
    engine_sources = {k: v for k, v in audit["source_hashes"].items() if k.startswith("engine_board")}
    assert {Path(v["path"]).name for v in engine_sources.values()} == {
        "core.py", "graph.py", "roads.py", "__init__.py"}
    assert all(Path(v["path"]).parent.name == "board" for v in engine_sources.values())
    assert len(audit["benchmark_excluded_game_ids"]) == 13
    assert audit["exclusion_reason_counts"]["diagnostic_reserved"] == 64
    assert "NOT certified" in audit["certification"]
    all_ids = {r["sample_id"] for r in read_jsonl(DEFAULT_ROOT / "manifest.jsonl")}
    excluded_ids = {r["state_id"] for r in audit["exclusions"]}
    assert excluded_ids <= all_ids
    for split, rows in files.items():
        task_set = {r["task_type"] for r in rows}
        assert task_set <= (TRANSFER_TASKS if split.startswith("transfer_") else TRAIN_TASKS)
        for row in rows:
            p = row["metadata"].get("provenance")
            if p:
                assert p["state_id"] in all_ids - excluded_ids
                assert p["source"]["game_id"] not in audit["benchmark_excluded_game_ids"]
                assert p["split"] == split.removeprefix("transfer_")
                assert p["source"]["kind"] in ("engine_rollout", "colonist_replay")
    assert metadata["source_kind_counts"]["train"]["colonist_replay"] > 0
    assert metadata["source_kind_counts"]["train"]["engine_rollout"] > 0
    assert metadata["historical_exposure_audit"]["status"] == "known_sources_audited_history_incomplete"
    assert "Compositional transfer" in metadata["transfer_semantics"]
    inputs = read_json(output / "dataset_inputs.json")
    for key in ("train_jsonl", "validation_jsonl", "test_jsonl", "transfer_validation_jsonl", "transfer_test_jsonl", "metadata", "manifest", "token_inventory"):
        assert Path(inputs[key]).is_absolute() and Path(inputs[key]).is_file()
    assert not inputs["proposed_paired_experiment_inputs"]["final_sft_budget_approved"]
    assert read_json(output / "token_inventory.json")["tokens"] == atlas_tokens()


def test_record_schema_no_leakage_full_state_recomputation_and_active_ports(built_dataset: BuiltDataset) -> None:
    _, _, _, metadata, files = built_dataset
    assert set(metadata["coverage"]["active_query_tokens"]["train"]) == set(atlas_tokens())
    for rows in files.values():
        for row in rows:
            assert "image" not in json.dumps(row).lower()
            assert len(row["messages"]) == 2
            prompt = row["messages"][0]["content"]
            for forbidden in ("adjacent_tiles", "node_tokens", "coord", "road_lengths", "has_longest_road", "legal_nodes"):
                assert forbidden not in prompt
            state = row["metadata"]["target"]["state"]
            if state:
                assert set(state) == {"board", "colors"}
                assert len(state["board"].split(";")) == 155
                assert state["board"] in prompt
    for task in TRAIN_TASKS - STATIC_TASKS:
        row = next(r for r in files["train"] if r["task_type"] == task)
        gold = row["messages"][1]["content"]
        assert score_symbolic_task("bogus cached expected", gold, row["metadata"])["correct"]
        bad = copy.deepcopy(row["metadata"])
        bad["target"]["state"]["leaked_gold"] = gold
        with pytest.raises(ValueError):
            score_symbolic_task(gold, gold, bad)


def test_full_validation_catches_prompt_leak_or_target_mismatch(built_dataset: BuiltDataset) -> None:
    _, _, _, metadata, files = built_dataset
    pairs = directional_pair_manifest(metadata["seed"],
                                      [p.split() for p in metadata["historical_exposure_audit"]["pair_sources"]])
    changed = {k: list(v) for k, v in files.items()}
    changed["train"][0] = copy.deepcopy(changed["train"][0])
    changed["train"][0]["messages"][0]["content"] += " Endpoint table: <E00_01> -> <N00> <N01>"
    with pytest.raises(ValueError, match="prompt"):
        validate_rows(changed, pairs)
    changed["train"][0] = copy.deepcopy(files["train"][0])
    changed["train"][0]["metadata"]["row_position"] = 5
    with pytest.raises(ValueError, match="ordering"):
        validate_rows(changed, pairs)
