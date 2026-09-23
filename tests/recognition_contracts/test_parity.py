"""Frozen pre-reorganization witnesses; do not regenerate from current code."""
import importlib
import pickle
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from data_pipeline.board_recognition import dataset, spatial_localization
from data_pipeline.board_recognition.replay_dataset import BoardStateCandidate

from .reference import baseline
from .witnesses import FIXTURE, witnesses

EXPECTED = {
    "inverse_rows": "f23f0cfe33158645795d5043e43f83b1ca73880d8657955928987d4fb0242b74",
    "pair_placements": "d9c1a69023dd93368e3adedb6cbccf0a28473694fa40782f6092968acdb9e52a",
    "query_schedule": "10846c9c66a5058fa0933e7e2354c62206c582c7191610e5c981e334fd2330fa",
    "semantic_rows": "fc14d8ed120ec8d401b68d2ef343b511835d1fc25a6a45f5af23c977e84dcd74",
    "density_records": "a7d5674c086fc69d69287ee85c2986044bf0c413d3bc2ee006e451cd00b55a33",
    "production_rows": "47872b388303d4dacdfd748403d41fc1f66e5fbf747d85feb5fb80fecc650636",
    "regions": "8f1202bc21541f7f7c2141972ee97914a00d9a6fd65e6208192d7b8c5d762e9b",
    "relations": "60f15f0d90082d589a81f4b96a8529d1c8ad4762198a119c2c5e9a75fbcd4151",
    "marker_rows": "cf9d55b6e8551891190ef2f1ca11189b911ac47bb5aa420bb1d6cc79a1d567a3",
    "probe_rows": "d121c658c2de65d90bede723728d72d0a50a15cf3ac668ca22a0a0f5c1777bca",
    "png_bytes": "c5c08303168cc8e0f79e7a020eca029713f6a6081ddf5aa827cbc84a20accd91",
    "dataset_queries": "0d4cee4b35f7f74e42eb2ce66cb07f8f33cd9ffdf8b0021417b5a686316d9610",
}


def test_pre_reorganization_output_witnesses(tmp_path: Path) -> None:
    assert witnesses(tmp_path) == EXPECTED


def test_dataset_path_and_pickle_identity() -> None:
    assert dataset.PROJECT_ROOT == Path(__file__).resolve().parents[2]
    original = dataset.BoardRecognitionStateDataset(FIXTURE, split="train")
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is dataset.BoardRecognitionStateDataset
    assert restored[0]["queries"] == original[0]["queries"]
    assert dataset.BoardRecognitionStateDataset.__module__ == "data_pipeline.board_recognition.dataset"
    assert BoardStateCandidate.__module__ == "data_pipeline.board_recognition.replay_dataset"


def test_dataset_sampler_delegates_through_public_helpers() -> None:
    original = baseline("dataset")
    original_states = original.BoardRecognitionStateDataset(FIXTURE, queries_per_state=16, seed=123)
    current_states = dataset.BoardRecognitionStateDataset(FIXTURE, queries_per_state=16, seed=123)
    implementation = importlib.import_module("data_pipeline.board_recognition.dataset.sampling")
    helpers = (
        "sample_state_queries", "attribute_ids", "class_balanced_slot_index",
        "class_vocabulary", "stable_seed", "normalize_class_name",
    )
    for name in helpers:
        assert getattr(dataset, name) is getattr(implementation, name)
    with ExitStack() as stack:
        original_calls = {
            name: stack.enter_context(patch.object(original, name, wraps=getattr(original, name)))
            for name in helpers
        }
        current_calls = {
            name: stack.enter_context(patch.object(dataset, name, wraps=getattr(dataset, name)))
            for name in helpers
        }
        expected = original_states[0]["queries"]
        actual: Any = current_states[0]["queries"]
        assert actual == expected
        assert len(actual) == 16
        assert current_calls["class_balanced_slot_index"].call_count == 15
        for name in helpers:
            assert original_calls[name].call_count > 0
            assert current_calls[name].call_args_list == original_calls[name].call_args_list


def test_reexports_retain_function_identity_and_patch_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    implementation = importlib.import_module("data_pipeline.board_recognition.spatial_localization.relations")
    assert implementation._deterministic_replay is spatial_localization._deterministic_replay
    calls: list[tuple[Any, ...]] = []

    def rank(*parts: object) -> int:
        calls.append(parts)
        return 0

    monkeypatch.setattr(spatial_localization, "_stable_rank", rank)
    assert spatial_localization._deterministic_shuffle([{"row_id": "fixture"}], "test") == [{"row_id": "fixture"}]
    assert calls == [("fixture", 0, "", "test")]


@pytest.mark.parametrize("name", ["spatial_localization", "adjacent_pair_localization"])
def test_module_entrypoint(name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"data_pipeline.board_recognition.{name}", "--help"],
        check=True, text=True, capture_output=True,
    )
    assert "--output-dir" in result.stdout
    assert "dataset_dir" in result.stdout
