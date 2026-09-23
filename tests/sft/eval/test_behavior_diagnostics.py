import json
from pathlib import Path

import pytest

from sft.analysis.behavior_diagnostics import (
    behavior_names,
    build_behavior_history,
    history_markdown,
    load_checkpoint_results,
    summarize_behaviors,
)


def _record(*, metadata: dict, expected: str, correct: bool = True, row_id: str = "r1") -> dict:
    return {
        "id": row_id,
        "response": expected if correct else "wrong",
        "expected": expected,
        "metadata": metadata,
        "score": {"correct": correct, "expected_normalized": expected},
        "candidate_score": None,
    }


def test_behavior_names_separate_pair_single_tile_and_full_board_heads() -> None:
    pair = _record(
        metadata={
            "task_family": "adjacent_pair_localization",
            "task_type": "occupancy_positive",
            "piece": "ROAD",
            "pair_kind": "edge_edge",
        },
        expected="red road",
    )
    assert behavior_names(pair) == (
        "pair.positive",
        "pair.road_positive",
        "pair.edge_edge_positive",
    )

    adjacent = _record(
        metadata={
            "task_family": "single_piece_localization",
            "task_type": "occupancy_negative_adjacent",
            "piece": "ROAD",
        },
        expected="empty",
    )
    assert behavior_names(adjacent) == ("single.empty", "single.adjacent_empty")

    tile = _record(
        metadata={
            "task_family": "single_piece_localization",
            "task_type": "tile_to_token",
            "category": "inverse.tile",
        },
        expected="<T03>",
    )
    assert behavior_names(tile) == ("tile.inverse",)

    full_board = _record(
        metadata={"suite": "bidirectional", "category": "edge.owner"},
        expected="empty",
    )
    assert behavior_names(full_board) == (
        "full_board.edge_owner",
        "full_board.edge_empty",
    )

    robber = _record(
        metadata={
            "suite": "spatial_robber",
            "category": "robber",
            "task_family": "robber",
        },
        expected="yes",
    )
    assert behavior_names(robber) == ("full_board.robber",)


def test_summarize_behaviors_carries_candidate_accuracy() -> None:
    record = _record(
        metadata={
            "task_family": "single_piece_localization",
            "task_type": "occupancy_positive",
            "piece": "ROAD",
        },
        expected="red road",
        correct=False,
    )
    record["candidate_score"] = {"correct": True}

    result = summarize_behaviors([record])

    assert result["single.positive"]["exact_accuracy"] == 0
    assert result["single.road_positive"]["candidate_exact_accuracy"] == 1


def test_behavior_history_computes_best_so_far_minus_current() -> None:
    values = (0.942, 0.825, 0.908, 0.892)
    checkpoints = []
    for index, value in enumerate(values):
        checkpoints.append(
            {
                "label": f"ck{index}",
                "adapter_dir": f"adapter-{index}",
                "sources": [],
                "cells": {
                    "single-v7/single.positive": {
                        "description": "single positives",
                        "sample_fingerprint": "same-rows",
                        "exact_accuracy": value,
                    }
                },
            }
        )

    history = build_behavior_history(checkpoints)
    cells = history["behaviors"]["single-v7/single.positive"]["checkpoints"]

    assert [cells[f"ck{index}"]["forgetting"] for index in range(4)] == pytest.approx(
        [0.0, 0.117, 0.034, 0.05]
    )
    assert "best-so-far minus current" in history_markdown(history)


def test_behavior_history_marks_missing_and_rejects_noncomparable_rows() -> None:
    checkpoints = [
        {
            "label": "before",
            "cells": {
                "set/behavior": {
                    "description": "test",
                    "sample_fingerprint": "aaa",
                    "exact_accuracy": 0.8,
                }
            },
        },
        {
            "label": "after",
            "cells": {
                "set/behavior": {
                    "description": "test",
                    "sample_fingerprint": "bbb",
                    "exact_accuracy": 0.9,
                },
                "new/behavior": {
                    "description": "new",
                    "sample_fingerprint": "ccc",
                    "exact_accuracy": 1.0,
                },
            },
        },
    ]

    history = build_behavior_history(checkpoints)

    assert history["errors"]
    assert history["behaviors"]["set/behavior"]["checkpoints"]["after"]["status"] == "incomparable"
    assert history["behaviors"]["new/behavior"]["checkpoints"]["before"]["status"] == "missing"
    assert "N/A" in history_markdown(history)


def test_behavior_history_aligns_historical_set_names_by_exact_row_fingerprint() -> None:
    checkpoints = [
        {
            "label": "v3",
            "cells": {
                "full-panel/single.positive": {
                    "behavior": "single.positive",
                    "description": "single positives",
                    "eval_set_fingerprint": "same-eval-set",
                    "sample_fingerprint": "identical",
                    "exact_accuracy": 0.9,
                }
            },
        },
        {
            "label": "pairs-v2",
            "cells": {
                "ck256/single.positive": {
                    "behavior": "single.positive",
                    "description": "single positives",
                    "eval_set_fingerprint": "same-eval-set",
                    "sample_fingerprint": "identical",
                    "exact_accuracy": 0.8,
                }
            },
        },
    ]

    history = build_behavior_history(checkpoints)

    cells = history["behaviors"]["full-panel/single.positive"]["checkpoints"]
    assert cells["pairs-v2"]["forgetting"] == pytest.approx(0.1)
    assert history["eval_set_aliases"] == [
        {
            "checkpoint": "pairs-v2",
            "source": "ck256/single.positive",
            "canonical": "full-panel/single.positive",
        }
    ]


def test_eval_set_fingerprint_keeps_shared_behavior_subsets_distinct() -> None:
    checkpoint = {
        "label": "one",
        "cells": {
            "v2/single.positive": {
                "behavior": "single.positive",
                "description": "single positives",
                "eval_set_fingerprint": "v2-all-rows",
                "sample_fingerprint": "shared-positive-rows",
                "exact_accuracy": 0.9,
            },
            "v3/single.positive": {
                "behavior": "single.positive",
                "description": "single positives",
                "eval_set_fingerprint": "v3-all-rows",
                "sample_fingerprint": "shared-positive-rows",
                "exact_accuracy": 0.9,
            },
        },
    }

    history = build_behavior_history([checkpoint])

    assert set(history["behaviors"]) == {
        "v2/single.positive",
        "v3/single.positive",
    }


def test_load_checkpoint_results_reads_only_original_variant(tmp_path: Path) -> None:
    original = tmp_path / "set-a" / "original"
    blank = tmp_path / "set-a" / "blank"
    original.mkdir(parents=True)
    blank.mkdir(parents=True)
    summary = {"adapter_dir": "/runs/adapter", "eval_set_id": "set-a"}
    record = _record(
        metadata={
            "task_family": "single_piece_localization",
            "task_type": "occupancy_positive",
            "piece": "ROAD",
            "eval_variant": "original",
        },
        expected="red road",
    )
    for directory, variant in ((original, "original"), (blank, "blank")):
        (directory / "summary.json").write_text(
            json.dumps({**summary, "image_variant": variant})
        )
        (directory / "records.jsonl").write_text(json.dumps(record) + "\n")

    result = load_checkpoint_results("checkpoint", [tmp_path])

    assert set(result["cells"]) == {"set-a/single.positive", "set-a/single.road_positive"}
