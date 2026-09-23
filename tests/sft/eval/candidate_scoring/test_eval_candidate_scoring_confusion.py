"""Neighbor confusion buckets and pair metadata."""

from sft.scripts.eval.eval_qwen_vl_adapter import (
    evaluation_metadata,
    summarize,
    summarize_neighbor_confusion,
)

from .support import _occupancy_record


def test_neighbor_confusion_buckets_wrong_answers_by_what_the_model_named() -> None:
    records = [
        _occupancy_record(correct=True, expected="dark blue settlement", response="dark blue settlement"),
        _occupancy_record(correct=False, expected="dark blue settlement", response="red city"),
        _occupancy_record(correct=False, expected="dark blue settlement", response="red city", partner_distance=2),
        _occupancy_record(correct=False, expected="empty", response="dark blue settlement", negative_distance=1),
        _occupancy_record(correct=False, expected="empty", response="red city", negative_distance=2),
        _occupancy_record(correct=False, expected="dark blue settlement", response="empty"),
        _occupancy_record(correct=False, expected="dark blue settlement", response="white road"),
        _occupancy_record(correct=False, expected="dark blue settlement", response="EMPTY", task_type="occupancy_single", partner=False),
        _occupancy_record(correct=False, expected="orange city", response="dark blue settlement", task_type="occupancy_single", partner=False),
        {"response": "x", "metadata": {"category": "tile.resource"}, "score": {"correct": False}},
        {"response": "x", "metadata": {"task_type": "occupancy_single"}, "score": {"correct": False}},
    ]

    confusion = summarize_neighbor_confusion(records)

    assert set(confusion) == {"occupancy_pair", "occupancy_single"}
    pair = confusion["occupancy_pair"]
    assert pair["total"] == 7
    assert pair["total_wrong"] == 6
    assert pair["names_target"] == 1
    assert pair["names_partner"] == 3
    assert pair["answers_empty"] == 1
    assert pair["other"] == 1
    assert pair["by_distance"] == {
        "names_target": {"1": 1},
        "names_partner": {"1": 1, "2": 2},
    }
    single = confusion["occupancy_single"]
    assert single["total"] == 3
    assert single["total_wrong"] == 3
    assert single["names_target"] == 1
    assert single["names_partner"] == 0
    assert single["answers_empty"] == 1
    assert single["other"] == 1
    assert single["by_distance"] == {"names_target": {"None": 1}, "names_partner": {}}


def test_summary_exposes_neighbor_confusion_and_pair_dimensions() -> None:
    records = [
        _occupancy_record(correct=False, expected="dark blue settlement", response="red city"),
        {"response": "x", "metadata": {"pair_kind": "adjacent"}, "score": {"correct": True}},
    ]

    summary = summarize(records)

    assert summary["neighbor_confusion"]["occupancy_pair"]["names_partner"] == 1
    assert summary["by_pair_kind"]["adjacent"]["correct"] == 1
    assert summary["by_negative_distance"]["unknown"]["total"] == 2


def test_evaluation_metadata_copies_pair_fields_from_row() -> None:
    row = {
        "metadata": {"category": "node.occupancy"},
        "task_type": "occupancy_pair",
        "target_token": "<N07>",
        "queried_token": "<N07>",
        "pair_kind": "adjacent",
        "partner_token": "<N08>",
        "partner_piece": "City",
        "partner_color": "red",
        "partner_distance": 1,
        "same_color": False,
        "negative_distance": 2,
        "negative_kind": "empty_neighbor",
        "piece": "Settlement",
        "color": "dark_blue",
        "eval_set_id": "pairs-v2-validation",
        "eval_source_sha256": "abc123",
    }

    metadata = evaluation_metadata(row, image_variant="original")

    assert metadata["partner_token"] == "<N08>"
    assert metadata["negative_distance"] == 2
    assert metadata["queried_token"] == "<N07>"
    assert metadata["pair_kind"] == "adjacent"
    assert metadata["same_color"] is False
    assert metadata["negative_kind"] == "empty_neighbor"
    assert metadata["eval_set_id"] == "pairs-v2-validation"
    assert metadata["eval_source_sha256"] == "abc123"
    assert metadata["eval_variant"] == "original"


def test_summary_exposes_curated_behavior_metrics() -> None:
    record = {
        "response": "red road",
        "metadata": {
            "task_family": "adjacent_pair_localization",
            "task_type": "occupancy_positive",
            "piece": "ROAD",
            "pair_kind": "edge_edge",
        },
        "score": {"correct": True, "expected_normalized": "red road"},
        "candidate_score": None,
    }

    summary = summarize([record])

    assert summary["by_behavior"]["pair.positive"]["exact_accuracy"] == 1.0
    assert summary["by_behavior"]["pair.road_positive"]["total"] == 1
