import pytest
import torch

from sft.scripts.eval_qwen_vl_adapter import (
    is_long_answer,
    score_response,
    candidate_answers,
    candidate_token_ids,
    evaluation_metadata,
    score_candidates,
    summarize,
    summarize_neighbor_confusion,
    summarize_occupancy_classes,
    occupancy_class,
    score_readout,
)


ATLAS = [f"<N{i:02d}>" for i in range(54)] + ["<E00_01>", "<E00_05>"] + [f"<T{i:02d}>" for i in range(19)] + [f"<P{i:02d}>" for i in range(9)]


class _Tokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[text]]


def test_candidate_sets_follow_entity_type_marker_letters_and_polarity():
    probe = {"metadata": {"entity_type": "node", "task_type": "neutral_probe_token_return"}}
    assert candidate_answers(probe, "<N07>", ATLAS) == [f"<N{i:02d}>" for i in range(54)]
    marker = {"task_type": "token_to_marker", "entity_type": "edge"}
    assert candidate_answers(marker, "C", ATLAS) == ["A", "B", "C", "D"]
    untyped = {"metadata": {}}
    assert candidate_answers(untyped, "<T04>", ATLAS) == [f"<T{i:02d}>" for i in range(19)]
    assert candidate_answers({"metadata": {"polarity": "positive"}}, "yes", ATLAS) == ["yes", "no"]
    assert candidate_answers({"metadata": {}}, "red settlement", ATLAS) is None
    with pytest.raises(ValueError):
        candidate_answers({"metadata": {"entity_type": "tile"}}, "<N07>", ATLAS)


def test_candidate_ranking_reports_argmax_and_expected_rank():
    candidates = ["<N00>", "<N01>", "<N02>"]
    tokenizer = _Tokenizer({"<N00>": 5, "<N01>": 6, "<N02>": 7, "<T00>": 9})
    ids = candidate_token_ids(tokenizer, candidates)
    logits = torch.full((12,), -5.0)
    logits[9] = 4.0  # a tile token wins the open argmax; it is not a candidate
    logits[6] = 2.0
    logits[7] = 1.0
    logits[5] = 0.0

    result = score_candidates(logits, candidates, ids, "<N02>")

    assert result["predicted"] == "<N01>"
    assert result["correct"] is False
    assert result["expected_rank"] == 2
    assert [item["answer"] for item in result["top"]] == ["<N01>", "<N02>", "<N00>"]
    assert result["expected_logprob"] < 0

    exact = score_candidates(logits, candidates, ids, "<N01>")
    assert exact["correct"] is True and exact["expected_rank"] == 1

    with pytest.raises(ValueError):
        candidate_token_ids(_Tokenizer({"a": 1, "b": 1}), ["a", "b"])


def test_summary_carries_candidate_accuracy_beside_exact_match():
    def record(correct: bool, candidate: dict | None, style: str) -> dict:
        return {
            "response": "x",
            "metadata": {"probe_style": style},
            "score": {"correct": correct},
            "candidate_score": candidate,
        }

    records = [
        record(False, {"correct": True, "expected_rank": 1}, "small"),
        record(False, {"correct": False, "expected_rank": 3}, "small"),
        record(True, {"correct": True, "expected_rank": 1}, "large"),
        record(True, None, "large"),
    ]

    summary = summarize(records)

    assert summary["exact_accuracy"] == pytest.approx(0.5)
    assert summary["candidate_rows"] == 3
    assert summary["candidate_exact_accuracy"] == pytest.approx(2 / 3)
    assert summary["candidate_expected_rank_le3"] == pytest.approx(1.0)
    small = summary["by_probe_style"]["small"]
    assert small["candidate_exact_accuracy"] == pytest.approx(0.5)
    assert small["candidate_expected_rank_mean"] == pytest.approx(2.0)
    assert "candidate_total" not in summary["by_probe_style"]["large"] or summary["by_probe_style"]["large"]["candidate_total"] == 1


def test_eval_jobs_keep_single_layout_and_nest_batches(tmp_path):
    import argparse

    from sft.scripts.eval_qwen_vl_adapter import eval_jobs

    single = argparse.Namespace(
        eval_jsonl=["probes/validation.jsonl"], image_variant="original", output_dir=str(tmp_path)
    )
    assert eval_jobs(single) == [
        {"eval_jsonl": "probes/validation.jsonl", "image_variant": "original", "output_dir": str(tmp_path)}
    ]

    batch = argparse.Namespace(
        eval_jsonl=["probes/validation.jsonl", "stage1/validation.jsonl"],
        image_variant="original, blank",
        output_dir=str(tmp_path),
    )
    jobs = eval_jobs(batch)
    assert [job["output_dir"] for job in jobs] == [
        str(tmp_path / "probes-validation" / "original"),
        str(tmp_path / "probes-validation" / "blank"),
        str(tmp_path / "stage1-validation" / "original"),
        str(tmp_path / "stage1-validation" / "blank"),
    ]
    clash = argparse.Namespace(
        eval_jsonl=["v2/stage1/validation.jsonl", "v3/stage1/validation.jsonl"],
        image_variant="original",
        output_dir=str(tmp_path),
    )
    assert len({job["output_dir"] for job in eval_jobs(clash)}) == 2
    with pytest.raises(ValueError):
        eval_jobs(argparse.Namespace(eval_jsonl=["a.jsonl"], image_variant="sepia", output_dir="x"))


def _occupancy_record(
    *,
    correct: bool,
    expected: str,
    response: str,
    task_type: str = "occupancy_pair",
    partner: bool = True,
    partner_distance: int | None = 1,
    negative_distance: int | None = None,
) -> dict:
    metadata = {
        "task_type": task_type,
        "category": "node.occupancy",
        "piece": "Settlement",
        "color": "dark_blue",
    }
    if partner:
        metadata.update(
            {
                "partner_piece": "City",
                "partner_color": "red",
                "partner_distance": partner_distance,
            }
        )
    if negative_distance is not None:
        metadata["negative_distance"] = negative_distance
    return {
        "response": response,
        "metadata": metadata,
        "score": {
            "correct": correct,
            "expected_normalized": expected,
            "response_normalized": response,
        },
    }


def test_neighbor_confusion_buckets_wrong_answers_by_what_the_model_named():
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


def test_summary_exposes_neighbor_confusion_and_pair_dimensions():
    records = [
        _occupancy_record(correct=False, expected="dark blue settlement", response="red city"),
        {"response": "x", "metadata": {"pair_kind": "adjacent"}, "score": {"correct": True}},
    ]

    summary = summarize(records)

    assert summary["neighbor_confusion"]["occupancy_pair"]["names_partner"] == 1
    assert summary["by_pair_kind"]["adjacent"]["correct"] == 1
    assert summary["by_negative_distance"]["unknown"]["total"] == 2


def test_evaluation_metadata_copies_pair_fields_from_row():
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


def test_summary_exposes_curated_behavior_metrics():
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


def test_long_answer_rows_are_detected_by_task_type_or_length():
    short = {"messages": [{"role": "user", "content": "<image>\n<T00> resource?"}, {"role": "assistant", "content": "wood"}]}
    readout = {"task_type": "terrain_readout", "messages": [{"role": "user", "content": "<image>\nRead all tiles and ports."}, {"role": "assistant", "content": "<T00> wood 11; <T01> brick 2"}]}
    long_text = {"messages": [{"role": "user", "content": "<image>\nx"}, {"role": "assistant", "content": "a" * 60}]}
    assert is_long_answer(short) is False
    assert is_long_answer(readout) is True
    assert is_long_answer(long_text) is True


def test_readout_scoring_is_whitespace_tolerant_and_counts_items():
    expected = "<T00> wood 11; <T01> brick 2; <T02> desert none; <P00> 3:1 port; <P01> ore port"
    exact = score_response(expected, "<T00> wood 11;<T01> brick 2; <T02> desert none;<P00> 3:1 port; <P01> ore port")
    assert exact["scoring"] == "readout_items" and exact["correct"] is True and exact["items_correct"] == 5
    partial = score_response(expected, "<T00> wood 11; <T01> brick 3; <T02> desert none; <P00> 3:1 port")
    assert partial["correct"] is False and partial["items_correct"] == 3 and partial["items_total"] == 5
    short = score_response("wood", "wood")
    assert short["scoring"] != "readout_items" and short["correct"] is True


def test_readout_scores_split_occupied_and_empty_items():
    expected = "<N00> empty; <N01> red settlement; <N02> empty; <N03> blue city"
    response = "<N00> empty; <N01> red settlement; <N02> red road; <N03> empty"
    score = score_readout(expected, response)
    assert score["occupied_items_total"] == 2 and score["occupied_items_correct"] == 1
    assert score["empty_items_total"] == 2 and score["empty_items_correct"] == 1
    assert score["items_correct"] == 2 and not score["correct"]


def test_occupancy_classes_report_recall_precision_and_balanced_accuracy():
    def record(expected, response, category="node.occupancy"):
        return {"metadata": {"category": category}, "score": {"correct": expected == response, "expected_normalized": expected, "response_normalized": response}}

    records = [
        record("red settlement", "red settlement"),
        record("blue settlement", "empty"),
        record("green city", "green city"),
        record("empty", "empty"),
        record("empty", "empty"),
        record("empty", "orange road", "edge.owner"),
        record("orange road", "empty", "edge.owner"),
        record("wood 11", "wood 11", "tile.resource"),
    ]
    assert occupancy_class("mystic blue city") == "city" and occupancy_class("empty") == "empty" and occupancy_class("") == "empty"
    summary = summarize_occupancy_classes(records)
    classes = summary["classes"]
    assert classes["settlement"] == {"expected": 2, "predicted": 1, "correct": 1, "recall": 0.5, "precision": 1.0}
    assert classes["city"]["recall"] == 1.0 and classes["road"]["recall"] == 0.0 and classes["road"]["predicted"] == 1
    assert classes["empty"] == {"expected": 3, "predicted": 4, "correct": 2, "recall": 2 / 3, "precision": 0.5}
    assert abs(summary["balanced_accuracy"] - (0.5 + 1.0 + 0.0 + 2 / 3) / 4) < 1e-9
    assert summary["occupied_recall"] == 2 / 4 and summary["empty_precision"] == 0.5
