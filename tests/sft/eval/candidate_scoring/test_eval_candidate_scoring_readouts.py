"""Readout scoring and occupancy class metrics."""

from sft.scripts.eval.eval_qwen_vl_adapter import (
    is_long_answer,
    occupancy_class,
    score_readout,
    score_response,
    summarize_occupancy_classes,
)


def test_long_answer_rows_are_detected_by_task_type_or_length() -> None:
    short = {"messages": [{"role": "user", "content": "<image>\n<T00> resource?"}, {"role": "assistant", "content": "wood"}]}
    readout = {"task_type": "terrain_readout", "messages": [{"role": "user", "content": "<image>\nRead all tiles and ports."}, {"role": "assistant", "content": "<T00> wood 11; <T01> brick 2"}]}
    long_text = {"messages": [{"role": "user", "content": "<image>\nx"}, {"role": "assistant", "content": "a" * 60}]}
    assert is_long_answer(short) is False
    assert is_long_answer(readout) is True
    assert is_long_answer(long_text) is True


def test_readout_scoring_is_whitespace_tolerant_and_counts_items() -> None:
    expected = "<T00> wood 11; <T01> brick 2; <T02> desert none; <P00> 3:1 port; <P01> ore port"
    exact = score_response(expected, "<T00> wood 11;<T01> brick 2; <T02> desert none;<P00> 3:1 port; <P01> ore port")
    assert exact["scoring"] == "readout_items" and exact["correct"] is True and exact["items_correct"] == 5
    partial = score_response(expected, "<T00> wood 11; <T01> brick 3; <T02> desert none; <P00> 3:1 port")
    assert partial["correct"] is False and partial["items_correct"] == 3 and partial["items_total"] == 5
    short = score_response("wood", "wood")
    assert short["scoring"] != "readout_items" and short["correct"] is True


def test_readout_scores_split_occupied_and_empty_items() -> None:
    expected = "<N00> empty; <N01> red settlement; <N02> empty; <N03> blue city"
    response = "<N00> empty; <N01> red settlement; <N02> red road; <N03> empty"
    score = score_readout(expected, response)
    assert score["occupied_items_total"] == 2 and score["occupied_items_correct"] == 1
    assert score["empty_items_total"] == 2 and score["empty_items_correct"] == 1
    assert score["items_correct"] == 2 and not score["correct"]


def test_occupancy_classes_report_recall_precision_and_balanced_accuracy() -> None:
    def record(
        expected: str, response: str, category: str = "node.occupancy"
    ) -> dict[str, object]:
        return {
            "metadata": {"category": category},
            "score": {
                "correct": expected == response,
                "expected_normalized": expected,
                "response_normalized": response,
            },
        }

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
