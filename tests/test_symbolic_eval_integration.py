"""Exercise strict symbolic dispatch through the real shared evaluator."""

import copy

import pytest

from sft.scripts import eval_qwen_vl_adapter as evaluator
from sft.symbolic_board_tasks import symbolic_answer


def row():
    task = "symbolic_neighbors"
    target = {"state": None, "query": {"token": "<N00>"}}
    metadata = {"task_type": task, "training_family": task,
                "split": "validation", "task_role": "component_eval", "target": target}
    return {**metadata, "metadata": copy.deepcopy(metadata)}


def test_strict_symbolic_dispatch_before_readout_normalization():
    example = row()
    # Use the task's own role contract rather than duplicating its spelling.
    from_role = evaluator.symbolic_task_role(example["task_type"], "validation")
    example["task_role"] = example["metadata"]["task_role"] = from_role
    metadata = evaluator.evaluation_metadata(example, image_variant="original")
    answer = symbolic_answer(example["task_type"], metadata["target"])
    assert evaluator.score_response(answer, " ".join(reversed(answer.split())), metadata=metadata)["correct"]
    for response in ("Answer: " + answer, answer + " explanation", answer + " " + answer.split()[0]):
        assert not evaluator.score_response(answer, response, metadata=metadata)["correct"]
    with pytest.raises(ValueError):
        evaluator.score_response(answer, answer, metadata={**metadata, "target": {}})


@pytest.mark.parametrize("field,value", [("task_type", "other"), ("training_family", "other"),
                                          ("task_role", "train"), ("split", "train")])
def test_contradictory_symbolic_declarations_fail(field, value):
    example = row()
    example[field] = value
    with pytest.raises(ValueError):
        evaluator.evaluation_metadata(example, image_variant="original")


def test_transfer_summary_uses_supplied_macro_weights():
    records = [{"id": str(i), "expected": "NONE", "response": "NONE",
                "score": {"correct": correct, "scoring": "symbolic_settlement_locations"},
                "metadata": {"task_type": "symbolic_settlement_locations", "macro_weight": weight}}
               for i, (correct, weight) in enumerate(((True, 0.25), (False, 0.75)))]
    result = evaluator.summarize(records)["symbolic_families"]["symbolic_settlement_locations"]
    assert result["accuracy"] == 0.25 and result["correct"] == 1
