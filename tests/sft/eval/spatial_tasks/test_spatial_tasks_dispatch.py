"""Dynamic JSON normalization and eval job dispatch."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sft.scripts.eval.eval_qwen_vl_adapter as evaluator
from cle.game_engine.public_board import JsonValue
from sft.board.spatial_tasks import (
    local_node_tiles,
    score_spatial_task,
)

from .support import DICE_METADATA, LOCAL_METADATA, PATH, PATH_METADATA, ROOT, TILE_METADATA, ZERO


@pytest.mark.parametrize("task", ["local_node_tiles", "dice_production"])
def test_dynamic_json_is_order_insensitive_but_strict_and_never_extracts_prose(contract: dict[str, JsonValue], task: str) -> None:
    metadata = LOCAL_METADATA if task == "local_node_tiles" else DICE_METADATA
    value = (
        local_node_tiles(contract, "<N00>") if task == "local_node_tiles" else {**ZERO, "ore": 1}
    )
    gold = json.dumps(value)
    reordered = {
        key: dict(reversed(list(item.items()))) if isinstance(item, dict) else item
        for key, item in reversed(list(value.items()))
    }
    assert score_spatial_task(gold, json.dumps(reordered, indent=2), metadata)["correct"]
    key = next(iter(value))
    invalid = [
        "",
        "null",
        "[]",
        "{}",
        json.dumps(list(value)),
        gold + " trailing prose",
        "Here is the answer: " + gold,
        gold + gold,
        json.dumps({k: v for k, v in value.items() if k != key}),
        json.dumps({**value, "extra": 0}),
        gold[:-1] + ", " + json.dumps(key) + ": " + json.dumps(value[key]) + "}",
    ]
    if task == "local_node_tiles":
        for replacement in (
            None,
            [],
            {"resource": "wood"},
            {"resource": "wood", "number": 8, "extra": 0},
            {"resource": "WOOD", "number": 8},
            {"resource": "desert", "number": 8},
            {"resource": "wood", "number": None},
            {"resource": "wood", "number": "8"},
            {"resource": "wood", "number": True},
            {"resource": "wood", "number": 8.0},
            {"resource": "wood", "number": 7},
            {"resource": "gold", "number": 8},
        ):
            invalid.append(json.dumps({**value, key: replacement}))
        invalid.append(gold.replace('"resource": "ore"', '"resource": "ore", "resource": "ore"', 1))
    else:
        for replacement in (True, False, 0.0, -1, "0", None, [], {}, float("nan"), float("inf")):
            invalid.append(json.dumps({**value, key: replacement}))
        invalid.append(gold.replace('"ore": 1', '"ore": 1.0'))
        invalid.append(gold.replace('"ore": 1', '"ore": true'))
        invalid.append(gold.replace('"ore": 1', '"ore": 1e0'))
    for response in invalid:
        assert not score_spatial_task(gold, response, metadata)["correct"], response
        with pytest.raises(ValueError):
            score_spatial_task(response, gold, metadata)
    changed = copy.deepcopy(value)
    if task == "local_node_tiles":
        changed[key]["number"] = 6
    else:
        changed["ore"] = 2
    assert not score_spatial_task(gold, json.dumps(changed), metadata)["correct"]


def test_normalization_is_shared_by_gold_and_response_and_legacy_is_untouched(contract: dict[str, JsonValue]) -> None:
    cases = [
        ("<T00> <T05> <T06>", "<T06> <T00> <T05>", TILE_METADATA),
        (" ".join(PATH), "<N00> <N05> <N04> <N03>", PATH_METADATA),
        (
            json.dumps(local_node_tiles(contract, "<N00>")),
            json.dumps(local_node_tiles(contract, "<N00>")),
            LOCAL_METADATA,
        ),
        (json.dumps(ZERO), json.dumps(ZERO), DICE_METADATA),
    ]
    for gold, response, metadata in cases:
        for wrapper in (" {} ", "```json\n{}\n```", "Answer: {}", "Assistant: {}<|im_end|>"):
            assert evaluator.score_response(
                wrapper.format(gold), wrapper.format(response), metadata=metadata
            )["correct"]
    for metadata in ({}, {"task_type": "unrelated", "target": None}):
        assert score_spatial_task("not gold", "anything", metadata) is None
        assert evaluator.score_response(
            "yes", "yes, because", metadata=metadata
        ) == evaluator.score_response("yes", "yes, because")
        assert evaluator.score_response('{"x": 1}', 'prose {"x": true}', metadata=metadata)[
            "correct"
        ]


@pytest.mark.parametrize("task_type", [None, "", "node_tiles", "full_board_readout"])
def test_conflicting_task_declarations_cannot_bypass_ordered_path_scoring(task_type: str | None) -> None:
    row = {"task_type": task_type, "metadata": PATH_METADATA}
    with pytest.raises(ValueError, match="conflicting spatial task declarations"):
        evaluator.evaluation_metadata(row, image_variant="original")


def test_nested_path_declaration_survives_without_top_level_override() -> None:
    for row in ({"metadata": PATH_METADATA}, {"task_type": "shortest_node_path", "metadata": PATH_METADATA}):
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        assert not evaluator.score_response(
            " ".join(PATH), "<N00> <N02> <N01> <N03>", metadata=metadata
        )["correct"]
    with pytest.raises(ValueError, match="conflicting spatial task declarations"):
        evaluator.evaluation_metadata(
            {"task_type": "unrelated", "metadata": {"category": "shortest_node_path"}},
            image_variant="original",
        )


def test_eval_job_dispatches_nested_target_before_readout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "id": "path-dispatch",
        "image": "unused.png",
        "task_type": "shortest_node_path",
        "metadata": {"target": PATH_METADATA["target"]},
        "messages": [
            {"role": "user", "content": "Give a shortest path from <N00> to <N03>."},
            {"role": "assistant", "content": " ".join(PATH)},
        ],
    }
    source = tmp_path / "eval.jsonl"
    source.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(evaluator, "resolve_dataset_asset", lambda *args: tmp_path / "unused.png")
    # Only inference is stubbed: exercise the real row, metadata, score, and receipt pipeline offline.
    monkeypatch.setattr(
        evaluator, "generate_responses", lambda **kwargs: (["<N00> <N02> <N01> <N03>"], None)
    )
    args = argparse.Namespace(
        image_root=None,
        limit=None,
        batch_size=1,
        long_batch_size=1,
        max_new_tokens=16,
        long_max_new_tokens=128,
        occlusion_margin=0.03,
        candidate_scoring=False,
        model_id="offline",
        adapter_dir=None,
        bits=16,
        token_inventory=None,
    )
    summary = evaluator.run_eval_job(
        model=None,
        processor=SimpleNamespace(tokenizer=None),
        args=args,
        adapter_evidence={"semantic_tokens": {"tokens": []}},
        eval_jsonl=str(source),
        image_variant="original",
        output_dir=tmp_path / "out",
    )
    record = json.loads((tmp_path / "out" / "records.jsonl").read_text())
    assert record["metadata"]["target"] == PATH_METADATA["target"]
    assert record["score"]["scoring"] == "shortest_node_path"
    assert not record["score"]["correct"] and summary["exact_accuracy"] == 0


@pytest.mark.parametrize(
    "relative_path,total,correct",
    [
        ("full-board-new-layouts-ck128-spatial-answer-only-v1-20260908-r01/records.jsonl", 120, 57),
        ("full-board-new-layouts-20260907/heldout-records.jsonl", 64, 64),
    ],
)
def test_archived_matched_spatial_and_full_board_scores_are_unchanged(
    relative_path: str, total: int, correct: int
) -> None:
    path = ROOT / "artifacts" / "runs" / "sft" / relative_path
    if not path.is_file():
        pytest.skip("local archived eval records are not included in a clean checkout")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(records) == total
    scores = [
        evaluator.score_response(row["expected"], row["response"], metadata=row["metadata"])
        for row in records
    ]
    assert sum(score["correct"] for score in scores) == correct
    for row, score in zip(records, scores):
        assert score == row["score"]
        assert score == evaluator.score_response(row["expected"], row["response"])
