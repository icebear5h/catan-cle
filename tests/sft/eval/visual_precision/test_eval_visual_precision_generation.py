"""Generation, candidate autocast, and receipts."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

import sft.scripts.eval.eval_qwen_vl_adapter as evaluator

from .support import GenerationModel, GenerationProcessor


@pytest.mark.parametrize("preserve,outer_autocast", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("candidate_scoring", [False, True])
def test_eval_generation_and_candidates_share_autocast_and_write_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool, outer_autocast: bool, candidate_scoring: bool
) -> None:
    row = {
        "image": "unused.png",
        "messages": [
            {"role": "user", "content": "<image>\nIs <N00> above <N01>?"},
            {"role": "assistant", "content": "yes"},
        ],
    }
    source = tmp_path / "eval.jsonl"
    source.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(evaluator, "evaluation_image", lambda *a, **k: object())
    monkeypatch.setattr(evaluator, "resolve_dataset_asset", lambda *a: tmp_path / "unused.png")
    args = argparse.Namespace(
        image_root=None,
        limit=None,
        batch_size=1,
        long_batch_size=1,
        max_new_tokens=2,
        long_max_new_tokens=4,
        occlusion_margin=0.03,
        model_id="tiny",
        adapter_dir="checkpoint-128",
        bits=16,
        token_inventory="tokens.json",
        candidate_scoring=candidate_scoring,
    )
    # Imported pilot/rank callers construct Namespaces without the new option.
    if preserve:
        args.preserve_visual_fp32 = True
    model = GenerationModel(preserve)
    with torch.autocast("cpu", dtype=torch.bfloat16) if outer_autocast else nullcontext():
        summary = evaluator.run_eval_job(
            model=model,
            processor=GenerationProcessor(),
            args=args,
            adapter_evidence={"semantic_tokens": {"tokens": []}},
            eval_jsonl=str(source),
            image_variant="original",
            output_dir=tmp_path / "out",
        )
    assert model.contexts == [(preserve or outer_autocast, True)]
    assert not torch.is_autocast_enabled("cpu")
    assert model.visual.weight.dtype == (torch.float32 if preserve else torch.bfloat16)
    assert summary["exact_accuracy"] == 1
    if candidate_scoring:
        assert summary["candidate_exact_accuracy"] == 1
        record = json.loads((tmp_path / "out/records.jsonl").read_text())
        assert record["candidate_score"]["expected_logprob"] == pytest.approx(
            torch.log_softmax(torch.tensor([2.0, 1.0]), dim=0)[0].item()
        )
    assert summary["precision"] == {
        "preserve_visual_fp32": preserve,
        "generation_autocast": (
            {"device_type": "cpu", "dtype": "torch.bfloat16"}
            if preserve
            else {"policy": "caller_context"}
        ),
    }
    assert (
        json.loads((tmp_path / "out/summary.json").read_text())["precision"] == summary["precision"]
    )
