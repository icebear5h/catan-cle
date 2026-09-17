"""Text evaluation on real tiny PEFT models, with every media entry point forbidden."""

import json
from types import SimpleNamespace

import PIL.Image
import pytest
import torch
import transformers
from safetensors.torch import load_file

from sft.scripts import eval_qwen_vl_adapter as evaluator
from sft.scripts import train_trl_catan_vision as training
from test_trl_catan_text import TinyTextVLM, text_row, text_tokenizer, write_text_source


def forbid_media(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("text evaluation reached a media entry point")

    for name in ("evaluation_image", "image_reference", "resolve_dataset_image", "resolve_dataset_asset"):
        monkeypatch.setattr(evaluator, name, forbidden)
    monkeypatch.setattr(PIL.Image, "open", forbidden)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", forbidden)


def test_text_messages_and_cli_preserve_historical_defaults():
    row = text_row("board <N00>")
    assert evaluator.build_qwen_messages(row, input_mode="text") == row["messages"][:1]
    default = evaluator.parse_args(["--eval-jsonl", "e", "--output-dir", "o"])
    assert default.input_mode == "vision" and default.bits == 4
    assert default.model_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert default.image_root is None and default.max_new_tokens == 256
    text = evaluator.parse_args([
        "--eval-jsonl", "e", "--output-dir", "o", "--input-mode", "text",
        "--max-sequence-length", "8192", "--adapter-dir", "checkpoint-128",
    ])
    assert evaluator.eval_jobs(text) == [{"eval_jsonl": "e", "image_variant": "original", "output_dir": "o"}]
    for variant in ("blank", "shuffle", "target_occlusion", "control_occlusion", "original,blank"):
        text.image_variant = variant
        with pytest.raises(ValueError, match="does not support image variants"):
            evaluator.eval_jobs(text)


@pytest.mark.parametrize("return_logits", [False, True])
def test_real_no_image_generation_matches_direct_language_forward(tmp_path, monkeypatch, return_logits):
    source, tokenizer, setup, components, _ = write_text_source(tmp_path / "parent")
    inventory_path = tmp_path / "tokens.json"
    training.write_json_atomic(inventory_path, training.semantic_recognition_token_inventory())
    forbid_media(monkeypatch)
    factory_calls = []

    def factory(model_id, **kwargs):
        factory_calls.append((model_id, kwargs))
        return TinyTextVLM().bfloat16()

    monkeypatch.setattr(transformers, "AutoModelForMultimodalLM", SimpleNamespace(from_pretrained=factory), raising=False)
    model, saved_tokenizer, evidence = evaluator.load_model(
        model_id="offline-tiny", adapter_dir=str(tmp_path / "parent"), bits=16,
        disable_flash_attn2=True, token_inventory=str(inventory_path), input_mode="text",
    )
    assert factory_calls[0][0] == "offline-tiny"
    assert factory_calls[0][1]["dtype"] == torch.bfloat16
    assert saved_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert evidence["semantic_tokens"]["token_ids"] == list(setup.token_ids)
    assert evidence["visual_precision"]["promoted_before_restore"]
    for name, value in load_file(tmp_path / "parent" / training.VISUAL_STATE_FILE).items():
        assert torch.equal(model.state_dict()[name], value)
        assert model.state_dict()[name].dtype == value.dtype
    assert all(not p.requires_grad for p in training.resolve_wrapped_module(model, components.vision).parameters())
    rows = [text_row(), text_row("board")]
    batches = []

    def inspect_inputs(module, args, kwargs):
        assert not training.TEXT_MEDIA_KEYS.intersection(kwargs)
        batches.append({key: kwargs[key].clone() for key in ("input_ids", "attention_mask")})

    handle = model.get_base_model().register_forward_pre_hook(inspect_inputs, with_kwargs=True)
    responses, logits = evaluator.generate_responses(
        model=model, processor=saved_tokenizer, rows=rows, max_new_tokens=1,
        input_mode="text", max_sequence_length=128, return_first_logits=return_logits,
    )
    handle.remove()
    assert responses == ["yes", "yes"]
    assert len(batches) == 1 and batches[0]["attention_mask"][1, 0] == 0
    with torch.no_grad():
        expected = source(**batches[0]).logits[:, -1].float()
    if return_logits:
        assert torch.equal(logits, expected)
    else:
        assert logits is None


def test_text_eval_writes_scores_summaries_without_resolving_images(tmp_path, monkeypatch):
    model, tokenizer, setup, _, _ = write_text_source(tmp_path / "parent")
    forbid_media(monkeypatch)
    source = tmp_path / "eval.jsonl"
    row = {**text_row(), "metadata": {"category": "spatial_direction", "eval_set_id": "offline"}}
    source.write_text(json.dumps(row) + "\n")
    args = evaluator.parse_args([
        "--eval-jsonl", str(source), "--output-dir", str(tmp_path / "out"),
        "--input-mode", "text", "--max-sequence-length", "128", "--max-new-tokens", "1",
        "--long-max-new-tokens", "1", "--image-root", "/must/not/be/resolved",
    ])
    summary = evaluator.run_eval_job(
        model=model, processor=tokenizer, adapter_evidence={"semantic_tokens": setup.as_dict()},
        args=args, eval_jsonl=str(source), image_variant="original", output_dir=tmp_path / "out",
    )
    assert summary["input_mode"] == "text" and summary["max_sequence_length"] == 128
    assert summary["image_root"] is None and not summary["truncation"]
    assert summary["exact_accuracy"] == summary["candidate_exact_accuracy"] == 1
    assert summary["precision"] == {
        "preserve_visual_fp32": True, "generation_autocast": {"policy": "caller_context"},
    }
    record = json.loads((tmp_path / "out" / "records.jsonl").read_text())
    assert record["metadata"]["category"] == "spatial_direction"
    assert record["metadata"]["input_mode"] == "text"
    assert record["score"]["scoring"] == "text_exact"


@pytest.mark.parametrize("extra", [
    {"image": "never.png"}, {"videos": []},
    {"messages": text_row("<image> board")["messages"]},
])
def test_text_eval_rejects_media_before_generation_or_path_resolution(tmp_path, monkeypatch, extra):
    forbid_media(monkeypatch)
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({**text_row(), **extra}) + "\n")
    args = evaluator.parse_args([
        "--eval-jsonl", str(source), "--output-dir", str(tmp_path / "out"),
        "--input-mode", "text", "--max-sequence-length", "128",
    ])
    with pytest.raises(ValueError, match="media"):
        evaluator.run_eval_job(
            model=TinyTextVLM(), processor=text_tokenizer(), adapter_evidence={}, args=args,
            eval_jsonl=str(source), image_variant="original", output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_generation_budget_counts_native_header_and_reserved_output(monkeypatch):
    forbid_media(monkeypatch)
    tokenizer, model, row = text_tokenizer(), TinyTextVLM(), text_row()
    prefix = training.text_chat_ids(tokenizer, row["messages"][:1], generation=True)
    budget = len(prefix) + 4
    evaluator.generate_responses(
        model=model, processor=tokenizer, rows=[row], input_mode="text",
        max_new_tokens=4, max_sequence_length=budget,
    )
    with pytest.raises(ValueError, match="generation budget.*truncation is forbidden"):
        evaluator.generate_responses(
            model=model, processor=tokenizer, rows=[row], input_mode="text",
            max_new_tokens=5, max_sequence_length=budget,
        )


def test_input_mode_does_not_reinterpret_existing_task_scores():
    gold = "<N00> <N01> <N02> <N03>"
    alternative = "<N00> <N05> <N04> <N03>"
    row = {**text_row(answer=gold), "metadata": {
        "task_type": "shortest_node_path", "target": {"start": "<N00>", "end": "<N03>"},
    }}
    metadata = evaluator.evaluation_metadata(row, image_variant="original")
    old = evaluator.score_response(gold, alternative, metadata=metadata)
    text = evaluator.score_response(gold, alternative, metadata={**metadata, "input_mode": "text"})
    assert old == text and text["correct"] and text["scoring"] == "shortest_node_path"
    plain = evaluator.score_response(gold, alternative)
    assert plain["scoring"] == "readout_items"  # Historical two-argument dispatch is retained.


def test_text_eval_entrypoint_enforces_pinned_runtime_before_model_load(monkeypatch):
    args = evaluator.parse_args([
        "--eval-jsonl", "e", "--output-dir", "o", "--input-mode", "text",
        "--max-sequence-length", "8192",
    ])
    monkeypatch.setattr(transformers, "__version__", "offline-unpinned")
    with pytest.raises(RuntimeError, match="runtime version mismatch"):
        evaluator.run_eval(args)


def test_text_model_load_requires_a_real_checkpoint_before_loading():
    with pytest.raises(ValueError, match="requires --adapter-dir"):
        evaluator.load_model(
            model_id="must-not-download", adapter_dir=None, bits=16,
            disable_flash_attn2=True, input_mode="text",
        )
