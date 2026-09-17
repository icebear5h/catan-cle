import argparse
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import peft
import pytest
import torch
import transformers
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BatchFeature, PretrainedConfig, PreTrainedTokenizerFast

import sft.modal_qwen_series_eval as launcher
import sft.scripts.eval_qwen_vl_adapter as evaluator
import sft.scripts.train_trl_catan_vision as trainer
from evals.catan_board_bench.tokens import semantic_recognition_token_inventory


class TinyVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig(tie_word_embeddings=False)
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.embed_tokens = torch.nn.Embedding(256, 4)
        self.model.language_model.proj = torch.nn.Linear(4, 4)
        self.model.visual = torch.nn.Module()
        self.model.visual.proj = torch.nn.Linear(4, 4)
        self.model.visual.merger = torch.nn.Linear(4, 4)
        self.model.visual.register_buffer("scale", torch.tensor(1.0001))
        self.model.visual.register_buffer("indices", torch.arange(2))
        self.lm_head = torch.nn.Linear(4, 256, bias=False)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, size, **kwargs):
        assert size == 256  # Like Qwen, the base already has padded vocabulary rows.
        return self.get_input_embeddings()

    def prepare_inputs_for_generation(self, **kwargs):
        return kwargs

    def forward(self, input_ids, **kwargs):
        return self.lm_head(self.model.language_model.proj(self.get_input_embeddings()(input_ids)))


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_load_preserves_checkpoint_after_peft_without_changing_lora_or_atlas(
    tmp_path, monkeypatch, preserve, source_dtype
):
    inventory = semantic_recognition_token_inventory()
    inventory_path = tmp_path / "tokens.json"
    inventory_path.write_text(json.dumps(inventory))
    raw_tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "node": 1, "edge": 2, "tile": 3, "port": 4}, unk_token="[UNK]")
    )
    raw_tokenizer.pre_tokenizer = Whitespace()
    processor = SimpleNamespace(
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, unk_token="[UNK]")
    )
    base = TinyVLM().bfloat16()
    setup, components = trainer.prepare_semantic_tokens(processor, base, inventory)
    wrapped = peft.get_peft_model(
        base,
        peft.LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            target_modules=["model.language_model.proj"],
            trainable_token_indices={
                components.input_embedding: list(setup.token_ids),
                components.output_head: list(setup.token_ids),
            },
        ),
    )
    visual = trainer.resolve_wrapped_module(wrapped, components.vision).float()
    with torch.no_grad():
        for parameter in visual.parameters():
            parameter.fill_(1.0001)  # Not representable in BF16.
        for name, parameter in wrapped.named_parameters():
            if ".lora_" in name or "trainable_tokens_delta" in name:
                parameter.fill_(0.125)
    adapter_state = {
        name: value.clone()
        for name, value in wrapped.state_dict().items()
        if ".lora_" in name or "trainable_tokens_delta" in name
    }
    assert len(adapter_state) == 4
    checkpoint = tmp_path / "checkpoint-128"
    wrapped.save_pretrained(checkpoint, save_embedding_layers=False)
    # The final-bundle writer casts buffers too; periodic checkpoints preserve them.
    trainer.save_visual_state(
        wrapped,
        components,
        checkpoint,
        dtype=None if source_dtype == torch.float32 else source_dtype,
    )
    source = load_file(checkpoint / trainer.VISUAL_STATE_FILE)
    fresh = TinyVLM().bfloat16()
    base_state = {name: value.clone() for name, value in fresh.state_dict().items()}
    factory = Mock(return_value=fresh)
    monkeypatch.setattr(
        transformers,
        "AutoModelForMultimodalLM",
        SimpleNamespace(from_pretrained=factory),
        raising=False,
    )
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor)
    shared_loader = trainer.load_visual_state
    load_events = []

    def checked_load(model, path):
        assert isinstance(model, peft.PeftModel)
        assert any("trainable_tokens_delta" in name for name, _ in model.named_parameters())
        dtype = trainer.resolve_wrapped_module(model, components.vision).proj.weight.dtype
        load_events.append(dtype)
        return shared_loader(model, path)

    monkeypatch.setattr(evaluator, "load_visual_state", checked_load)
    options = {"preserve_visual_fp32": True} if preserve else {}
    model, _, evidence = evaluator.load_model(
        model_id="offline-tiny",
        adapter_dir=str(checkpoint),
        bits=16,
        disable_flash_attn2=True,
        token_inventory=str(inventory_path),
        **options,
    )
    expected_dtype = torch.float32 if preserve else torch.bfloat16
    assert load_events == [expected_dtype]  # Exactly one restore, already at the right dtype.
    assert factory.call_args.kwargs["dtype"] == torch.bfloat16
    assert trainer.load_visual_state is shared_loader  # No mutation of the training module.
    state = model.state_dict()
    for name, value in source.items():
        expected = value.to(state[name].dtype)
        assert torch.equal(state[name], expected)
        if value.is_floating_point() and not name.endswith("indices"):
            assert state[name].dtype == expected_dtype
    for name, value in adapter_state.items():
        assert state[name].dtype == value.dtype
        assert torch.equal(state[name], value)
    for name in (
        "model.language_model.proj.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    ):
        module, _, leaf = name.rpartition(".")
        layer = trainer.resolve_wrapped_module(model, module)
        if hasattr(layer, "token_adapter"):
            layer = layer.token_adapter
        value = getattr(layer.get_base_layer(), leaf)
        assert value.dtype == torch.bfloat16
        assert torch.equal(value, base_state[name])
    assert evidence["semantic_tokens"]["token_ids"] == list(setup.token_ids)
    assert len(setup.token_ids) == 154
    assert evidence["visual_state"]["sha256"] == trainer.sha256_file(
        checkpoint / trainer.VISUAL_STATE_FILE
    )
    assert not model.training
    key = "base_model.model.model.visual.proj.weight"
    if preserve:
        assert torch.equal(state[key], source[key].float())
        precision = evidence["visual_precision"]
        assert precision["base_load_dtype"] == "torch.bfloat16"
        assert precision["promoted_before_restore"] is True
        assert precision["loaded_dtypes"] == {"torch.float32": 5, "torch.int64": 1}
        assert precision["source_dtypes"] == (
            {"F32": 5, "I64": 1} if source_dtype == torch.float32 else {"BF16": 6}
        )
    else:
        assert "visual_precision" not in evidence
        if source_dtype == torch.float32:
            assert not torch.equal(state[key].float(), source[key])


@pytest.mark.parametrize("adapter", [None, "missing-checkpoint"])
def test_preservation_requires_visual_source_before_loading_model(adapter):
    with pytest.raises(ValueError, match="requires an --adapter-dir containing"):
        evaluator.load_model(
            model_id="must-not-download",
            adapter_dir=adapter,
            bits=16,
            disable_flash_attn2=True,
            preserve_visual_fp32=True,
        )


class GenerationModel(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self, preserve):
        super().__init__()
        self.visual = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.visual.weight.copy_(torch.tensor([[2.0001, 0], [1.0001, 0]]))
        if not preserve:
            self.visual.bfloat16()
        self.contexts = []

    def generate(self, input_ids, pixel_values, output_logits, **kwargs):
        self.contexts.append((torch.is_autocast_enabled("cpu"), torch.is_inference_mode_enabled()))
        logits = self.visual(pixel_values.to(self.visual.weight.dtype))
        return SimpleNamespace(
            sequences=torch.cat((input_ids, logits.argmax(-1, keepdim=True)), dim=1),
            logits=(logits,) if output_logits else None,
        )


class GenerationProcessor:
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [{"yes": 0, "no": 1}[text]])

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "prompt"

    def __call__(self, text, **kwargs):
        return BatchFeature(
            {
                "input_ids": torch.tensor([[1, 2]] * len(text)),
                "pixel_values": torch.tensor([[1.0, 0.0]] * len(text)),
            }
        )

    def batch_decode(self, sequences, **kwargs):
        return ["yes" if sequence.tolist() == [0] else "no" for sequence in sequences]


@pytest.mark.parametrize("preserve,outer_autocast", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("candidate_scoring", [False, True])
def test_eval_generation_and_candidates_share_autocast_and_write_receipt(
    tmp_path, monkeypatch, preserve, outer_autocast, candidate_scoring
):
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


@pytest.mark.parametrize("preserve", [False, True])
def test_cli_forwards_opt_in_to_model_loading(tmp_path, monkeypatch, preserve):
    argv = ["eval", "--eval-jsonl", "spatial.jsonl", "--output-dir", str(tmp_path)]
    if preserve:
        argv.append("--preserve-visual-fp32")
    monkeypatch.setattr(sys, "argv", argv)
    args = evaluator.parse_args()
    assert args.preserve_visual_fp32 is preserve
    load = Mock(return_value=(object(), object(), {}))
    job = Mock(return_value={"rows": 1, "exact_accuracy": 1})
    monkeypatch.setattr(evaluator, "load_model", load)
    monkeypatch.setattr(evaluator, "run_eval_job", job)
    evaluator.run_eval(args)
    assert load.call_args.kwargs["preserve_visual_fp32"] is preserve
    assert job.call_args.kwargs["args"].preserve_visual_fp32 is preserve


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("remote_name", ["eval_remote", "eval_h200"])
def test_remote_forwards_precision_through_subprocess_command(
    tmp_path, monkeypatch, preserve, remote_name
):
    run = Mock()
    commit = Mock()
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "sft_runs", SimpleNamespace(commit=commit))
    options = {"preserve_visual_fp32": True} if preserve else {}
    getattr(launcher, remote_name).get_raw_f()(
        eval_jsonl="/data/spatial.jsonl",
        output_dir=str(tmp_path),
        adapter_dir="/runs/checkpoint-128",
        token_inventory="/data/tokens.json",
        **options,
    )
    command = run.call_args.args[0]
    assert ("--preserve-visual-fp32" in command) is preserve
    assert command[:3] == ["python", "-m", "sft.scripts.eval_qwen_vl_adapter"]
    assert run.call_args.kwargs == {"check": True}
    commit.assert_called_once_with()


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("gpu,spawn", [("l40s", False), ("h200", True)])
def test_series_entrypoint_forwards_precision_without_paid_calls(monkeypatch, preserve, gpu, spawn):
    remote = Mock(return_value={})
    spawn_call = Mock(return_value=SimpleNamespace(object_id="offline-test"))
    function = SimpleNamespace(remote=remote, spawn=spawn_call)
    monkeypatch.setattr(launcher, "eval_remote", function)
    monkeypatch.setattr(launcher, "eval_h200", function)
    monkeypatch.setattr(
        launcher,
        "upload_eval_jsonl",
        Mock(return_value=("/data/spatial.jsonl", "/data/tokens.json")),
    )
    options = {"preserve_visual_fp32": True} if preserve else {}
    launcher.main(eval_jsonl="spatial.jsonl", gpu=gpu, spawn_eval=spawn, **options)
    called, unused = (spawn_call, remote) if spawn else (remote, spawn_call)
    assert called.call_args.kwargs["preserve_visual_fp32"] is preserve
    unused.assert_not_called()
