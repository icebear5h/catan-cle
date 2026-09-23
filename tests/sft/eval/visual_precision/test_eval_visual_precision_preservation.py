"""Checkpoint preservation across PEFT loading."""

import json
from pathlib import Path
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
from transformers import PreTrainedTokenizerFast

import sft.scripts.eval.eval_qwen_vl_adapter as evaluator
import sft.scripts.train.train_trl_catan_vision as trainer
from evals.catan_board_bench.tokens import semantic_recognition_token_inventory

from .support import TinyVLM


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_load_preserves_checkpoint_after_peft_without_changing_lora_or_atlas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve: bool,
    source_dtype: torch.dtype,
) -> None:
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
    load_events: list[torch.dtype] = []

    def checked_load(model: peft.PeftModel, path: Path) -> object:
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
def test_preservation_requires_visual_source_before_loading_model(adapter: str | None) -> None:
    with pytest.raises(ValueError, match="requires an --adapter-dir containing"):
        evaluator.load_model(
            model_id="must-not-download",
            adapter_dir=adapter,
            bits=16,
            disable_flash_attn2=True,
            preserve_visual_fp32=True,
        )
