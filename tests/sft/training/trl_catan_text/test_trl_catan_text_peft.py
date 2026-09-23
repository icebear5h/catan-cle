"""Checkpoint loading, tokenizer mapping, and PEFT target selection."""

import json
from pathlib import Path

import peft
import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import (
    PreTrainedTokenizerFast,
)

from sft.scripts.train import train_trl_catan_vision as training

from .support import TinyTextVLM, load_text_source, text_row, write_text_source


@pytest.mark.parametrize("run_name", [
    "full-board-new-layouts-20260907", "spatial-continuation-20260912-r01",
])
def test_real_checkpoint_load_keeps_replacement_rows_visual_fp32_and_fresh_optimizer(tmp_path: Path, run_name: str) -> None:
    bundle = tmp_path / run_name / "checkpoints" / "checkpoint-128"
    source, _, setup, components, config = write_text_source(bundle)
    tokenizer = training.load_checkpoint_text_tokenizer(bundle, setup.tokens)
    loaded, receipt = load_text_source(bundle, tokenizer, config)
    assert isinstance(loaded, peft.PeftModel)
    loaded.eval()
    ids = torch.tensor([[setup.token_ids[0], 11, setup.token_ids[-1]]])
    with torch.no_grad():
        assert torch.equal(source(input_ids=ids).logits, loaded(input_ids=ids).logits)
    assert receipt["source_input_mode"] == "vision"
    assert not any(receipt[k] for k in (
        "optimizer_state_restored", "scheduler_state_restored", "rng_state_restored",
    ))
    for name, tensor in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    for path in (components.input_embedding, components.output_head):
        wrapper = training.resolve_wrapped_module(loaded, path)
        replacement = wrapper.token_adapter.trainable_tokens_delta["default"]
        assert replacement.shape == (154, 8)
        functional = wrapper.weight[list(setup.token_ids)]
        assert torch.equal(functional.float(), replacement.to(functional.dtype).float())
        base_rows = wrapper.token_adapter.get_base_layer().weight[list(setup.token_ids)]
        assert not torch.allclose(functional.float(), base_rows.float() + replacement.float())
    scope = training.audit_trainable_scope(loaded, components, setup, config, tmp_path / "scope.json")
    assert not scope["errors"]
    optimizer = training.build_optimizer(loaded, components, config)
    assert not optimizer.state
    assert {g["catan_name"]: g["lr"] for g in optimizer.param_groups} == {
        "token_rows": 5e-4, "language_lora": 1e-4,
    }
    coverage = training.audit_optimizer_coverage(loaded, optimizer, tmp_path / "coverage.json")
    assert coverage["trainable_tensors"] == coverage["covered_tensors"] == 6
    frozen = {name: p.clone() for name, p in loaded.named_parameters() if not p.requires_grad}
    batch = training.TextCompletionCollator(tokenizer, max_sequence_length=128)([
        text_row("board <N00>", "<N01>"),
    ])
    loaded(**batch).loss.backward()
    assert all(p.grad is not None for p in loaded.parameters() if p.requires_grad)
    optimizer.step()
    for name, parameter in loaded.named_parameters():
        if name in frozen:
            assert parameter.grad is None and torch.equal(parameter, frozen[name]), name


def test_actual_tokenizer_mapping_and_adapter_layout_are_validated(tmp_path: Path) -> None:
    _, tokenizer, setup, components, config = write_text_source(tmp_path / "parent")
    bundle = tmp_path / "parent"
    raw = json.loads(tokenizer.backend_tokenizer.to_str())
    atlas = [item for item in raw["added_tokens"] if item["content"] in setup.tokens]
    atlas[0]["content"], atlas[1]["content"] = atlas[1]["content"], atlas[0]["content"]
    swapped = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(raw)))
    with pytest.raises(ValueError, match="actual checkpoint tokenizer mapping"):
        training.validate_checkpoint_tokenizer(swapped, bundle, setup.tokens)
    adapter_path = bundle / "adapter_config.json"
    adapter = json.loads(adapter_path.read_text())
    adapter["target_modules"].append("model.visual.proj")
    adapter_path.write_text(json.dumps(adapter))
    with pytest.raises(ValueError, match="compatible language rank-8"):
        training.validate_text_adapter(TinyTextVLM(), bundle, setup, components, config=config)


def test_actual_peft_condensation_of_twenty_targets_loads_exact_language_modules(tmp_path: Path) -> None:
    bundle = tmp_path / "parent"
    source, tokenizer, setup, components, config = write_text_source(bundle, num_layers=10)
    intended = set(training.language_linear_targets(TinyTextVLM(10), components))
    assert len(intended) == 20
    saved_targets = set(json.loads((bundle / "adapter_config.json").read_text())["target_modules"])
    assert saved_targets == {"q_proj", "down_proj"}  # Produced by PEFT, not manually condensed.
    loaded, _ = load_text_source(bundle, tokenizer, config, num_layers=10)
    assert set(loaded.base_model.targeted_module_names) == intended
    loaded.eval()
    ids = torch.tensor([[setup.token_ids[0], 11]])
    with torch.no_grad():
        assert torch.equal(source(input_ids=ids).logits, loaded(input_ids=ids).logits)


@pytest.mark.parametrize("extra_path", ["model.visual", "model", "model.language_model"])
@pytest.mark.parametrize("linear", [True, False])
def test_suffix_selectors_cannot_admit_extra_visual_base_or_nonlinear_modules(tmp_path: Path, extra_path: str, linear: bool) -> None:
    bundle = tmp_path / "parent"
    _, _, setup, components, config = write_text_source(bundle, num_layers=10)
    base = TinyTextVLM(10)
    extra = base.get_submodule(extra_path)
    extra.q_proj = torch.nn.Linear(8, 8) if linear else torch.nn.Identity()
    with pytest.raises(ValueError, match="extra=.*q_proj"):
        training.validate_text_adapter(base, bundle, setup, components, config=config)


def test_peft_exclusions_and_layer_filters_cannot_hide_missing_language_targets(tmp_path: Path) -> None:
    bundle = tmp_path / "parent"
    _, _, setup, components, config = write_text_source(bundle, num_layers=10)
    path = bundle / "adapter_config.json"
    original = json.loads(path.read_text())
    for change in ({"exclude_modules": ["q_proj"]}, {"layers_to_transform": [0]}):
        path.write_text(json.dumps({**original, **change}))
        with pytest.raises(ValueError, match="missing=.*q_proj"):
            training.validate_text_adapter(TinyTextVLM(10), bundle, setup, components, config=config)


def test_real_peft_output_bridge_matches_direct_loss_and_gradients(tmp_path: Path) -> None:
    _, tokenizer, _, components, config = write_text_source(tmp_path / "parent")
    model, _ = load_text_source(tmp_path / "parent", tokenizer, config)
    model.float().eval()
    head = training.resolve_wrapped_module(model, components.output_head)
    replacement = head.token_adapter.trainable_tokens_delta["default"]
    hidden = torch.linspace(-1, 1, 32).reshape(4, 8).requires_grad_(True)
    labels = torch.tensor([11, 18, 19, 171])
    direct = F.cross_entropy(head(hidden).float(), labels)
    direct.backward()
    expected_hidden, expected_rows = hidden.grad.clone(), replacement.grad.clone()
    hidden.grad = replacement.grad = None
    original = model.get_base_model().get_output_embeddings()
    with training.expose_trainable_tokens_head_to_chunked_nll(model):
        view = model.get_base_model().get_output_embeddings()
        # Project separate chunks using the exact same functional weight TRL captures.
        weight = view.weight
        logits = torch.cat([F.linear(chunk, weight) for chunk in hidden.split(2)])
        chunked = F.cross_entropy(logits.float(), labels)
        chunked.backward()
    assert model.get_base_model().get_output_embeddings() is original
    torch.testing.assert_close(chunked, direct)
    torch.testing.assert_close(hidden.grad, expected_hidden)
    torch.testing.assert_close(replacement.grad, expected_rows)
    assert head.token_adapter.get_base_layer().weight.grad is None
