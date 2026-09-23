"""Module resolution, parameter categories, and bundle round trips."""
import json
from dataclasses import asdict
from pathlib import Path

import peft
import pytest
import torch

from sft.scripts.train.train_trl_catan_vision import (
    PROFILE_VISION_TOKENS_LORA,
    RUN_CONFIG_FILE,
    TRAINABLE_SCOPE_FILE,
    ModelComponents,
    TokenSetup,
    TrainConfig,
    build_optimizer,
    load_initial_bundle,
    load_visual_state,
    parameter_category,
    promote_visual_master_weights,
    resolve_wrapped_module,
    save_visual_state,
)

from .support import _NestedModel, _OptimizerModel


def test_wrapped_module_resolution_does_not_match_every_child() -> None:
    model = _NestedModel()

    resolved = resolve_wrapped_module(model, "model.visual")

    assert resolved is model.base_model.model.model.visual


def test_parameter_categories_separate_merger_from_rest_of_vision() -> None:
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=16,
    )

    assert parameter_category(
        "base_model.model.model.visual.blocks.0.weight", components
    ) == "vision"
    assert parameter_category(
        "base_model.model.model.visual.merger.linear.weight", components
    ) == "merger"
    assert parameter_category(
        "base_model.model.lm_head.trainable_tokens_delta.default", components
    ) == "atlas_output_rows"
    assert parameter_category(
        "base_model.model.model.language_model.layers.0.q_proj.weight", components
    ) == "forbidden"


def test_visual_state_round_trip_is_complete(tmp_path: Path) -> None:
    model = _NestedModel()
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=16,
    )
    model._catan_components = components
    expected_weight = model.base_model.model.model.visual[0].weight.detach().clone()
    saved = save_visual_state(model, components, tmp_path)
    model.base_model.model.model.visual[0].weight.data.zero_()

    loaded = load_visual_state(model, tmp_path)

    assert saved["tensors"] == loaded["expected_tensors"] == loaded["tensors"]
    assert torch.equal(model.base_model.model.model.visual[0].weight, expected_weight)


def test_initial_bundle_preserves_fp32_deltas_when_base_is_bf16(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _NestedModel()
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens", output_head="lm_head",
        language="model.language_model", vision="model.visual", merger="model.visual.merger",
        vocab_size=128, hidden_size=16,
    )
    expected = torch.full_like(model.base_model.model.model.visual[0].weight, 1.0001)
    model.base_model.model.model.visual[0].weight.data.copy_(expected)
    save_visual_state(model, components, tmp_path)
    config = TrainConfig(train_jsonl="train", image_root="images", token_inventory="tokens",
                         output_dir="output", initial_bundle=str(tmp_path), token_init="keep")
    (tmp_path / RUN_CONFIG_FILE).write_text(json.dumps(asdict(config)))
    (tmp_path / TRAINABLE_SCOPE_FILE).write_text(json.dumps({"semantic_tokens": {"token_ids": [10]}}))
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"mock adapter")
    model.bfloat16()
    assert not torch.equal(model.base_model.model.model.visual[0].weight.float(), expected)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda *args, **kwargs: model)
    setup = TokenSetup(tokens=("<N00>",), token_ids=(10,), tokenizer_size=128,
                       model_vocab_size=128, added_tokens=1)
    loaded, report = load_initial_bundle(model, setup, components, config)
    assert torch.equal(loaded.base_model.model.model.visual[0].weight, expected)
    assert not report["optimizer_state_restored"]


def test_production_default_uses_lower_lr_language_lora_group(tmp_path: Path) -> None:
    config = TrainConfig(
        train_jsonl="train.jsonl",
        image_root="images",
        token_inventory="tokens.json",
        output_dir=str(tmp_path),
    )
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=2,
    )

    optimizer = build_optimizer(_OptimizerModel(), components, config)
    rates = {group["catan_name"]: group["lr"] for group in optimizer.param_groups}

    assert config.profile == PROFILE_VISION_TOKENS_LORA
    assert rates == {
        "vision": 5e-6,
        "merger": 5e-5,
        "token_rows": 5e-4,
        "language_lora": 1e-4,
    }
    assert config.warmup_ratio == 0.1


def test_visual_promotion_yields_fp32_trainable_master_weights() -> None:
    model = _NestedModel().to(torch.bfloat16)
    model.requires_grad_(False)
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=16,
    )

    report = promote_visual_master_weights(model, components)

    visual = model.base_model.model.model.visual
    assert all(p.dtype == torch.float32 and p.requires_grad for p in visual.parameters())
    assert report["dtypes"] == {"torch.float32": 2}


def test_visual_state_can_be_saved_in_bf16_for_the_final_bundle(tmp_path: Path) -> None:
    model = _NestedModel()
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=16,
    )

    (tmp_path / "ckpt").mkdir()
    (tmp_path / "final").mkdir()
    fp32 = save_visual_state(model, components, tmp_path / "ckpt")
    bf16 = save_visual_state(model, components, tmp_path / "final", dtype=torch.bfloat16)

    assert fp32["dtype"] == "torch.float32"
    assert bf16["dtype"] == "torch.bfloat16"
    assert bf16["bytes"] < fp32["bytes"]
