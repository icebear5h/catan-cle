"""LoRA targets, protected bases, and frozen bundle loading."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sft.scripts.train.train_trl_catan_vision import (
    PROFILE_OLORA_FROZEN_BUNDLE,
    ModelComponents,
    TrainConfig,
    load_frozen_bundle,
    load_protected_bases,
    module_key,
    orthogonal_penalty,
    orthonormal_rows,
    parameter_category,
    vision_linear_targets,
)

from .support import _TINY_COMPONENTS, _TINY_SETUP, _AdapterHolder, _TinyVLM, _write_parent_bundle


def test_olora_config_and_token_init_keep() -> None:
    base = dict(train_jsonl="t.jsonl", image_root="images", token_inventory="tokens.json", output_dir="out")
    config = TrainConfig(**base, profile=PROFILE_OLORA_FROZEN_BUNDLE, frozen_bundle="/runs/parent", token_init="keep")
    config.validate()
    assert config.olora and config.language_lora
    with pytest.raises(ValueError, match="frozen_bundle"):
        TrainConfig(**base, profile=PROFILE_OLORA_FROZEN_BUNDLE, token_init="keep").validate()
    with pytest.raises(ValueError, match="token_init=keep"):
        TrainConfig(**base, profile=PROFILE_OLORA_FROZEN_BUNDLE, frozen_bundle="/runs/parent").validate()
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainConfig(**base, profile=PROFILE_OLORA_FROZEN_BUNDLE, frozen_bundle="/runs/parent", initial_bundle="/runs/x", token_init="keep").validate()
    with pytest.raises(ValueError, match="olora profile"):
        TrainConfig(**base, frozen_bundle="/runs/parent").validate()


def test_module_key_and_orthonormal_rows() -> None:
    assert module_key("base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.default.weight") == "model.language_model.layers.0.mlp.down_proj"
    assert module_key("model.visual.blocks.3.attn.qkv.lora_A.weight") == "model.visual.blocks.3.attn.qkv"
    assert module_key("base_model.model.model.visual.merger.linear_fc1.lora_B.default.weight") == "model.visual.merger.linear_fc1"
    rows = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]])
    basis = orthonormal_rows(rows)
    assert basis.shape == (2, 4)
    assert torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-6)
    assert torch.allclose(basis[:, 2:], torch.zeros(2, 2), atol=1e-6)


def test_orthogonal_penalty_measures_projection_onto_protected_rows() -> None:
    protected = orthonormal_rows(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
    inside = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    outside = torch.tensor([[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    bases = {"model.layer": protected}
    cache: dict[str, torch.Tensor] = {}
    penalty, fraction, unprotected = orthogonal_penalty(_AdapterHolder(inside, inside), bases, cache)
    assert torch.isclose(penalty, torch.tensor(25.0)) and fraction == pytest.approx(1.0) and unprotected == 1
    penalty, fraction, _ = orthogonal_penalty(_AdapterHolder(outside, inside), bases, cache)
    assert torch.isclose(penalty, torch.tensor(0.0)) and fraction == pytest.approx(0.0)
    mixed = inside + outside
    penalty, fraction, _ = orthogonal_penalty(_AdapterHolder(mixed, inside), bases, cache)
    assert torch.isclose(penalty, torch.tensor(25.0)) and fraction == pytest.approx(0.5)
    penalty.backward()
    assert cache["model.layer"] is not None
    with pytest.raises(RuntimeError, match="protected basis"):
        orthogonal_penalty(_AdapterHolder(inside, inside), {"model.nothing": protected}, {})


def test_load_protected_bases_merges_adapter_and_factor_rows(tmp_path: Path) -> None:
    adapter = tmp_path / "frozen"
    adapter.mkdir()
    save_file(
        {
            "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.default.weight": torch.eye(2, 6),
            "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.default.weight": torch.zeros(6, 2),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    factors = tmp_path / "factors.safetensors"
    save_file({"model.visual.blocks.0.attn.qkv.lora_A.weight": torch.tensor([[0.0, 0.0, 0.0, 2.0]]), "model.visual.blocks.0.attn.qkv.lora_B.weight": torch.zeros(4, 1)}, str(factors))
    bases = load_protected_bases(adapter, factors)
    assert set(bases) == {"model.language_model.layers.0.mlp.down_proj", "model.visual.blocks.0.attn.qkv"}
    assert bases["model.language_model.layers.0.mlp.down_proj"].shape == (2, 6)
    assert torch.allclose(bases["model.visual.blocks.0.attn.qkv"].abs(), torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_vision_lora_targets_and_categories() -> None:
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        hidden_size=8,
        vocab_size=32,
    )
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Module()
    model.model.visual.blocks = torch.nn.ModuleList([torch.nn.Module()])
    block = model.model.visual.blocks[0]
    block.attn = torch.nn.Module()
    block.attn.qkv = torch.nn.Linear(4, 12)
    block.attn.proj = torch.nn.Linear(4, 4)
    block.norm1 = torch.nn.LayerNorm(4)
    block.mlp = torch.nn.Module()
    block.mlp.linear_fc1 = torch.nn.Linear(4, 8)
    block.mlp.linear_fc2 = torch.nn.Linear(8, 4)
    model.model.visual.merger = torch.nn.Module()
    model.model.visual.merger.linear_fc1 = torch.nn.Linear(4, 4)
    model.model.visual.merger.linear_fc2 = torch.nn.Linear(4, 4)
    model.model.visual.patch_embed = torch.nn.Module()
    model.model.visual.patch_embed.proj = torch.nn.Linear(4, 4)
    targets = vision_linear_targets(model, components)
    assert targets == [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.blocks.0.mlp.linear_fc2",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
    ]
    assert parameter_category("base_model.model.model.visual.blocks.0.attn.qkv.lora_A.default.weight", components) == "vision_lora"
    assert parameter_category("base_model.model.model.visual.merger.linear_fc1.lora_B.default.weight", components) == "vision_lora"
    assert parameter_category("base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight", components) == "language_lora"
    assert parameter_category("base_model.model.model.visual.blocks.0.norm1.weight", components) == "vision"


def test_load_frozen_bundle_restores_vision_merges_parent_and_adds_fresh_adapters(tmp_path: Path) -> None:
    bundle = tmp_path / "checkpoint-384"
    parent_state, parent_visual = _write_parent_bundle(bundle)
    config = TrainConfig(
        train_jsonl="train.jsonl",
        image_root="images",
        token_inventory="tokens.json",
        output_dir=str(tmp_path / "out"),
        profile=PROFILE_OLORA_FROZEN_BUNDLE,
        frozen_bundle=str(bundle),
        token_init="keep",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )
    config.validate()
    base = _TinyVLM(seed=2).bfloat16()
    for name in parent_visual:
        assert not torch.equal(base.state_dict()[name].float(), parent_visual[name]), name

    model, report = load_frozen_bundle(base, _TINY_SETUP, _TINY_COMPONENTS, config)

    state = model.state_dict()
    # Vision: the parent's fp32 weights land exactly, no bf16 round trip.
    for name, expected in parent_visual.items():
        module, _, leaf = name.rpartition(".")
        loaded = state.get(f"base_model.model.{name}", state.get(f"base_model.model.{module}.base_layer.{leaf}"))
        assert loaded is not None, name
        assert loaded.dtype == torch.float32
        assert torch.equal(loaded, expected), name
    # Language and atlas rows: the parent's LoRA and token deltas are merged into the base weights.
    for name in ("model.language_model.layers.0.q_proj", "model.language_model.layers.0.down_proj"):
        assert torch.allclose(state[f"base_model.model.{name}.base_layer.weight"].float(), parent_state[f"{name}.weight"], atol=2e-2)
    embed_key = [name for name in state if "embed_tokens" in name and name.endswith(".weight") and "trainable_tokens" not in name]
    assert len(embed_key) == 1, embed_key
    embed = state[embed_key[0]].float()
    assert torch.allclose(embed[14:], parent_state["model.language_model.embed_tokens.weight"][14:], atol=2e-2)
    # Fresh adapters on every language linear and every vision projection, counted before PEFT renamed them.
    assert report["new_adapter"] == {"rank": 2, "alpha": 4, "language_targets": 2, "vision_targets": 6}
    lora_a = {name for name, parameter in model.named_parameters() if ".lora_A." in name and parameter.requires_grad}
    assert len(lora_a) == 8
    assert "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.default.weight" in lora_a
    assert "base_model.model.model.visual.patch_embed.proj.lora_A.default.weight" not in lora_a
    assert report["protected_modules"] == 2
    assert report["unprotected_modules"] == [
        "model.visual.blocks.0.attn.proj",
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.blocks.0.mlp.linear_fc2",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
    ]
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all(".lora_" in name or "trainable_tokens_delta" in name for name in trainable)
    assert not any(".lora_" in name for name in parent_state)
    logits = model(input_ids=torch.tensor([[1, 14]]), pixels=torch.zeros(1, 2, 4))
    assert logits.shape == (1, 2, 16)
