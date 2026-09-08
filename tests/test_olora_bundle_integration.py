"""Exercise parent merge, visual restoration and reload with actual PEFT wrappers."""

import copy
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sft.scripts.train_trl_catan_vision import (
    FROZEN_ADAPTER_DIR,
    ModelComponents,
    PROFILE_OLORA_FROZEN_BUNDLE,
    TokenSetup,
    TrainConfig,
    apply_frozen_adapter,
    audit_trainable_scope,
    build_optimizer,
    freeze_visual_except_lora,
    language_linear_targets,
    load_frozen_bundle,
    load_visual_state,
    orthogonal_penalty,
    resolve_wrapped_module,
    save_visual_state,
    vision_linear_targets,
)


class _LanguageContainer(torch.nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)


@pytest.mark.parametrize("base_dtype", [torch.float32, torch.bfloat16])
def test_frozen_parent_initialization_update_and_reload(tmp_path: Path, base_dtype):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=192, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=False,
    ))
    base.model = _LanguageContainer(base.model)
    visual = torch.nn.Module()
    visual.attn = torch.nn.Module()
    visual.attn.qkv = torch.nn.Linear(16, 16)
    visual.attn.proj = torch.nn.Linear(16, 16)
    visual.merger = torch.nn.Module()
    visual.merger.linear_fc1 = torch.nn.Linear(16, 16)
    visual.merger.linear_fc2 = torch.nn.Linear(16, 16)
    base.model.visual = visual
    base.to(base_dtype)
    original = copy.deepcopy(base)
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens", output_head="lm_head",
        language="model.language_model", vision="model.visual", merger="model.visual.merger",
        vocab_size=192, hidden_size=16,
    )
    ids = tuple(range(38, 192))
    setup = TokenSetup(tuple(f"<T{i}>" for i in ids), ids, 192, 192, 154)
    parent = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0,
        target_modules=language_linear_targets(base, components),
        trainable_token_indices={components.input_embedding: list(ids), "lm_head": list(ids)},
    ))
    parent._catan_components = components
    resolve_wrapped_module(parent, components.vision).float()
    with torch.no_grad():
        for name, parameter in parent.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.01)
            if "trainable_tokens_delta" in name:
                parameter.add_(0.013)
        # Deliberately non-BF16 values catch restoration through BF16 storage.
        for parameter in visual.parameters():
            parameter.add_(0.0001234)
    visual_expected = {name: value.clone() for name, value in visual.state_dict().items()}
    parent_dir = tmp_path / "parent"
    parent.save_pretrained(parent_dir, save_embedding_layers=False)
    save_visual_state(parent, components, parent_dir)
    (parent_dir / "trainable_parameters.json").write_text(json.dumps({
        "semantic_tokens": {"token_ids": ids},
    }))
    factors = tmp_path / "factors.safetensors"
    targets = vision_linear_targets(original, components)
    save_file({name + ".lora_A.weight": torch.randn(2, 16) for name in targets}, str(factors))
    config = TrainConfig(
        train_jsonl="unused", image_root="unused", token_inventory="unused",
        output_dir=str(tmp_path / "run"), profile=PROFILE_OLORA_FROZEN_BUNDLE,
        frozen_bundle=str(parent_dir), visual_delta_factors=str(factors),
        token_init="keep", lora_rank=4, lora_alpha=8, lora_dropout=0,
    )

    model, report = load_frozen_bundle(copy.deepcopy(original), setup, components, config)
    assert report["new_adapter"]["vision_targets"] == len(targets) == 4
    assert report["new_adapter"]["language_targets"] == 7
    assert report["protected_modules"] == 11
    assert report["unprotected_modules"] == []
    assert audit_trainable_scope(model, components, setup, config, tmp_path / "scope.json")["errors"] == []
    current_visual = resolve_wrapped_module(model, components.vision)
    for name, value in current_visual.state_dict().items():
        if "lora_" not in name:
            assert value.dtype == torch.float32
            assert torch.equal(value, visual_expected[name.replace(".base_layer", "")])

    # Compare against the parent merge itself to isolate fresh-adapter identity
    # from the small numerical error of folding BF16 language weights.
    reference_base = copy.deepcopy(original)
    reference_base._catan_components = components
    reference, _ = apply_frozen_adapter(reference_base, parent_dir, restore_visual=True)
    tokens = torch.tensor([[38, 39, 40, 41]])
    pixels = torch.randn(1, 4, 16)

    def logits(target):
        vision = resolve_wrapped_module(target, components.vision)
        features = vision.attn.proj(vision.attn.qkv(pixels))
        features = vision.merger.linear_fc2(vision.merger.linear_fc1(features))
        embeddings = target.get_input_embeddings()(tokens)
        return target(inputs_embeds=embeddings + features.to(embeddings.dtype)).logits.float()

    model.eval()
    reference.eval()
    with torch.no_grad():
        torch.testing.assert_close(logits(model), logits(reference), rtol=0, atol=0)

    frozen_before = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
    trainable_before = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = build_optimizer(model, components, config)
    penalty, _, unprotected = orthogonal_penalty(model, model._catan_orthogonal_bases, {})
    loss = torch.nn.functional.cross_entropy(logits(model).reshape(-1, 192), tokens.reshape(-1)) + config.orthogonal_lambda * penalty
    assert torch.isfinite(loss) and unprotected == 0
    loss.backward()
    optimizer.step()
    for name, p in model.named_parameters():
        if name in frozen_before:
            assert torch.equal(p, frozen_before[name]), name
    for marker in ("visual", "layers", "trainable_tokens_delta"):
        assert any(marker in name and not torch.equal(p, trainable_before[name])
                   for name, p in model.named_parameters() if name in trainable_before)

    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint, save_embedding_layers=False)
    save_visual_state(model, components, checkpoint)
    shutil.copytree(parent_dir, checkpoint / FROZEN_ADAPTER_DIR)
    reload_base, _ = apply_frozen_adapter(copy.deepcopy(original), checkpoint / FROZEN_ADAPTER_DIR)
    reloaded = PeftModel.from_pretrained(reload_base, checkpoint)
    reloaded._catan_components = components
    freeze_visual_except_lora(reloaded, components)
    load_visual_state(reloaded, checkpoint)
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(logits(reloaded), logits(model), rtol=0, atol=0)

    # Retain strict failure for corrupt/incomplete visual state.
    save_file({"wrong.weight": torch.ones(2, 2)}, str(checkpoint / "visual_model.safetensors"))
    with pytest.raises(RuntimeError, match="visual checkpoint key mismatch"):
        load_visual_state(reloaded, checkpoint)
