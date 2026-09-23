"""Shared helpers for vision-mode trl training contracts, bundles, and objectives."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import PretrainedConfig

from sft.scripts.train.train_trl_catan_vision import (
    ModelComponents,
    TokenSetup,
    language_linear_targets,
    save_visual_state,
)


def _row(image: str, stage: str | None) -> dict:
    row = {
        "images": [image],
        "messages": [
            {"role": "user", "content": "<image>\nIs <N00> above <N01>?"},
            {"role": "assistant", "content": "yes"},
        ],
    }
    if stage is not None:
        row["curriculum_stage"] = stage
    return row


def _write_dataset(tmp_path: Path, stages: list[str | None]) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True)
    train_jsonl = tmp_path / "train.jsonl"
    with train_jsonl.open("w") as handle:
        for index, stage in enumerate(stages):
            image_name = f"{index}.png"
            (image_root / image_name).write_bytes(b"not-decoded-by-contract-audit")
            handle.write(json.dumps(_row(image_name, stage)) + "\n")
    return train_jsonl, image_root


class _NestedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.model = torch.nn.Module()
        self.base_model.model.model.visual = torch.nn.Sequential(torch.nn.Linear(2, 2))


class _OptimizerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.model = torch.nn.Module()
        visual = torch.nn.Module()
        visual.block = torch.nn.Linear(2, 2)
        visual.merger = torch.nn.Linear(2, 2)
        self.base_model.model.model.visual = visual
        language = torch.nn.Module()
        language.embed_tokens = torch.nn.Module()
        language.embed_tokens.register_parameter(
            "trainable_tokens_delta", torch.nn.Parameter(torch.ones(154, 2))
        )
        language.layers = torch.nn.ModuleList([torch.nn.Module()])
        language.layers[0].q_proj = torch.nn.Module()
        language.layers[0].q_proj.register_parameter(
            "lora_A", torch.nn.Parameter(torch.ones(2, 2))
        )
        self.base_model.model.model.language_model = language
        self.base_model.model.lm_head = torch.nn.Module()
        self.base_model.model.lm_head.register_parameter(
            "trainable_tokens_delta", torch.nn.Parameter(torch.ones(154, 2))
        )


class _EmbeddingModel(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.model.language_model.embed_tokens.weight[-154:] = 7.0
            self.lm_head.weight[-154:] = -7.0


class _FakeTrainableTokensHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = torch.nn.Linear(3, 7, bias=False)
        self.base_layer.requires_grad_(False)
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False
        self.indices = torch.tensor([1, 5])
        self.delta = torch.nn.Parameter(self.base_layer.weight[self.indices].detach().clone())

    def get_base_layer(self) -> torch.nn.Module:
        return self.base_layer

    def get_merged_weights(self, active_adapters: list[str]) -> torch.Tensor:
        assert active_adapters == ["default"]
        return self.base_layer.weight.index_copy(0, self.indices, self.delta)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.get_merged_weights(self.active_adapters))


class _FakeTrainableTokensWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_adapter = _FakeTrainableTokensHead()

    @property
    def weight(self) -> torch.Tensor:
        return self.token_adapter.get_merged_weights(self.token_adapter.active_adapters)


class _FakeCollator:
    def __call__(self, examples: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1], [2]])}


class _AdapterHolder(torch.nn.Module):
    def __init__(self, a_new: torch.Tensor, frozen: torch.Tensor) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layer = torch.nn.Module()
        self.model.layer.lora_A = torch.nn.Module()
        self.model.layer.lora_A.default = torch.nn.Module()
        self.model.layer.lora_A.default.weight = torch.nn.Parameter(a_new.clone())
        self.model.other = torch.nn.Module()
        self.model.other.lora_A = torch.nn.Module()
        self.model.other.lora_A.default = torch.nn.Module()
        self.model.other.lora_A.default.weight = torch.nn.Parameter(frozen.clone(), requires_grad=False)
        self.model.free = torch.nn.Module()
        self.model.free.lora_A = torch.nn.Module()
        self.model.free.lora_A.default = torch.nn.Module()
        self.model.free.lora_A.default.weight = torch.nn.Parameter(a_new.clone())


class _TinyVLM(torch.nn.Module):
    """Qwen-shaped module tree: language layers, a vision tower with the O-LoRA suffixes, tied-shape heads."""

    def __init__(self, vocab: int = 16, hidden: int = 4, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = PretrainedConfig(tie_word_embeddings=False)
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.embed_tokens = torch.nn.Embedding(vocab, hidden)
        layer = torch.nn.Module()
        layer.q_proj = torch.nn.Linear(hidden, hidden)
        layer.down_proj = torch.nn.Linear(hidden, hidden)
        self.model.language_model.layers = torch.nn.ModuleList([layer])
        self.model.visual = torch.nn.Module()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.qkv = torch.nn.Linear(hidden, hidden)
        block.attn.proj = torch.nn.Linear(hidden, hidden)
        block.norm1 = torch.nn.LayerNorm(hidden)
        block.mlp = torch.nn.Module()
        block.mlp.linear_fc1 = torch.nn.Linear(hidden, hidden)
        block.mlp.linear_fc2 = torch.nn.Linear(hidden, hidden)
        self.model.visual.blocks = torch.nn.ModuleList([block])
        self.model.visual.merger = torch.nn.Module()
        self.model.visual.merger.linear_fc1 = torch.nn.Linear(hidden, hidden)
        self.model.visual.merger.linear_fc2 = torch.nn.Linear(hidden, hidden)
        self.model.visual.patch_embed = torch.nn.Module()
        self.model.visual.patch_embed.proj = torch.nn.Linear(hidden, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def prepare_inputs_for_generation(self, *args: object, **kwargs: object) -> dict[str, object]:
        return kwargs

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        block = self.model.visual.blocks[0]
        visual = block.norm1(block.attn.proj(block.attn.qkv(pixels)))
        visual = block.mlp.linear_fc2(block.mlp.linear_fc1(visual))
        visual = self.model.visual.merger.linear_fc2(self.model.visual.merger.linear_fc1(visual))
        layer = self.model.language_model.layers[0]
        hidden = self.model.language_model.embed_tokens(input_ids)
        hidden = hidden + visual.to(hidden.dtype)
        return self.lm_head(layer.down_proj(layer.q_proj(hidden)))


_TINY_COMPONENTS = ModelComponents(
    input_embedding="model.language_model.embed_tokens",
    output_head="lm_head",
    language="model.language_model",
    vision="model.visual",
    merger="model.visual.merger",
    vocab_size=16,
    hidden_size=4,
)


_TINY_SETUP = TokenSetup(tokens=("<a>", "<b>"), token_ids=(14, 15), tokenizer_size=16, model_vocab_size=16, added_tokens=0)


def _write_parent_bundle(bundle: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Train-shaped parent: language LoRA and atlas rows in PEFT, a full-visual file saved from the wrapped model."""

    parent = _TinyVLM(seed=2)
    parent.requires_grad_(False)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=language_linear_targets(parent, _TINY_COMPONENTS),
        trainable_token_indices={_TINY_COMPONENTS.input_embedding: [14, 15], _TINY_COMPONENTS.output_head: [14, 15]},
    )
    wrapped = get_peft_model(parent, peft_config)
    wrapped._catan_components = _TINY_COMPONENTS
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if ".lora_B." in name or "trainable_tokens_delta" in name or ".model.visual." in name:
                parameter.add_(torch.randn_like(parameter))
    bundle.mkdir(parents=True)
    wrapped.save_pretrained(str(bundle))
    save_visual_state(wrapped, _TINY_COMPONENTS, bundle)
    (bundle / "trainable_parameters.json").write_text(json.dumps({"semantic_tokens": _TINY_SETUP.as_dict()}))
    merged = wrapped.merge_and_unload()
    merged_state = {name: tensor.detach().clone() for name, tensor in merged.state_dict().items()}
    visual_state = {name: tensor for name, tensor in merged_state.items() if name.startswith("model.visual.")}
    return merged_state, visual_state
