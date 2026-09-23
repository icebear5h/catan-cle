"""Shared helpers for text-mode trl training contracts over the local catan stack."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import (
    CLIPImageProcessor,
    GenerationMixin,
    LlavaProcessor,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory
from sft.scripts.train import train_trl_catan_vision as training

TextRow = dict[str, object]


def text_row(prompt: object = "board <N00> above <N01>?", answer: object = "yes") -> TextRow:
    return {"messages": [
        {"role": "user", "content": prompt}, {"role": "assistant", "content": answer},
    ]}


def text_tokenizer() -> PreTrainedTokenizerFast:
    words = ["[UNK]", "<|im_start|>", "<|im_end|>", "user", "assistant", "\n", " ",
             "node", "edge", "tile", "port", "yes", "no", "board", "wood", "empty", "above", "?"]
    raw = Tokenizer(WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]"))
    raw.pre_tokenizer = Split(Regex(r"\s"), behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", eos_token="<|im_end|>",
        pad_token="<|im_end|>", additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = (
        "{% if enable_thinking != false or preserve_thinking != false %}"
        "{{ raise_exception('thinking must be explicitly disabled') }}{% endif %}"
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content']"
        " + '<|im_end|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer.add_tokens(semantic_recognition_token_inventory()["tokens"], special_tokens=False)
    return tokenizer


class DormantVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.merger = torch.nn.Linear(8, 8)
        self.register_buffer("scale", torch.tensor(1.0001))
        self.register_buffer("indices", torch.tensor([0, 257], dtype=torch.int64))

    def forward(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("a text forward must never execute the visual tower")


class TinyLanguage(torch.nn.Module):
    def __init__(self, num_layers: int = 1) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(256, 8)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            layer = torch.nn.Module()
            layer.q_proj = torch.nn.Linear(8, 8, bias=False)
            layer.down_proj = torch.nn.Linear(8, 8, bias=False)
            self.layers.append(layer)

    def forward(self, input_ids: torch.Tensor) -> BaseModelOutput:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = torch.sigmoid(layer.down_proj(torch.tanh(layer.q_proj(hidden))))
        return BaseModelOutput(last_hidden_state=hidden)


class TinyBackbone(torch.nn.Module):
    """Qwen's ``model`` wrapper: TRL 1.12 chunked NLL runs ``base_model`` without the head."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.language_model = TinyLanguage(num_layers)
        self.visual = DormantVisual()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object) -> BaseModelOutput:
        assert not training.TEXT_MEDIA_KEYS.intersection(kwargs)
        assert "_prediction_loss_only" not in kwargs
        return self.language_model(input_ids)


class TinyTextVLM(PreTrainedModel, GenerationMixin):
    """Real numerical forward/generation with Qwen's module paths and untied heads."""

    config_class = PretrainedConfig
    base_model_prefix = "model"

    def __init__(self, num_layers: int = 1) -> None:
        super().__init__(PretrainedConfig(
            vocab_size=256, hidden_size=8, num_hidden_layers=num_layers, tie_word_embeddings=False,
            eos_token_id=2, pad_token_id=2, use_cache=False,
            max_position_embeddings=16384,
        ))
        self.model = TinyBackbone(num_layers)
        self.lm_head = torch.nn.Linear(8, 256, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                values = torch.arange(parameter.numel()).reshape(parameter.shape)
                parameter.copy_(torch.sin(values.float() + 1) * 0.03)
            self.lm_head.weight[11].fill_(1.0)  # The fixture's deterministic greedy answer is yes.

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.lm_head

    def resize_token_embeddings(self, new_num_tokens: int, **kwargs: object) -> torch.nn.Embedding:
        assert new_num_tokens == 256
        return self.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> CausalLMOutputWithPast:
        hidden = self.model(input_ids, attention_mask, **kwargs).last_hidden_state
        logits = self.lm_head(hidden)
        loss = None if labels is None else F.cross_entropy(
            logits[:, :-1].float().reshape(-1, 256), labels[:, 1:].reshape(-1),
        )
        return CausalLMOutputWithPast(logits=logits, loss=loss)


def write_text_source(
    bundle: Path,
    *,
    source_mode: str | None = None,
    num_layers: int = 1,
    legacy_processor_serialization: bool = False,
) -> tuple[object, PreTrainedTokenizerFast, object, object, object]:
    """Save a real rank-8/154-row parent, including nonempty optimizer state."""
    tokenizer = text_tokenizer()
    inventory = semantic_recognition_token_inventory()
    base = TinyTextVLM(num_layers).bfloat16()
    setup, components = training.prepare_semantic_tokens(tokenizer, base, inventory)
    config = training.TrainConfig(
        train_jsonl="unused.jsonl", image_root="unused-images", token_inventory="tokens.json",
        output_dir=str(bundle), model_id="offline-tiny", token_init="keep",
    )
    model = training.wrap_trainable_model(base, setup, components, config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "trainable_tokens_delta" in name:
                values = torch.arange(parameter.numel()).reshape(parameter.shape)
                parameter.copy_(torch.cos(values.float() + 1) * 0.1)
            elif ".lora_" in name:
                parameter.fill_(0.0125)
        for parameter in training.resolve_wrapped_module(model, components.vision).parameters():
            parameter.fill_(1.0001)
        training.resolve_wrapped_module(model, components.vision).scale.fill_(1.0001)
    model.eval()
    optimizer = training.build_optimizer(model, components, config)
    batch = training.TextCompletionCollator(tokenizer, max_sequence_length=128)([text_row()])
    model(**batch).loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    bundle.mkdir(parents=True)
    model.save_pretrained(bundle, save_embedding_layers=False)
    # A real HF image/text processor exercises serialization without torchvision.
    # This fixture is not a claim that Qwen's pinned processor has been executed.
    processor = LlavaProcessor(
        tokenizer=tokenizer, image_processor=CLIPImageProcessor(),
        chat_template=tokenizer.chat_template,
    )
    processor.save_pretrained(bundle)
    if legacy_processor_serialization:
        # transformers 5 always nests the image processor; rebuild the 4.x split layout.
        processor.image_processor.save_pretrained(bundle)
        nested = json.loads((bundle / "processor_config.json").read_text())
        nested.pop("image_processor")
        training.write_json_atomic(bundle / "processor_config.json", nested)
    # Explicit video configuration fixture: preserve bytes without importing the
    # torchvision-dependent video processor or manufacturing video data.
    training.write_json_atomic(bundle / "video_preprocessor_config.json", {
        "video_processor_type": "Qwen2VLVideoProcessor", "patch_size": 14,
        "temporal_patch_size": 2, "merge_size": 2, "min_pixels": 128 * 28 * 28,
        "max_pixels": 768 * 28 * 28,
    })
    training.save_visual_state(model, components, bundle)
    torch.save(optimizer.state_dict(), bundle / "optimizer.pt")
    assert optimizer.state
    saved_config = asdict(config)
    if source_mode is None:
        saved_config.pop("input_mode")  # Historical checkpoints predate this field.
    else:
        saved_config["input_mode"] = source_mode
    training.write_json_atomic(bundle / training.RUN_CONFIG_FILE, saved_config)
    training.write_json_atomic(bundle / training.TRAINABLE_SCOPE_FILE, {
        "semantic_tokens": setup.as_dict(),
    })
    return model, tokenizer, setup, components, replace(
        config, input_mode="text", image_root=None, initial_bundle=str(bundle),
        max_sequence_length=8192,
    )


def load_text_source(
    bundle: Path, tokenizer: PreTrainedTokenizerFast, config: object, *, num_layers: int = 1
) -> tuple[object, object]:
    base = TinyTextVLM(num_layers).bfloat16()
    setup, components = training.prepare_semantic_tokens(
        tokenizer, base, semantic_recognition_token_inventory(),
    )
    return training.load_initial_bundle(base, setup, components, config)
