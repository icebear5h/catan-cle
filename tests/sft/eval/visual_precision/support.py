"""Shared helpers for visual precision preservation, generation, and cli forwarding."""

from collections.abc import Iterable, Sequence
from types import SimpleNamespace

import torch
from transformers import BatchFeature, PretrainedConfig


class TinyVLM(torch.nn.Module):
    def __init__(self) -> None:
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

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.lm_head

    def resize_token_embeddings(self, size: int, **kwargs: object) -> torch.nn.Embedding:
        assert size == 256  # Like Qwen, the base already has padded vocabulary rows.
        return self.get_input_embeddings()

    def prepare_inputs_for_generation(self, **kwargs: object) -> dict[str, object]:
        return kwargs

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.lm_head(self.model.language_model.proj(self.get_input_embeddings()(input_ids)))


class GenerationModel(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self, preserve: bool) -> None:
        super().__init__()
        self.visual = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.visual.weight.copy_(torch.tensor([[2.0001, 0], [1.0001, 0]]))
        if not preserve:
            self.visual.bfloat16()
        self.contexts: list[tuple[bool, bool]] = []

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        output_logits: bool,
        **kwargs: object,
    ) -> SimpleNamespace:
        self.contexts.append((torch.is_autocast_enabled("cpu"), torch.is_inference_mode_enabled()))
        logits = self.visual(pixel_values.to(self.visual.weight.dtype))
        return SimpleNamespace(
            sequences=torch.cat((input_ids, logits.argmax(-1, keepdim=True)), dim=1),
            logits=(logits,) if output_logits else None,
        )


class GenerationProcessor:
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [{"yes": 0, "no": 1}[text]])

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        assert kwargs["enable_thinking"] is False
        return "prompt"

    def __call__(self, text: Sequence[str], **kwargs: object) -> BatchFeature:
        return BatchFeature(
            {
                "input_ids": torch.tensor([[1, 2]] * len(text)),
                "pixel_values": torch.tensor([[1.0, 0.0]] * len(text)),
            }
        )

    def batch_decode(self, sequences: Iterable[torch.Tensor], **kwargs: object) -> list[str]:
        return ["yes" if sequence.tolist() == [0] else "no" for sequence in sequences]
