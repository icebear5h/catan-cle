"""Shared helpers for qwen vision sft commands, fingerprints, and trainable scope audits."""

from collections.abc import Iterator, Sequence
from types import SimpleNamespace

from sft.qwen_series_vision_sft import (
    VisionSftConfig,
    build_vision_sft_command,
)


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _command(profile: str) -> list[str]:
    return build_vision_sft_command(
        VisionSftConfig(profile=profile),
        train_json="/data/train.json",
        image_folder="/data/images",
        output_dir="/runs/output",
    )


class _Module:
    def __init__(self, weight: object = None) -> None:
        self.weight = weight if weight is not None else object()


class _TokenModel:
    def __init__(self, *, tied: bool) -> None:
        input_weight = object()
        output_weight = input_weight if tied else object()
        self.embed = _Module(input_weight)
        self.output = _Module(output_weight)

    def named_modules(self) -> list[tuple[str, object]]:
        return [
            ("", self),
            ("model.language_model.embed_tokens", self.embed),
            ("lm_head", self.output),
        ]

    def get_input_embeddings(self) -> _Module:
        return self.embed

    def get_output_embeddings(self) -> _Module:
        return self.output


class _Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        return list(range(100, 100 + len(tokens)))


class _Parameter:
    def __init__(self, count: int, requires_grad: bool = True) -> None:
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.count


class _AuditModel:
    config = SimpleNamespace(tie_word_embeddings=False)

    def __init__(self, parameters: Sequence[tuple[str, _Parameter]]) -> None:
        self.parameters = parameters

    def named_parameters(self) -> Iterator[tuple[str, _Parameter]]:
        return iter(self.parameters)
