"""modules."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import torch
from transformers import PreTrainedTokenizerBase


class TokenEmbeddingModel(Protocol):
    """The slice of a Hugging Face model the token-row target lookup reads."""

    def named_modules(self) -> Iterator[tuple[str, torch.nn.Module]]: ...

    def get_input_embeddings(self) -> torch.nn.Module: ...

    def get_output_embeddings(self) -> torch.nn.Module | None: ...


def _find_language_embed_tokens_name(model: TokenEmbeddingModel) -> str:
    """Return the concrete language embedding module name."""

    candidates: list[tuple[str, torch.nn.Module]] = []
    name: str
    module: torch.nn.Module
    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and "visual" not in name:
            candidates.append((name, module))

    if not candidates:
        raise RuntimeError("Could not find language embed_tokens module")

    if len(candidates) == 1:
        return candidates[0][0]

    for name, _ in candidates:
        if "language_model" in name:
            return name

    return candidates[0][0]


def _find_module_name(
    model: TokenEmbeddingModel,
    target: torch.nn.Module,
    description: str,
) -> str:
    name: str
    module: torch.nn.Module
    for name, module in model.named_modules():
        if module is target:
            return name
    raise RuntimeError(f"Could not find {description} module name")


def _language_token_target_names(model: TokenEmbeddingModel) -> list[str]:
    """Return selective input and untied output token modules."""

    embed_name = _find_language_embed_tokens_name(model)
    names = [embed_name]
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Catan tokens require a trainable output embedding module")

    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    weights_are_tied = input_weight is not None and input_weight is output_weight
    if not weights_are_tied:
        names.append(_find_module_name(model, output_embeddings, "language output embedding"))
    return names


def _catan_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    catan_tokens: list[str],
) -> list[int]:
    token_ids = tokenizer.convert_tokens_to_ids(catan_tokens)
    if isinstance(token_ids, int):
        raise TypeError("convert_tokens_to_ids returned one id for a token list")
    if len(token_ids) != len(catan_tokens) or len(set(token_ids)) != len(token_ids):
        raise RuntimeError("Catan trainable tokens did not map to distinct tokenizer rows")
    if tokenizer.unk_token_id in token_ids:
        raise RuntimeError("A Catan trainable token mapped to the unknown-token row")
    return token_ids
