"""Semantic atlas-token inventories and their PEFT trainable-token setup."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # Heavy; ms-swift imports transformers lazily at call time.
    from transformers import PreTrainedTokenizerBase

import torch

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory
from sft.json_types import as_dict, as_list, as_str, loads_json

from ._base import PEFT_VERSION, JsonDict, ModelComponentPaths, SemanticTokenSetup
from ._components import _validate_token_modules, embedding_module, resolve_model_module


def load_semantic_token_inventory(path: str | Path) -> JsonDict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = as_dict(loads_json(resolved.read_text()))
    expected = semantic_recognition_token_inventory()
    if payload != expected:
        raise ValueError("token inventory must be the exact 154-token semantic inventory")
    if payload["counts"] != {
        "node": 54,
        "edge": 72,
        "tile": 19,
        "port": 9,
        "atlas": 154,
        "total": 154,
    }:
        raise ValueError("semantic token inventory counts changed")
    return payload


def prepare_peft_trainable_token_targets(
    model: torch.nn.Module,
    components: ModelComponentPaths,
    token_ids: Iterable[int],
    *,
    peft_version: str,
) -> dict[str, list[int]]:
    """Validate both token targets and apply the guarded PEFT 0.17.1 Linear shim."""

    if peft_version != PEFT_VERSION:
        raise RuntimeError(f"Catan SFT requires peft=={PEFT_VERSION}; found {peft_version}")
    ids = tuple(int(token_id) for token_id in token_ids)
    if len(ids) != 154 or len(set(ids)) != 154:
        raise ValueError("trainable token targets require exactly 154 unique token IDs")
    if min(ids) < 0 or max(ids) >= components.vocab_size:
        raise ValueError("trainable token IDs fall outside the resized model vocabulary")

    input_embedding = resolve_model_module(model, components.input_embedding)
    output_head = resolve_model_module(model, components.output_head)
    if not isinstance(input_embedding, torch.nn.Embedding):
        raise ValueError("PEFT input token target is not torch.nn.Embedding")
    if not isinstance(output_head, torch.nn.Linear):
        raise ValueError("PEFT output token target is not torch.nn.Linear")
    if output_head.bias is not None:
        raise ValueError("PEFT output token target must be biasless")
    if tuple(input_embedding.weight.shape) != (
        components.vocab_size,
        components.hidden_size,
    ) or tuple(output_head.weight.shape) != (
        components.vocab_size,
        components.hidden_size,
    ):
        raise ValueError("token target shapes changed after component discovery")
    if output_head.in_features != components.hidden_size:
        raise ValueError("output head in_features changed after component discovery")

    existing_dimension = getattr(output_head, "embedding_dim", None)
    if existing_dimension is not None and existing_dimension != output_head.in_features:
        raise ValueError("output head has an incompatible embedding_dim attribute")
    # PEFT 0.17.1's TrainableTokensLayer reads embedding_dim for both Embedding
    # and Linear targets. Linear does not define it, so add the equivalent only
    # after the strict type, bias, vocabulary, and hidden-shape checks above.
    output_head.embedding_dim = output_head.in_features
    ids_list = list(ids)
    return {
        components.input_embedding: ids_list,
        components.output_head: list(ids_list),
    }


def prepare_semantic_tokens(
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    inventory: JsonDict,
    *,
    pad_to_multiple_of: int = 128,
) -> SemanticTokenSetup:
    """Add regular atlas tokens, resize untied embeddings, and assert atomic IDs."""

    if inventory != semantic_recognition_token_inventory():
        raise ValueError("semantic inventory content changed")
    _validate_token_modules(model)

    tokens = [as_str(token) for token in as_list(inventory["tokens"])]
    vocabulary = tokenizer.get_vocab()
    present = [token for token in tokens if token in vocabulary]
    if present and len(present) != len(tokens):
        raise ValueError("tokenizer contains only part of the semantic atlas inventory")
    original_size = len(tokenizer)
    added = tokenizer.add_tokens(tokens, special_tokens=False)
    expected_added = 0 if present else len(tokens)
    if added != expected_added:
        raise ValueError(f"tokenizer added {added} semantic tokens; expected {expected_added}")

    # transformers' `resize_token_embeddings` is not declared on `torch.nn.Module`.
    resize = getattr(model, "resize_token_embeddings")
    resize(
        len(tokenizer),
        pad_to_multiple_of=pad_to_multiple_of,
        mean_resizing=False,
    )
    model_vocab_size, _ = _validate_token_modules(model)

    raw_ids = tokenizer.convert_tokens_to_ids(list(tokens))
    if isinstance(raw_ids, int):
        raise TypeError("convert_tokens_to_ids returned one id for a token list")
    token_ids = tuple(raw_ids)
    if len(token_ids) != 154 or len(set(token_ids)) != 154:
        raise ValueError("semantic token IDs are not 154 unique rows")
    if tuple(sorted(token_ids)) != tuple(range(min(token_ids), min(token_ids) + len(token_ids))):
        raise ValueError("semantic token IDs are not one contiguous added block")
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    if special_ids.intersection(token_ids):
        raise ValueError("semantic atlas tokens were added as special tokens")
    for token, token_id in zip(tokens, token_ids, strict=True):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(f"semantic token is not atomic: {token} -> {encoded}")

    # Freeze only the base output weight. Calling requires_grad_ on the module
    # after PEFT wrapping would recursively freeze the selective output adapter.
    output_head = embedding_module(model, "get_output_embeddings")
    if output_head is None:
        raise ValueError("model must expose an output head")
    output_head.weight.requires_grad_(False)
    if model_vocab_size < len(tokenizer) or model_vocab_size % pad_to_multiple_of:
        raise ValueError("resized model vocabulary is missing padded semantic rows")
    setup = SemanticTokenSetup(
        token_ids=token_ids,
        original_tokenizer_size=original_size,
        tokenizer_size=len(tokenizer),
        model_vocab_size=model_vocab_size,
        added_tokens=added,
    )
    setattr(model, "_catan_semantic_token_setup", setup)
    return setup
