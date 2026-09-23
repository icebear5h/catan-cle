"""Module discovery and protected-subspace primitives."""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Iterator, Protocol, runtime_checkable

import torch
from peft import PeftModel
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import TrainableTokensWrapper

from sft.scripts.train.train_trl_catan_vision._config import ModelComponents


@runtime_checkable
class EmbeddingModel(Protocol):
    """A Hugging Face style model exposing its input and output embeddings."""

    def get_input_embeddings(self) -> torch.nn.Module: ...

    def get_output_embeddings(self) -> torch.nn.Module | None: ...


@runtime_checkable
class TokenAdapterLayer(Protocol):
    """The PEFT `TrainableTokensLayer` surface the chunked-NLL view reads."""

    @property
    def active_adapters(self) -> list[str]: ...

    @property
    def disable_adapters(self) -> bool: ...

    @property
    def merged(self) -> bool: ...

    def get_base_layer(self) -> torch.nn.Module: ...

    def get_merged_weights(self, active_adapters: list[str]) -> torch.Tensor: ...


def embedding_model(model: torch.nn.Module) -> EmbeddingModel:
    """Narrow `model` to one that exposes Hugging Face embedding accessors."""
    if not isinstance(model, EmbeddingModel):
        raise TypeError(f"{type(model).__name__} does not expose input/output embeddings")
    return model


def module_tensor(module: torch.nn.Module, name: str) -> torch.Tensor:
    """A tensor attribute (parameter, buffer or property) of `module`."""
    value = getattr(module, name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{type(module).__name__}.{name} is not a tensor")
    return value


def as_tensor(value: object) -> torch.Tensor:
    """A tensor batch field, narrowed from the trainer's loosely typed inputs."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a tensor batch field, got {type(value).__name__}")
    return value


def required_text(value: str | None, name: str) -> str:
    """A configuration path that the current mode requires."""
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def catan_components(model: torch.nn.Module) -> ModelComponents:
    """The component paths recorded on a model under `_catan_components`."""
    components = getattr(model, "_catan_components")
    if not isinstance(components, ModelComponents):
        raise TypeError(f"{type(model).__name__}._catan_components is not ModelComponents")
    return components


def submodule(module: torch.nn.Module, name: str) -> torch.nn.Module:
    """A child module attribute of `module`."""
    value = getattr(module, name)
    if not isinstance(value, torch.nn.Module):
        raise TypeError(f"{type(module).__name__}.{name} is not a module")
    return value


def _token_layer(head: object) -> TokenAdapterLayer:
    if not isinstance(head, TokenAdapterLayer):
        raise TypeError(f"{type(head).__name__} is not a PEFT trainable-tokens layer")
    return head


class _ChunkedNLLTrainableTokensHead:
    """Expose PEFT TrainableTokens as one differentiable output-head weight.

    TRL's chunked NLL intentionally reads ``lm_head.weight`` instead of calling
    the module so it can project only supervised positions. PEFT's
    ``TrainableTokensLayer.weight`` is the frozen base weight, while
    ``get_merged_weights`` is the exact functional weight with the selected
    trainable rows replaced. This view gives TRL that functional weight without
    merging the adapter in-place or making the full output head trainable.
    """

    def __init__(self, head: torch.nn.Module) -> None:
        self._head = head

    @property
    def weight(self) -> torch.Tensor:
        head = self._head
        if hasattr(head, "token_adapter"):
            # LoraConfig(trainable_token_indices=...) uses PEFT's auxiliary
            # TrainableTokensWrapper. Its weight property is already the
            # differentiable merged view from the inner token adapter.
            return module_tensor(head, "weight")
        layer = _token_layer(head)
        active_adapters = list(layer.active_adapters)
        if layer.disable_adapters or layer.merged or not active_adapters:
            return module_tensor(layer.get_base_layer(), "weight")
        return layer.get_merged_weights(active_adapters)

    @property
    def bias(self) -> torch.Tensor | None:
        head = self._head
        adapter = _token_layer(head.token_adapter if hasattr(head, "token_adapter") else head)
        bias = getattr(adapter.get_base_layer(), "bias", None)
        if bias is not None and not isinstance(bias, torch.Tensor):
            raise TypeError("output head bias is not a tensor")
        return bias


@contextmanager
def expose_trainable_tokens_head_to_chunked_nll(model: torch.nn.Module) -> Iterator[None]:
    """Let TRL capture a PEFT-aware head while installing chunked NLL.

    The override exists only during ``SFTTrainer.__init__``. TRL's patched
    forward retains the lightweight view in its closure; the model immediately
    regains its normal ``get_output_embeddings`` method for saving, generation,
    and reload validation.
    """

    if not isinstance(model, PeftModel):
        yield
        return
    target = embedding_model(model.get_base_model())
    head = target.get_output_embeddings()
    if not isinstance(head, (BaseTunerLayer, TrainableTokensWrapper)):
        yield
        return
    if not isinstance(head, (TrainableTokensLayer, TrainableTokensWrapper)):
        raise TypeError(
            "chunked NLL only supports a PEFT-wrapped output head when the wrapper is "
            f"TrainableTokensLayer; received {type(head).__name__}"
        )

    view = _ChunkedNLLTrainableTokensHead(head)
    sentinel = object()
    previous = target.__dict__.get("get_output_embeddings", sentinel)

    def get_output_embeddings(_self: torch.nn.Module) -> _ChunkedNLLTrainableTokensHead:
        return view

    setattr(target, "get_output_embeddings", types.MethodType(get_output_embeddings, target))
    try:
        yield
    finally:
        if previous is sentinel:
            delattr(target, "get_output_embeddings")
        else:
            setattr(target, "get_output_embeddings", previous)


def module_name_for_instance(model: torch.nn.Module, target: torch.nn.Module) -> str:
    names: list[str] = [
        name for name, module in model.named_modules() if module is target and name
    ]
    if len(names) != 1:
        raise ValueError(f"expected one module path for {type(target).__name__}; found {names}")
    return names[0]


def discover_components(model: torch.nn.Module) -> ModelComponents:
    embeddings = embedding_model(model)
    input_embedding = embeddings.get_input_embeddings()
    output_head = embeddings.get_output_embeddings()
    inner = submodule(model, "model")
    visual = submodule(inner, "visual")
    merger = submodule(visual, "merger")
    language = submodule(inner, "language_model")
    if not isinstance(input_embedding, torch.nn.Embedding):
        raise TypeError("input embedding must be torch.nn.Embedding")
    if not isinstance(output_head, torch.nn.Linear) or output_head.bias is not None:
        raise TypeError("untied output head must be a biasless torch.nn.Linear")
    if input_embedding.weight is output_head.weight:
        raise ValueError("Catan selective input/output rows require untied weights")
    vocab_size, hidden_size = input_embedding.weight.shape
    if tuple(output_head.weight.shape) != (vocab_size, hidden_size):
        raise ValueError("input embedding and output head shapes disagree")
    components = ModelComponents(
        input_embedding=module_name_for_instance(model, input_embedding),
        output_head=module_name_for_instance(model, output_head),
        language=module_name_for_instance(model, language),
        vision=module_name_for_instance(model, visual),
        merger=module_name_for_instance(model, merger),
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
    )
    if getattr(model.config, "model_type", None) == "qwen3_5":
        expected = {
            "input_embedding": "model.language_model.embed_tokens",
            "output_head": "lm_head",
            "language": "model.language_model",
            "vision": "model.visual",
            "merger": "model.visual.merger",
        }
        actual = {key: getattr(components, key) for key in expected}
        if actual != expected:
            raise ValueError(f"Qwen3.5 component paths changed: {actual}")
    return components


def _matches_path(name: str, path: str) -> bool:
    return name == path or name.endswith(f".{path}") or f".{path}." in f".{name}."


def resolve_wrapped_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    matches: list[torch.nn.Module] = [
        module
        for name, module in model.named_modules()
        if name == path or name.endswith(f".{path}")
    ]
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise ValueError(f"expected one wrapped module ending in {path!r}; found {len(unique)}")
    return unique[0]
