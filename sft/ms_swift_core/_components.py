"""Discover and pin the vision, aligner and embedding module paths."""

from __future__ import annotations

from typing import Iterable, Iterator

import torch

from ._base import _COMPONENT_ATTRIBUTE, ModelComponentPaths


def _named_modules_with_aliases(model: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Module]]:
    # torch leaves the `named_modules` generator unannotated.
    modules: Iterator[tuple[str, torch.nn.Module]]
    try:
        modules = model.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - retained for older torch compatibility
        modules = model.named_modules()
    return modules


def embedding_module(model: torch.nn.Module, accessor: str) -> torch.nn.Module | None:
    """Call a transformers embedding accessor, which `torch.nn.Module` does not declare."""
    module = getattr(model, accessor)()
    if module is None:
        return None
    if isinstance(module, torch.nn.Module):
        return module
    raise ValueError(f"{accessor}() did not return a torch.nn.Module")


def _required_embedding(model: torch.nn.Module, accessor: str) -> torch.nn.Module:
    module = embedding_module(model, accessor)
    if module is None:
        raise ValueError(f"{accessor}() returned None")
    return module


def module_name_for_instance(model: torch.nn.Module, target: torch.nn.Module) -> str:
    matches = [
        name
        for name, module in _named_modules_with_aliases(model)
        if name and module is target
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one module name for {type(target).__name__}; found {matches}")
    return matches[0]


def resolve_model_module(model: torch.nn.Module, canonical_path: str) -> torch.nn.Module:
    """Resolve a pre-PEFT path on either a base model or a wrapped PEFT model."""

    matches: list[tuple[str, torch.nn.Module]] = []
    suffix = f".{canonical_path}"
    for name, module in _named_modules_with_aliases(model):
        if name == canonical_path or name.endswith(suffix):
            matches.append((name, module))
    unique = {id(module): module for _, module in matches}
    if len(unique) != 1:
        names = [name for name, _ in matches]
        raise ValueError(f"expected one module for canonical path {canonical_path!r}; found {names}")
    return next(iter(unique.values()))


def _embedding_weights_are_tied(model: torch.nn.Module) -> bool:
    input_embeddings = embedding_module(model, "get_input_embeddings")
    output_embeddings = embedding_module(model, "get_output_embeddings")
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("model must expose distinct input and output embeddings")
    if input_embeddings.weight is output_embeddings.weight:
        return True
    input_pointer = getattr(input_embeddings.weight, "data_ptr", None)
    output_pointer = getattr(output_embeddings.weight, "data_ptr", None)
    if callable(input_pointer) and callable(output_pointer):
        return bool(input_pointer() == output_pointer())
    return False


def _config_uses_tied_embeddings(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model must expose a config")
    if bool(getattr(config, "tie_word_embeddings", False)):
        return True
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return bool(getattr(get_text_config(), "tie_word_embeddings", False))
    return bool(getattr(getattr(config, "text_config", None), "tie_word_embeddings", False))


def _validate_token_modules(model: torch.nn.Module) -> tuple[int, int]:
    input_embedding = embedding_module(model, "get_input_embeddings")
    output_head = embedding_module(model, "get_output_embeddings")
    if not isinstance(input_embedding, torch.nn.Embedding):
        raise ValueError("input embeddings must be an untied torch.nn.Embedding")
    if not isinstance(output_head, torch.nn.Linear):
        raise ValueError("output head must be an untied torch.nn.Linear")
    if output_head.bias is not None:
        raise ValueError("selective output-row tuning requires a biasless output head")
    if _config_uses_tied_embeddings(model) or _embedding_weights_are_tied(model):
        raise ValueError("selective input/output-row tuning requires untied weights")

    input_shape = tuple(input_embedding.weight.shape)
    output_shape = tuple(output_head.weight.shape)
    if input_shape != output_shape:
        raise ValueError(
            "input embedding and output head must have identical [vocab, hidden] shapes; "
            f"found {input_shape} and {output_shape}"
        )
    vocab_size, hidden_size = input_shape
    if input_embedding.num_embeddings != vocab_size or input_embedding.embedding_dim != hidden_size:
        raise ValueError("input embedding metadata does not match its weight")
    if output_head.out_features != vocab_size or output_head.in_features != hidden_size:
        raise ValueError("output head metadata does not match its weight")
    return int(vocab_size), int(hidden_size)


def _normalize_arch_paths(model_arch: object, field: str) -> tuple[str, ...]:
    raw = getattr(model_arch, field, None)
    raw_paths: tuple[object, ...]
    if isinstance(raw, str):
        raw_paths = (raw,)
    elif isinstance(raw, (list, tuple)):
        raw_paths = tuple(raw)
    elif raw is None:
        raw_paths = ()
    else:
        raise ValueError(f"ms-swift model architecture field {field!r} has invalid type")
    paths = tuple(path for path in raw_paths if isinstance(path, str) and path)
    if not paths or len(paths) != len(raw_paths):
        raise ValueError(f"ms-swift model architecture field {field!r} must contain paths")
    if len(set(paths)) != len(paths):
        raise ValueError(f"ms-swift model architecture field {field!r} contains duplicate paths")
    return paths


def _path_is_within(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(f"{root}.") for root in roots)


def discover_model_components(model: torch.nn.Module) -> ModelComponentPaths:
    """Discover and validate token and multimodal modules without model-name checks."""

    model_meta = getattr(model, "model_meta", None)
    model_arch = getattr(model_meta, "model_arch", None)
    if model_arch is None:
        raise ValueError("model is missing ms-swift model_meta.model_arch")

    vocab_size, hidden_size = _validate_token_modules(model)
    input_path = module_name_for_instance(model, _required_embedding(model, "get_input_embeddings"))
    output_path = module_name_for_instance(
        model, _required_embedding(model, "get_output_embeddings")
    )
    if input_path == output_path:
        raise ValueError("input embedding and output head must be distinct modules")

    language_paths = _normalize_arch_paths(model_arch, "language_model")
    vision_paths = _normalize_arch_paths(model_arch, "vision_tower")
    aligner_paths = _normalize_arch_paths(model_arch, "aligner")
    for field, paths in (
        ("language_model", language_paths),
        ("vision_tower", vision_paths),
        ("aligner", aligner_paths),
    ):
        for path in paths:
            try:
                module = model.get_submodule(path)
            except AttributeError as exc:
                raise ValueError(
                    f"ms-swift model architecture path is missing: {field}={path}"
                ) from exc
            if not isinstance(module, torch.nn.Module):
                raise ValueError(f"ms-swift architecture path is not a module: {field}={path}")

    if not _path_is_within(input_path, language_paths):
        raise ValueError("input embedding is outside the declared ms-swift language modules")
    if not _path_is_within(output_path, language_paths):
        raise ValueError("output head is outside the declared ms-swift language modules")
    if _path_is_within(input_path, (*vision_paths, *aligner_paths)) or _path_is_within(
        output_path, (*vision_paths, *aligner_paths)
    ):
        raise ValueError("token modules overlap declared visual modules")

    architecture = str(
        getattr(model_arch, "arch_name", None)
        or getattr(model.config, "model_type", None)
        or type(model).__name__
    )
    components = ModelComponentPaths(
        architecture=architecture,
        input_embedding=input_path,
        output_head=output_path,
        language=language_paths,
        vision=vision_paths,
        aligner=aligner_paths,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )
    setattr(model, _COMPONENT_ATTRIBUTE, components)
    return components


def get_model_components(model: torch.nn.Module) -> ModelComponentPaths:
    components = getattr(model, _COMPONENT_ATTRIBUTE, None)
    if not isinstance(components, ModelComponentPaths):
        raise RuntimeError("model is missing the validated ms-swift component contract")
    return components


def attach_model_components(
    model: torch.nn.Module,
    components: ModelComponentPaths,
) -> None:
    setattr(model, _COMPONENT_ATTRIBUTE, components)
