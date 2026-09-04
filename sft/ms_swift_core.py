"""Pure Catan contracts used by the pinned ms-swift training integration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory


JsonDict = dict[str, Any]
MS_SWIFT_VERSION = "4.5.2"
PEFT_VERSION = "0.17.1"
CATAN_TUNER_TYPE = "catan_vision_tokens"
COMPONENT_SCHEMA = "catan_ms_swift_model_components/v1"
SCOPE_SCHEMA = "catan_ms_swift_trainable_scope/v2"
OPTIMIZER_COVERAGE_SCHEMA = "catan_ms_swift_optimizer_coverage/v1"
_COMPONENT_ATTRIBUTE = "_catan_ms_swift_model_components"


@dataclass(frozen=True)
class SemanticTokenSetup:
    token_ids: tuple[int, ...]
    original_tokenizer_size: int
    tokenizer_size: int
    model_vocab_size: int
    added_tokens: int

    def as_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class ModelComponentPaths:
    """Validated, model-independent paths for one ms-swift multimodal model."""

    architecture: str
    input_embedding: str
    output_head: str
    language: tuple[str, ...]
    vision: tuple[str, ...]
    aligner: tuple[str, ...]
    vocab_size: int
    hidden_size: int

    def as_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["schema"] = COMPONENT_SCHEMA
        return payload


def load_semantic_token_inventory(path: str | Path) -> JsonDict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text())
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


def _named_modules_with_aliases(model: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Module]]:
    try:
        return model.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - retained for older torch compatibility
        return model.named_modules()


def module_name_for_instance(model: Any, target: Any) -> str:
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


def _embedding_weights_are_tied(model: Any) -> bool:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("model must expose distinct input and output embeddings")
    if input_embeddings.weight is output_embeddings.weight:
        return True
    input_pointer = getattr(input_embeddings.weight, "data_ptr", None)
    output_pointer = getattr(output_embeddings.weight, "data_ptr", None)
    if callable(input_pointer) and callable(output_pointer):
        return input_pointer() == output_pointer()
    return False


def _config_uses_tied_embeddings(model: Any) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model must expose a config")
    if bool(getattr(config, "tie_word_embeddings", False)):
        return True
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return bool(getattr(get_text_config(), "tie_word_embeddings", False))
    return bool(getattr(getattr(config, "text_config", None), "tie_word_embeddings", False))


def _validate_token_modules(model: Any) -> tuple[int, int]:
    input_embedding = model.get_input_embeddings()
    output_head = model.get_output_embeddings()
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


def _normalize_arch_paths(model_arch: Any, field: str) -> tuple[str, ...]:
    raw = getattr(model_arch, field, None)
    if isinstance(raw, str):
        paths = (raw,)
    elif isinstance(raw, (list, tuple)):
        paths = tuple(raw)
    elif raw is None:
        paths = ()
    else:
        raise ValueError(f"ms-swift model architecture field {field!r} has invalid type")
    if not paths or any(not isinstance(path, str) or not path for path in paths):
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
    input_path = module_name_for_instance(model, model.get_input_embeddings())
    output_path = module_name_for_instance(model, model.get_output_embeddings())
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
    tokenizer: Any,
    model: Any,
    inventory: JsonDict,
    *,
    pad_to_multiple_of: int = 128,
) -> SemanticTokenSetup:
    """Add regular atlas tokens, resize untied embeddings, and assert atomic IDs."""

    if inventory != semantic_recognition_token_inventory():
        raise ValueError("semantic inventory content changed")
    _validate_token_modules(model)

    tokens = list(inventory["tokens"])
    vocabulary = tokenizer.get_vocab()
    present = [token for token in tokens if token in vocabulary]
    if present and len(present) != len(tokens):
        raise ValueError("tokenizer contains only part of the semantic atlas inventory")
    original_size = len(tokenizer)
    added = tokenizer.add_tokens(tokens, special_tokens=False)
    expected_added = 0 if present else len(tokens)
    if added != expected_added:
        raise ValueError(f"tokenizer added {added} semantic tokens; expected {expected_added}")

    resize = model.resize_token_embeddings
    resize(
        len(tokenizer),
        pad_to_multiple_of=pad_to_multiple_of,
        mean_resizing=False,
    )
    model_vocab_size, _ = _validate_token_modules(model)

    token_ids = tuple(tokenizer.convert_tokens_to_ids(tokens))
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
    model.get_output_embeddings().weight.requires_grad_(False)
    if model_vocab_size < len(tokenizer) or model_vocab_size % pad_to_multiple_of:
        raise ValueError("resized model vocabulary is missing padded semantic rows")
    setup = SemanticTokenSetup(
        token_ids=token_ids,
        original_tokenizer_size=original_size,
        tokenizer_size=len(tokenizer),
        model_vocab_size=model_vocab_size,
        added_tokens=added,
    )
    model._catan_semantic_token_setup = setup
    return setup


def _module_path_matches(parameter_name: str, module_path: str) -> bool:
    normalized = f".{parameter_name}."
    target = f".{module_path}."
    return target in normalized


def _matches_any(parameter_name: str, module_paths: Iterable[str]) -> bool:
    return any(_module_path_matches(parameter_name, path) for path in module_paths)


def parameter_matches_paths(parameter_name: str, module_paths: Iterable[str]) -> bool:
    """Return whether a possibly PEFT-prefixed parameter belongs to a canonical path."""

    return _matches_any(parameter_name, module_paths)


def _parameter_category(
    name: str,
    *,
    components: ModelComponentPaths,
) -> str:
    if "trainable_tokens_delta" in name:
        is_input = _module_path_matches(name, components.input_embedding)
        is_output = _module_path_matches(name, components.output_head)
        if is_input and not is_output:
            return "atlas_input_rows"
        if is_output and not is_input:
            return "atlas_output_rows"
        return "unknown"
    if _module_path_matches(name, components.input_embedding):
        return "input_embedding"
    if _module_path_matches(name, components.output_head):
        return "output_head"
    if "lora_A" in name or "lora_B" in name:
        return "language_lora" if _matches_any(name, components.language) else "nonlanguage_lora"
    if _matches_any(name, components.aligner):
        return "aligner"
    if _matches_any(name, components.vision):
        return "vision"
    if _matches_any(name, components.language):
        return "base_language"
    return "unknown"


def audit_trainable_scope(
    model: torch.nn.Module,
    *,
    language_lora: bool,
    output_path: str | Path | None = None,
) -> JsonDict:
    """Fail closed unless only visual modules, LoRA, and two atlas deltas train."""

    components = get_model_components(model)
    setup = getattr(model, "_catan_semantic_token_setup", None)
    if setup is None or len(setup.token_ids) != 154:
        raise RuntimeError("model is missing the exact semantic token setup")

    groups: dict[str, JsonDict] = defaultdict_group_manifest()
    parameter_rows = []
    errors = []
    expected_delta_shape = (154, components.hidden_size)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = _parameter_category(name, components=components)
        count = parameter.numel()
        groups[category]["tensors"] += 1
        groups[category]["parameters"] += count
        groups[category]["names"].append(name)
        parameter_rows.append(
            {
                "name": name,
                "category": category,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "parameters": count,
            }
        )
        if category in ("atlas_input_rows", "atlas_output_rows") and tuple(
            parameter.shape
        ) != expected_delta_shape:
            errors.append(
                f"{category} delta must have shape {expected_delta_shape}: "
                f"{name} {tuple(parameter.shape)}"
            )

    required = ("vision", "aligner", "atlas_input_rows", "atlas_output_rows")
    for category in required:
        if groups[category]["parameters"] == 0:
            errors.append(f"required trainable group is empty: {category}")
    for category in ("atlas_input_rows", "atlas_output_rows"):
        if groups[category]["tensors"] != 1:
            errors.append(f"exactly one {category} delta tensor is required")
    if language_lora and groups["language_lora"]["parameters"] == 0:
        errors.append("language LoRA profile has no trainable LoRA parameters")
    if not language_lora and groups["language_lora"]["parameters"]:
        errors.append("vision-only profile unexpectedly trains language LoRA")
    for category in (
        "input_embedding",
        "output_head",
        "base_language",
        "nonlanguage_lora",
        "unknown",
    ):
        if groups[category]["parameters"]:
            errors.append(f"forbidden trainable group is non-empty: {category}")

    manifest = {
        "schema": SCOPE_SCHEMA,
        "ms_swift_version": MS_SWIFT_VERSION,
        "peft_version": PEFT_VERSION,
        "language_lora": language_lora,
        "model_components": components.as_dict(),
        "semantic_token_setup": setup.as_dict(),
        "groups": groups,
        "parameters": parameter_rows,
        "trainable_tensors": len(parameter_rows),
        "trainable_parameters": sum(row["parameters"] for row in parameter_rows),
        "errors": errors,
    }
    if output_path is not None:
        write_json_atomic(Path(output_path), manifest)
    if errors:
        raise RuntimeError("invalid ms-swift trainable scope: " + "; ".join(errors))
    return manifest


def defaultdict_group_manifest() -> dict[str, JsonDict]:
    categories = (
        "vision",
        "aligner",
        "atlas_input_rows",
        "atlas_output_rows",
        "language_lora",
        "input_embedding",
        "output_head",
        "base_language",
        "nonlanguage_lora",
        "unknown",
    )
    return {
        category: {"tensors": 0, "parameters": 0, "names": []}
        for category in categories
    }


def audit_optimizer_coverage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    output_path: str | Path | None = None,
) -> JsonDict:
    trainable = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    occurrences: dict[int, int] = {}
    group_rows = []
    for index, group in enumerate(optimizer.param_groups):
        names = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            occurrences[parameter_id] = occurrences.get(parameter_id, 0) + 1
            names.append(trainable.get(parameter_id, f"<unknown:{parameter_id}>"))
        group_rows.append(
            {
                "index": index,
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_tensors": len(group["params"]),
                "names": names,
            }
        )
    missing = sorted(name for parameter_id, name in trainable.items() if parameter_id not in occurrences)
    duplicate = sorted(
        trainable.get(parameter_id, f"<unknown:{parameter_id}>")
        for parameter_id, count in occurrences.items()
        if count != 1
    )
    unknown = sorted(
        name
        for group in group_rows
        for name in group["names"]
        if name.startswith("<unknown:")
    )
    errors = []
    if missing:
        errors.append(f"optimizer omitted trainable parameters: {missing[:8]}")
    if duplicate:
        errors.append(f"optimizer duplicated trainable parameters: {duplicate[:8]}")
    if unknown:
        errors.append(f"optimizer contains frozen or unknown parameters: {unknown[:8]}")
    report = {
        "schema": OPTIMIZER_COVERAGE_SCHEMA,
        "trainable_tensors": len(trainable),
        "covered_tensors": len(occurrences),
        "groups": group_rows,
        "missing": missing,
        "duplicate": duplicate,
        "unknown": unknown,
        "errors": errors,
    }
    if output_path is not None:
        write_json_atomic(Path(output_path), report)
    if errors:
        raise RuntimeError("invalid optimizer coverage: " + "; ".join(errors))
    return report


def write_json_atomic(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
