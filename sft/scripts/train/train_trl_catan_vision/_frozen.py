"""Frozen bundles, adapters, and visual-state persistence."""

from __future__ import annotations

import importlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from peft import LoraConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils.other import TrainableTokensWrapper
from safetensors.torch import load_file, save_file

from sft.lora_expansion import expected_adapter_shapes, standard_lora_rank, tensor_headers
from sft.scripts.train.train_trl_catan_vision._common import (
    FROZEN_ADAPTER_DIR,
    FROZEN_BUNDLE_FILE,
    PROCESSOR_ASSET_FILES,
    RUN_CONFIG_FILE,
    TRAINABLE_SCOPE_FILE,
    VISUAL_STATE_FILE,
    JsonDict,
    sha256_file,
)
from sft.scripts.train.train_trl_catan_vision._config import (
    ModelComponents,
    TokenSetup,
    TrainConfig,
    validate_text_budget,
)
from sft.scripts.train.train_trl_catan_vision._model_tokens import language_linear_targets
from sft.scripts.train.train_trl_catan_vision._structure import (
    _matches_path,
    catan_components,
    resolve_wrapped_module,
)

if TYPE_CHECKING:  # Heavy; transformers loads lazily on the runtime path.
    from transformers import PreTrainedTokenizerBase
def frozen_adapter_files(bundle: Path) -> tuple[Path, ...]:
    return (bundle / "adapter_config.json", bundle / "adapter_model.safetensors")


def apply_frozen_adapter(
    base_model: torch.nn.Module,
    frozen_dir: Path,
    *,
    restore_visual: bool = False,
) -> tuple[torch.nn.Module, JsonDict]:
    """Merge a finished adapter (LoRA and atlas rows) into the base weights and drop its wrappers."""

    missing = [str(path) for path in frozen_adapter_files(frozen_dir) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen adapter is incomplete: {missing}")
    frozen = PeftModel.from_pretrained(base_model, frozen_dir, is_trainable=False)
    report: JsonDict = {"path": str(frozen_dir), "adapter_sha256": sha256_file(frozen_dir / "adapter_model.safetensors")}
    if restore_visual:
        # Parent bundles save the PEFT-wrapped state_dict names. Restore while
        # that same wrapper is present, before merging it or adding vision LoRA
        # (which would introduce base_layer names). Promote before copying so
        # the parent's FP32 visual weights never round through BF16.
        components = catan_components(base_model)
        setattr(frozen, "_catan_components", components)
        resolve_wrapped_module(frozen, components.vision).float()
        report["visual_state"] = load_visual_state(frozen, frozen_dir)
    merged: torch.nn.Module = frozen.merge_and_unload()
    leftover = [name for name, _ in merged.named_parameters() if ".lora_" in name or "trainable_tokens" in name]
    if leftover:
        raise RuntimeError(f"frozen adapter did not merge cleanly: {leftover[:4]}")
    return merged, report


def freeze_visual_except_lora(model: torch.nn.Module, components: ModelComponents) -> JsonDict:
    """Hold the vision tower at its loaded weights in fp32; only its LoRA tensors train."""

    visual = resolve_wrapped_module(model, components.vision)
    visual.float()
    trainable = frozen = 0
    for name, parameter in visual.named_parameters():
        is_lora = ".lora_" in f".{name}"
        parameter.requires_grad_(is_lora)
        trainable += is_lora
        frozen += not is_lora
    return {"module": components.vision, "trainable_tensors": trainable, "frozen_tensors": frozen}


def freeze_visual_weights(model: torch.nn.Module, components: ModelComponents) -> JsonDict:
    """Preserve dormant FP32 visual state, including floating buffers, in text mode."""
    visual = resolve_wrapped_module(model, components.vision)
    visual.float().requires_grad_(False)
    return {
        "module": components.vision, "trainable_tensors": 0,
        "frozen_tensors": sum(1 for _ in visual.parameters()),
        "dtypes": dict(Counter(str(t.dtype) for t in visual.state_dict().values())),
    }


def validate_checkpoint_tokenizer(
    tokenizer: PreTrainedTokenizerBase, bundle: str | Path, tokens: Sequence[str],
) -> None:
    """Compare the actual saved tokenizer mapping, not just two sidecar ID lists."""
    bundle = Path(bundle)
    if not (bundle / "tokenizer_config.json").is_file():
        raise ValueError("text mode requires a checkpoint tokenizer_config.json")
    saved = json.loads((bundle / TRAINABLE_SCOPE_FILE).read_text())["semantic_tokens"]
    ids = saved.get("token_ids", [])
    if (
        len(tokens) != 154 or len(set(tokens)) != 154 or saved.get("tokens") != list(tokens)
        or len(ids) != 154 or len(set(ids)) != 154
    ):
        raise ValueError("checkpoint must declare the exact 154-token atlas mapping")
    vocab = tokenizer.get_vocab()
    if any(
        vocab.get(token) != token_id
        or tokenizer.encode(token, add_special_tokens=False) != [token_id]
        or token_id in tokenizer.all_special_ids
        for token, token_id in zip(tokens, ids, strict=True)
    ):
        raise ValueError("actual checkpoint tokenizer mapping differs from saved atlas rows")
    if not tokenizer.chat_template or tokenizer.pad_token_id is None:
        raise ValueError("checkpoint tokenizer needs its native chat template and padding token")


def load_checkpoint_text_tokenizer(bundle: str | Path,
                                   tokens: Sequence[str]) -> PreTrainedTokenizerBase:
    transformers = importlib.import_module("transformers")
    if not (Path(bundle) / "tokenizer_config.json").is_file():
        raise ValueError("text mode requires the checkpoint's saved tokenizer")
    tokenizer: PreTrainedTokenizerBase = transformers.AutoTokenizer.from_pretrained(
        bundle, local_files_only=True,
    )
    validate_checkpoint_tokenizer(tokenizer, bundle, tokens)
    return tokenizer


def validate_text_context_budget(model: torch.nn.Module, max_sequence_length: int | None) -> None:
    max_sequence_length = validate_text_budget(max_sequence_length)
    config = getattr(model.config, "text_config", model.config)
    limit = getattr(config, "max_position_embeddings", None)
    if limit is not None and max_sequence_length > limit:
        raise ValueError(f"max_sequence_length exceeds model context limit {limit}")


def validate_text_adapter(
    base_model: torch.nn.Module, bundle: str | Path, setup: TokenSetup,
    components: ModelComponents, *, config: TrainConfig | None = None,
) -> None:
    """Admit standard rank-8/16 language LoRA and exact saved replacement rows."""
    bundle = Path(bundle)
    saved = json.loads((bundle / "adapter_config.json").read_text())
    parent = json.loads((bundle / RUN_CONFIG_FILE).read_text())
    rank = standard_lora_rank(saved, parent)
    scope = json.loads((bundle / TRAINABLE_SCOPE_FILE).read_text())["semantic_tokens"]
    expected_rows = {
        components.input_embedding: list(setup.token_ids),
        components.output_head: list(setup.token_ids),
    }
    if (
        saved.get("trainable_token_indices") != expected_rows
        or not isinstance(saved.get("target_modules"), (list, str))
        or saved.get("target_parameters") or saved.get("layer_replication")
        or len(setup.token_ids) != 154 or len(set(setup.token_ids)) != 154
        or len(setup.tokens) != 154 or len(set(setup.tokens)) != 154
        or scope.get("token_ids") != list(setup.token_ids)
        or scope.get("tokens") != list(setup.tokens)
        or (bundle / FROZEN_ADAPTER_DIR).exists()
        or (bundle / FROZEN_BUNDLE_FILE).exists()
    ):
        raise ValueError(f"text mode requires compatible language rank-{rank} LoRA + 154 input/output rows")
    if isinstance(base_model, PeftModel) or any(
        isinstance(module, (BaseTunerLayer, TrainableTokensWrapper))
        for module in base_model.modules()
    ):
        raise ValueError("text adapter validation requires an unwrapped base model")
    # PEFT 0.20 automatically replaces >=20 full paths with minimal unambiguous
    # suffixes. Validate what its real matcher selects, including exclusions and
    # layer filters, over ALL base modules (not just the desired language ones).
    adapter_config = LoraConfig.from_pretrained(str(bundle), local_files_only=True)
    intended = set(language_linear_targets(base_model, components))
    matched = {
        name for name, _ in base_model.named_modules()
        if name and check_target_module_exists(adapter_config, name)
    }
    if not intended or matched != intended:
        raise ValueError(
            f"text mode requires compatible language rank-{rank} targets: "
            f"missing={sorted(intended - matched)} extra={sorted(matched - intended)}"
        )
    if config is not None and (
        rank != config.lora_rank or saved.get("lora_alpha") != config.lora_alpha
        or saved.get("lora_dropout") != config.lora_dropout
    ):
        raise ValueError("text LoRA rank/alpha/dropout must match the saved adapter")
    modules = dict(base_model.named_modules())
    expected = expected_adapter_shapes(
        {name: (modules[name].out_features, modules[name].in_features) for name in intended},
        {components.input_embedding: components.hidden_size, components.output_head: components.hidden_size},
        rank,
    )
    actual = {name: header["shape"] for name, header in tensor_headers(bundle / "adapter_model.safetensors").items()}
    if actual != expected:
        raise ValueError(f"saved adapter tensor names/shapes must match rank-{rank} language LoRA + 154 rows")


def processor_asset_hashes(bundle: str | Path) -> dict[str, str]:
    bundle = Path(bundle)
    hashes = {
        name: sha256_file(bundle / name)
        for name in PROCESSOR_ASSET_FILES if (bundle / name).is_file()
    }
    if not ({"processor_config.json", "preprocessor_config.json"} & hashes.keys()):
        raise ValueError("text bundles require the parent's saved processor configuration assets")
    return hashes


def carry_processor_assets(source: str | Path, destination: Path) -> dict[str, str]:
    """Copy native processor configuration bytes, never image/video data or weights."""
    source = Path(source)
    expected = processor_asset_hashes(source)
    for name in expected:
        if (source / name).resolve() != (destination / name).resolve():
            shutil.copyfile(source / name, destination / name)
    if processor_asset_hashes(destination) != expected:
        raise RuntimeError("saved processor configuration assets differ from the parent")
    return expected


def save_visual_state(
    model: torch.nn.Module,
    components: ModelComponents,
    output_dir: Path,
    *,
    dtype: torch.dtype | None = None,
) -> JsonDict:

    """Write the full visual state.

    Checkpoints keep the fp32 master weights so resume is exact. Vision final
    bundles pass ``dtype=torch.bfloat16`` for their historical eval/Hub contract;
    text final bundles retain the dormant visual state at its original precision.
    """

    visual_state = {
        name: (tensor.detach() if dtype is None else tensor.detach().to(dtype)).cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if _matches_path(name, components.vision)
    }
    if not visual_state:
        raise RuntimeError("visual checkpoint state is empty")
    path = output_dir / VISUAL_STATE_FILE
    save_file(visual_state, path, metadata={"format": "pt", "scope": "full_visual"})
    return {
        "path": str(path),
        "dtype": str(next(iter(visual_state.values())).dtype),
        "tensors": len(visual_state),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_visual_state(model: torch.nn.Module, checkpoint_dir: str | Path) -> JsonDict:
    path = Path(checkpoint_dir) / VISUAL_STATE_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    state = load_file(path)
    expected = {
        name for name in model.state_dict() if _matches_path(name, catan_components(model).vision)
    } if hasattr(model, "_catan_components") else {
        name for name in model.state_dict() if ".model.visual." in f".{name}."
    }
    missing = sorted(expected - set(state))
    extra = sorted(set(state) - expected)
    if missing or extra:
        raise RuntimeError(
            "visual checkpoint key mismatch: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [name for name in incompatible.unexpected_keys if name in state]
    if unexpected:
        raise RuntimeError(f"unexpected visual checkpoint keys: {unexpected[:8]}")
    return {
        "path": str(path),
        "tensors": len(state),
        "expected_tensors": len(expected),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
