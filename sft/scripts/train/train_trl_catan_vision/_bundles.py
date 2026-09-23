"""Initial/frozen bundle loading and PEFT wrapping."""

from __future__ import annotations

import json
from pathlib import Path

import peft
import torch
from peft import LoraConfig, PeftConfig, PeftModel, TrainableTokensConfig

from sft.json_types import as_dict, json_dict, json_list
from sft.scripts.train.train_trl_catan_vision._common import (
    FROZEN_ADAPTER_DIR,
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
)
from sft.scripts.train.train_trl_catan_vision._frozen import (
    apply_frozen_adapter,
    freeze_visual_except_lora,
    freeze_visual_weights,
    frozen_adapter_files,
    load_visual_state,
    processor_asset_hashes,
    validate_text_adapter,
)
from sft.scripts.train.train_trl_catan_vision._model_tokens import (
    initialize_semantic_token_rows,
    language_linear_targets,
    promote_visual_master_weights,
    vision_linear_targets,
)
from sft.scripts.train.train_trl_catan_vision._visual import load_protected_bases, module_key


def _get_peft_model(model: torch.nn.Module, peft_config: PeftConfig) -> PeftModel:
    # PEFT annotates `PreTrainedModel`, but wraps any `torch.nn.Module`; the tiny
    # test models rely on that, so resolve the function without its narrow stub.
    wrapped = getattr(peft, "get_peft_model")(model, peft_config)
    if not isinstance(wrapped, PeftModel):
        raise TypeError(f"get_peft_model returned {type(wrapped).__name__}, not a PeftModel")
    return wrapped


def wrap_trainable_model(
    model: torch.nn.Module,
    setup: TokenSetup,
    components: ModelComponents,
    config: TrainConfig,
) -> PeftModel:
    model.requires_grad_(False)
    token_init = initialize_semantic_token_rows(model, components, setup, seed=config.seed, mode=config.token_init)
    token_targets = {
        components.input_embedding: list(setup.token_ids),
        components.output_head: list(setup.token_ids),
    }
    peft_config: LoraConfig | TrainableTokensConfig
    if config.language_lora:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=language_linear_targets(model, components),
            trainable_token_indices=token_targets,
        )
    else:
        peft_config = TrainableTokensConfig(
            task_type="CAUSAL_LM",
            token_indices=list(setup.token_ids),
            target_modules=list(token_targets),
            init_weights=True,
        )
    wrapped = _get_peft_model(model, peft_config)
    visual = (
        freeze_visual_weights(wrapped, components) if config.text_only
        else promote_visual_master_weights(wrapped, components)
    )
    setattr(wrapped, "_catan_components", components)
    setattr(wrapped, "_catan_token_setup", setup)
    setattr(wrapped, "_catan_initialization",
            {"semantic_rows": token_init, "visual_master_weights": visual})
    return wrapped


def load_frozen_bundle(
    base_model: torch.nn.Module,
    setup: TokenSetup,
    components: ModelComponents,
    config: TrainConfig,
) -> tuple[PeftModel, JsonDict]:
    """The O-LoRA start: exact vision weights, the parent's adapter merged, fresh adapters on top."""

    if config.frozen_bundle is None:
        raise ValueError("frozen_bundle is required")
    bundle = Path(config.frozen_bundle).expanduser().resolve()
    required = frozen_adapter_files(bundle) + (bundle / VISUAL_STATE_FILE, bundle / TRAINABLE_SCOPE_FILE)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen bundle is incomplete: {missing}")
    parent_scope = json.loads((bundle / TRAINABLE_SCOPE_FILE).read_text())
    if tuple(parent_scope.get("semantic_tokens", {}).get("token_ids", [])) != setup.token_ids:
        raise RuntimeError("incompatible frozen bundle: semantic token IDs differ")
    factors = Path(config.visual_delta_factors).expanduser().resolve() if config.visual_delta_factors else None
    if factors is not None and not factors.is_file():
        raise FileNotFoundError(factors)

    setattr(base_model, "_catan_components", components)
    merged, frozen_report = apply_frozen_adapter(base_model, bundle, restore_visual=True)
    visual = as_dict(frozen_report["visual_state"])
    merged.requires_grad_(False)
    language_targets = language_linear_targets(merged, components)
    vision_targets = vision_linear_targets(merged, components)
    token_targets = {
        components.input_embedding: list(setup.token_ids),
        components.output_head: list(setup.token_ids),
    }
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=language_targets + vision_targets,
        trainable_token_indices=token_targets,
    )
    model = _get_peft_model(merged, peft_config)
    visual_freeze = freeze_visual_except_lora(model, components)
    bases = load_protected_bases(bundle, factors)
    trainable_keys = {module_key(name) for name, parameter in model.named_parameters() if ".lora_A." in name and parameter.requires_grad}
    protected = sorted(trainable_keys & set(bases))
    unprotected = sorted(trainable_keys - set(bases))
    if not protected:
        raise RuntimeError("no trainable adapter module has a protected basis; check the frozen adapter and factor file")
    setattr(model, "_catan_components", components)
    setattr(model, "_catan_token_setup", setup)
    setattr(model, "_catan_initialization",
            {"semantic_rows": {"mode": "keep"}, "visual_master_weights": visual_freeze})
    setattr(model, "_catan_orthogonal_bases", bases)
    setattr(model, "_catan_frozen_adapter_dir", bundle)
    report: JsonDict = {
        "schema": "catan_trl_frozen_bundle/v1",
        "path": str(bundle),
        "frozen_adapter": frozen_report,
        "visual_sha256": visual["sha256"],
        "visual_delta_factors": str(factors) if factors else None,
        "visual_delta_factors_sha256": sha256_file(factors) if factors else None,
        "new_adapter": {"rank": config.lora_rank, "alpha": config.lora_alpha, "language_targets": len(language_targets), "vision_targets": len(vision_targets)},
        "protected_modules": len(protected),
        "unprotected_modules": json_list(unprotected),
        "orthogonal_lambda": config.orthogonal_lambda,
        "optimizer_state_restored": False,
    }
    return model, report


def load_initial_bundle(
    base_model: torch.nn.Module,
    setup: TokenSetup,
    components: ModelComponents,
    config: TrainConfig,
) -> tuple[PeftModel, JsonDict]:
    """Load a completed parent stage without restoring optimizer/RNG state."""

    if config.initial_bundle is None:
        raise ValueError("initial_bundle is required")
    bundle = Path(config.initial_bundle).expanduser().resolve()
    required = (
        bundle / "adapter_config.json",
        bundle / "adapter_model.safetensors",
        bundle / VISUAL_STATE_FILE,
        bundle / TRAINABLE_SCOPE_FILE,
        bundle / RUN_CONFIG_FILE,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"initial bundle is incomplete: {missing}")
    parent_config = json.loads((bundle / RUN_CONFIG_FILE).read_text())
    parent_scope = json.loads((bundle / TRAINABLE_SCOPE_FILE).read_text())
    errors = []
    if parent_config.get("model_id") != config.model_id:
        errors.append("base model differs")
    if parent_config.get("profile") != config.profile:
        errors.append("PEFT profile differs")
    if parent_config.get("lora_rank") != config.lora_rank:
        errors.append("LoRA rank differs")
    parent_tokens = parent_scope.get("semantic_tokens", {})
    if tuple(parent_tokens.get("token_ids", [])) != setup.token_ids:
        errors.append("semantic token IDs differ")
    if errors:
        raise RuntimeError("incompatible initial bundle: " + "; ".join(errors))

    if config.text_only:
        if config.token_init != "keep":
            raise ValueError("text mode requires token_init=keep")
        validate_text_adapter(base_model, bundle, setup, components, config=config)
        processor_assets = processor_asset_hashes(bundle)
        base_model.requires_grad_(False)

    if (bundle / FROZEN_ADAPTER_DIR).is_dir():
        base_model, _ = apply_frozen_adapter(base_model, bundle / FROZEN_ADAPTER_DIR)
    model = PeftModel.from_pretrained(base_model, bundle, is_trainable=True)
    setattr(model, "_catan_components", components)
    setattr(model, "_catan_token_setup", setup)
    promoted = (
        freeze_visual_weights(model, components) if config.text_only
        else promote_visual_master_weights(model, components)
    )
    visual = load_visual_state(model, bundle)
    setattr(model, "_catan_initialization",
            {"semantic_rows": None, "visual_master_weights": promoted})
    report: JsonDict = {
        "schema": "catan_trl_initial_bundle/v1",
        "path": str(bundle),
        "adapter_sha256": sha256_file(bundle / "adapter_model.safetensors"),
        "visual_sha256": visual["sha256"],
        "parent_training_config_sha256": sha256_file(bundle / RUN_CONFIG_FILE),
        "optimizer_state_restored": False,
        "scheduler_state_restored": False,
        "rng_state_restored": False,
    }
    if config.text_only:
        report.update({
            "source_input_mode": parent_config.get("input_mode", "vision"),
            "input_mode": "text", "token_rows": "PEFT replacement rows (not additive)",
            "processor_assets_sha256": json_dict(processor_assets),
            "lora_rank": config.lora_rank, "lora_alpha": config.lora_alpha,
        })
    return model, report
