"""Runtime checks, base model loading, SFT config, bundle validation."""

from __future__ import annotations

import gc
import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import HfApi
from peft import PeftModel

from sft.json_types import json_dict
from sft.scripts.train.train_trl_catan_vision._common import (
    ACCELERATE_VERSION,
    DATASETS_VERSION,
    FROZEN_ADAPTER_DIR,
    PEFT_VERSION,
    RELOAD_REPORT_FILE,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
    TRL_VERSION,
    VISUAL_STATE_FILE,
    JsonDict,
    inventory_tokens,
    load_token_inventory,
    write_json_atomic,
)
from sft.scripts.train.train_trl_catan_vision._config import TokenSetup, TrainConfig
from sft.scripts.train.train_trl_catan_vision._frozen import (
    apply_frozen_adapter,
    freeze_visual_weights,
    load_checkpoint_text_tokenizer,
    load_visual_state,
    processor_asset_hashes,
    validate_text_adapter,
    validate_text_context_budget,
)
from sft.scripts.train.train_trl_catan_vision._model_tokens import prepare_semantic_tokens
from sft.scripts.train.train_trl_catan_vision._structure import required_text

if TYPE_CHECKING:  # Heavy; transformers loads lazily on the runtime path.
    from transformers import PreTrainedTokenizerBase, ProcessorMixin


def assert_runtime_versions() -> JsonDict:
    actual = {
        "torch": torch.__version__.split("+")[0],
        **{name: importlib.import_module(name).__version__ for name in (
            "transformers", "trl", "peft", "datasets", "accelerate",
        )},
    }
    expected = {
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "trl": TRL_VERSION,
        "peft": PEFT_VERSION,
        "datasets": DATASETS_VERSION,
        "accelerate": ACCELERATE_VERSION,
    }
    if actual != expected:
        raise RuntimeError(f"runtime version mismatch: expected={expected} actual={actual}")
    return actual


def _load_base_model_and_processor(
    config: TrainConfig, *, processor: ProcessorMixin | PreTrainedTokenizerBase | None = None,
) -> tuple[ProcessorMixin | PreTrainedTokenizerBase, torch.nn.Module]:
    transformers = importlib.import_module("transformers")
    if processor is None:
        if config.text_only:
            processor = load_checkpoint_text_tokenizer(
                required_text(config.initial_bundle, "initial_bundle"),
                inventory_tokens(load_token_inventory(config.token_inventory)),
            )
        else:
            processor = transformers.AutoProcessor.from_pretrained(
                config.model_id,
                size={
                    "shortest_edge": config.image_min_pixels,
                    "longest_edge": config.image_max_pixels,
                },
            )
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if config.text_only:
        validate_text_context_budget(model, config.max_sequence_length)
    model.config.use_cache = False
    return processor, model


def build_sft_config(
    config: TrainConfig,
    output_dir: str | Path,
    *,
    has_eval_dataset: bool,
) -> object:
    """Build the single authoritative TRL loss/data configuration."""

    SFTConfig = importlib.import_module("trl").SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        max_steps=-1 if config.max_steps is None else config.max_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        # Transformers 5 expresses fractional warmup through warmup_steps.
        warmup_steps=config.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps" if has_eval_dataset else "no",
        eval_steps=config.eval_steps,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        seed=config.seed,
        data_seed=config.seed,
        max_length=None,
        packing=False,
        shuffle_dataset=False,
        train_sampling_strategy="sequential",
        completion_only_loss=True,
        assistant_only_loss=False,
        # TRL normally rejects a PEFT-wrapped lm_head because its chunked path
        # reads the weight directly. CatanSFTTrainer exposes PEFT's exact,
        # differentiable TrainableTokens weight while TRL installs the patch.
        loss_type="chunked_nll",
        dataset_kwargs={"skip_prepare_dataset": True},
    )


def validate_saved_bundle(
    config: TrainConfig,
    final_dir: Path,
    inventory: JsonDict,
    expected_setup: TokenSetup,
) -> JsonDict:

    transformers = importlib.import_module("transformers")
    processor = (
        load_checkpoint_text_tokenizer(final_dir, inventory_tokens(inventory)) if config.text_only
        else transformers.AutoProcessor.from_pretrained(final_dir)
    )
    if config.text_only:
        processor_assets = processor_asset_hashes(final_dir)
        parent_bundle = required_text(config.initial_bundle, "initial_bundle")
        if processor_assets != processor_asset_hashes(parent_bundle):
            raise RuntimeError("saved processor configuration assets differ from the parent")
    base = transformers.AutoModelForMultimodalLM.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    setup, components = prepare_semantic_tokens(processor, base, inventory)
    if setup.token_ids != expected_setup.token_ids:
        raise RuntimeError("saved tokenizer changed the semantic token IDs")
    if config.text_only:
        validate_text_adapter(base, final_dir, setup, components, config=config)
    if (final_dir / FROZEN_ADAPTER_DIR).is_dir():
        base, _ = apply_frozen_adapter(base, final_dir / FROZEN_ADAPTER_DIR)
    reloaded = PeftModel.from_pretrained(base, final_dir)
    if config.text_only:
        freeze_visual_weights(reloaded, components)
    visual = load_visual_state(reloaded, final_dir)
    report: JsonDict = {
        "schema": "catan_trl_reload_validation/v1",
        "valid": True,
        "model_id": config.model_id,
        "input_mode": config.input_mode,
        "components": components.as_dict(),
        "semantic_tokens": setup.as_dict(),
        "visual_state": visual,
    }
    if config.text_only:
        report["processor_assets_sha256"] = json_dict(processor_assets)
        report.update(lora_rank=config.lora_rank, lora_alpha=config.lora_alpha)
    write_json_atomic(final_dir / RELOAD_REPORT_FILE, report)
    del reloaded, base, processor
    gc.collect()
    return report


def publish_bundle(config: TrainConfig, final_dir: Path) -> JsonDict:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required to publish the validated bundle")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.repo_info(config.hub_model_id, repo_type="model")
    commit = api.upload_folder(
        repo_id=config.hub_model_id,
        repo_type="model",
        folder_path=final_dir,
        commit_message="Upload validated Catan spatial SFT bundle",
    )
    report: JsonDict = {
        "schema": "catan_trl_hub_publish/v1",
        "repo_id": config.hub_model_id,
        "commit_url": str(commit.commit_url),
        "oid": commit.oid,
    }
    write_json_atomic(final_dir / "hub_publish.json", report)
    return report


def _write_model_card(final_dir: Path, config: TrainConfig) -> None:
    card = f"""---
license: apache-2.0
base_model: {config.model_id}
library_name: peft
tags:
- catan
- vision-language-model
- trl
- peft
---

# Catan Qwen3.8 spatial SFT

PEFT token-row adapter plus full visual encoder and merger state for
`{config.model_id}`. Load the base model, resize it using the bundled tokenizer,
load the PEFT adapter, and then load `{VISUAL_STATE_FILE}`.
"""
    (final_dir / "README.md").write_text(card)
