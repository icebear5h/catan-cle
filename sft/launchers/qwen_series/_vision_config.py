"""Rebuild a `VisionSftConfig` from its decoded JSON payload with checked field types."""

from __future__ import annotations

from dataclasses import fields

from sft.json_types import JsonDict
from sft.launchers._config_fields import (
    float_field,
    int_field,
    opt_int_field,
    reject_dropped,
    reject_unknown,
    str_field,
)
from sft.qwen_series_vision_sft import VisionSftConfig

__all__ = ["vision_config"]

_FIELDS = frozenset(f.name for f in fields(VisionSftConfig))


def vision_config(payload: JsonDict) -> VisionSftConfig:
    """Equivalent to `VisionSftConfig(**payload)` for a JSON-decoded config."""
    reject_unknown(payload, _FIELDS, "VisionSftConfig")
    d = VisionSftConfig()
    p = payload
    config = VisionSftConfig(
        profile=str_field(p, "profile", d.profile),
        model_id=str_field(p, "model_id", d.model_id),
        max_steps=opt_int_field(p, "max_steps", d.max_steps),
        num_train_epochs=float_field(p, "num_train_epochs", d.num_train_epochs),
        per_device_train_batch_size=int_field(p, "per_device_train_batch_size", d.per_device_train_batch_size),
        gradient_accumulation_steps=int_field(p, "gradient_accumulation_steps", d.gradient_accumulation_steps),
        learning_rate=float_field(p, "learning_rate", d.learning_rate),
        vision_lr=float_field(p, "vision_lr", d.vision_lr),
        merger_lr=float_field(p, "merger_lr", d.merger_lr),
        weight_decay=float_field(p, "weight_decay", d.weight_decay),
        warmup_ratio=float_field(p, "warmup_ratio", d.warmup_ratio),
        lora_rank=int_field(p, "lora_rank", d.lora_rank),
        lora_alpha=int_field(p, "lora_alpha", d.lora_alpha),
        lora_dropout=float_field(p, "lora_dropout", d.lora_dropout),
        image_min_pixels=int_field(p, "image_min_pixels", d.image_min_pixels),
        image_max_pixels=int_field(p, "image_max_pixels", d.image_max_pixels),
        max_seq_length=int_field(p, "max_seq_length", d.max_seq_length),
        save_steps=int_field(p, "save_steps", d.save_steps),
        save_total_limit=int_field(p, "save_total_limit", d.save_total_limit),
        dataloader_num_workers=int_field(p, "dataloader_num_workers", d.dataloader_num_workers),
        seed=int_field(p, "seed", d.seed),
    )
    reject_dropped(config, payload, "VisionSftConfig")
    return config
