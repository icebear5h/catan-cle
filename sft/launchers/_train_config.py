"""Rebuild a `TrainConfig` from its decoded JSON receipt with checked field types.

Equivalent to `TrainConfig(**payload)`: unknown fields and a missing
`train_jsonl` raise `TypeError` like the dataclass constructor would.
"""

from __future__ import annotations

from dataclasses import fields

from sft.json_types import JsonDict
from sft.launchers._config_fields import (
    bool_field,
    float_field,
    int_field,
    opt_int_field,
    opt_str_field,
    reject_dropped,
    reject_unknown,
    str_field,
)
from sft.scripts.train.train_trl_catan_vision import TrainConfig

__all__ = ["train_config"]

_FIELDS = frozenset(f.name for f in fields(TrainConfig))


def train_config(payload: JsonDict) -> TrainConfig:
    """Equivalent to `TrainConfig(**payload)` for a JSON-decoded config."""
    reject_unknown(payload, _FIELDS, "TrainConfig")
    if "train_jsonl" not in payload:
        raise TypeError("TrainConfig missing required field 'train_jsonl'")
    d = TrainConfig(train_jsonl="")
    p = payload
    config = TrainConfig(
        train_jsonl=str_field(p, "train_jsonl", ""),
        image_root=opt_str_field(p, "image_root", d.image_root),
        token_inventory=str_field(p, "token_inventory", d.token_inventory),
        output_dir=str_field(p, "output_dir", d.output_dir),
        eval_jsonl=opt_str_field(p, "eval_jsonl", d.eval_jsonl),
        eval_image_root=opt_str_field(p, "eval_image_root", d.eval_image_root),
        model_id=str_field(p, "model_id", d.model_id),
        profile=str_field(p, "profile", d.profile),
        hub_model_id=str_field(p, "hub_model_id", d.hub_model_id),
        publish_to_hub=bool_field(p, "publish_to_hub", d.publish_to_hub),
        require_curriculum=bool_field(p, "require_curriculum", d.require_curriculum),
        resume_from_checkpoint=opt_str_field(p, "resume_from_checkpoint", d.resume_from_checkpoint),
        initial_bundle=opt_str_field(p, "initial_bundle", d.initial_bundle),
        max_steps=opt_int_field(p, "max_steps", d.max_steps),
        num_train_epochs=float_field(p, "num_train_epochs", d.num_train_epochs),
        per_device_train_batch_size=int_field(p, "per_device_train_batch_size", d.per_device_train_batch_size),
        per_device_eval_batch_size=int_field(p, "per_device_eval_batch_size", d.per_device_eval_batch_size),
        gradient_accumulation_steps=int_field(p, "gradient_accumulation_steps", d.gradient_accumulation_steps),
        learning_rate=float_field(p, "learning_rate", d.learning_rate),
        language_lora_learning_rate=float_field(p, "language_lora_learning_rate", d.language_lora_learning_rate),
        vision_learning_rate=float_field(p, "vision_learning_rate", d.vision_learning_rate),
        merger_learning_rate=float_field(p, "merger_learning_rate", d.merger_learning_rate),
        weight_decay=float_field(p, "weight_decay", d.weight_decay),
        warmup_ratio=float_field(p, "warmup_ratio", d.warmup_ratio),
        lora_rank=int_field(p, "lora_rank", d.lora_rank),
        lora_alpha=int_field(p, "lora_alpha", d.lora_alpha),
        lora_dropout=float_field(p, "lora_dropout", d.lora_dropout),
        image_min_pixels=int_field(p, "image_min_pixels", d.image_min_pixels),
        image_max_pixels=int_field(p, "image_max_pixels", d.image_max_pixels),
        save_steps=int_field(p, "save_steps", d.save_steps),
        eval_steps=int_field(p, "eval_steps", d.eval_steps),
        save_total_limit=int_field(p, "save_total_limit", d.save_total_limit),
        dataloader_num_workers=int_field(p, "dataloader_num_workers", d.dataloader_num_workers),
        seed=int_field(p, "seed", d.seed),
        patch_loss_weight=float_field(p, "patch_loss_weight", d.patch_loss_weight),
        patch_temperature=float_field(p, "patch_temperature", d.patch_temperature),
        spatial_target_mode=str_field(p, "spatial_target_mode", d.spatial_target_mode),
        token_init=str_field(p, "token_init", d.token_init),
        frozen_bundle=opt_str_field(p, "frozen_bundle", d.frozen_bundle),
        visual_delta_factors=opt_str_field(p, "visual_delta_factors", d.visual_delta_factors),
        orthogonal_lambda=float_field(p, "orthogonal_lambda", d.orthogonal_lambda),
        vision_lora_learning_rate=float_field(p, "vision_lora_learning_rate", d.vision_lora_learning_rate),
        input_mode=str_field(p, "input_mode", d.input_mode),
        max_sequence_length=opt_int_field(p, "max_sequence_length", d.max_sequence_length),
    )
    reject_dropped(config, payload, "TrainConfig")
    return config
