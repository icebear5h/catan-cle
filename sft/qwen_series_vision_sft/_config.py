"""The immutable vision-SFT configuration and its hardware gate."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from sft.json_types import JsonDict

from ._base import (
    DEFAULT_27B_MODEL_ID,
    HARDWARE_PROFILES,
    L40S_PROFILE,
    VISION_LANGUAGE_LORA,
    VISION_SFT_PROFILES,
)


@dataclass(frozen=True)
class VisionSftConfig:
    """One immutable native token-SFT optimizer configuration."""

    profile: str = VISION_LANGUAGE_LORA
    model_id: str = DEFAULT_27B_MODEL_ID
    max_steps: int | None = 1
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    vision_lr: float = 1e-6
    merger_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    image_min_pixels: int = 256 * 256
    image_max_pixels: int = 1024 * 1024
    max_seq_length: int = 4096
    save_steps: int = 1
    save_total_limit: int = 3
    dataloader_num_workers: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.profile not in VISION_SFT_PROFILES:
            allowed = ", ".join(VISION_SFT_PROFILES)
            raise ValueError(f"Unsupported vision SFT profile {self.profile!r}; choose {allowed}")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or None")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        for field_name in ("learning_rate", "vision_lr", "merger_lr"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.image_min_pixels <= 0:
            raise ValueError("image_min_pixels must be positive")
        if self.image_max_pixels < self.image_min_pixels:
            raise ValueError("image_max_pixels must be at least image_min_pixels")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.save_steps <= 0 or self.save_total_limit <= 0:
            raise ValueError("checkpoint save settings must be positive")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must not be negative")

    @property
    def language_lora(self) -> bool:
        return self.profile == VISION_LANGUAGE_LORA

    def as_manifest_dict(self) -> JsonDict:
        payload: JsonDict = asdict(self)
        payload["language_lora"] = self.language_lora
        payload["bits"] = 16
        payload["freeze_llm"] = True
        payload["freeze_vision_tower"] = False
        payload["freeze_merger"] = False
        payload["enable_reasoning"] = False
        return payload


def validate_hardware_profile(model_id: str, hardware: str) -> None:
    """Reject known model/hardware combinations that cannot hold BF16 weights."""

    if hardware not in HARDWARE_PROFILES:
        allowed = ", ".join(HARDWARE_PROFILES)
        raise ValueError(f"Unsupported hardware profile {hardware!r}; choose {allowed}")

    normalized_model = model_id.lower().replace("_", "-")
    if hardware == L40S_PROFILE and "qwen3.8-27b" in normalized_model:
        raise ValueError("Qwen3.8-27B BF16 vision training is not admitted on one L40S; use h200")


def default_run_name(config: VisionSftConfig) -> str:
    """Return a stable output name for one profile and training horizon."""

    model_slug = re.sub(r"[^a-z0-9]+", "-", config.model_id.split("/")[-1].lower()).strip("-")
    if config.max_steps is None:
        horizon = f"epochs-{config.num_train_epochs:g}"
    else:
        horizon = f"steps-{config.max_steps}"
    return f"{model_slug}-{config.profile.replace('_', '-')}-{horizon}"
