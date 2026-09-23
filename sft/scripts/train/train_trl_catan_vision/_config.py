"""Training configuration dataclasses and lightweight validators."""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, cast

from sft.lora_expansion import SUPPORTED_TEXT_LORA_RANKS
from sft.scripts.train.train_trl_catan_vision._common import (
    HUB_MODEL_ID,
    INPUT_MODES,
    LORA_PROFILES,
    MODEL_ID,
    PROFILE_OLORA_FROZEN_BUNDLE,
    PROFILE_VISION_TOKENS_LORA,
    PROFILES,
    RUN_CONFIG_FILE,
    SPATIAL_TARGET_MODES,
    TOKEN_INIT_MODES,
    JsonDict,
)

if TYPE_CHECKING:  # Heavy; transformers loads lazily on the runtime path.
    import torch
    from transformers import PreTrainedTokenizerBase


# TRL is imported lazily and the Modal image pins a newer TRL/transformers than
# the local environment, so the trainer surface this package calls is described
# structurally rather than typed against whichever TRL happens to be installed.
class TrainerState(Protocol):
    log_history: list[dict[str, float]]
    global_step: int


class TrainOutput(Protocol):
    metrics: dict[str, float]


class TrainerAccelerator(Protocol):
    def backward(self, loss: torch.Tensor) -> None: ...


class CatanTrainer(Protocol):
    model: torch.nn.Module
    state: TrainerState
    accelerator: TrainerAccelerator
    data_collator: Callable[[list[JsonDict]], MutableMapping[str, torch.Tensor]]

    def add_callback(self, callback: object) -> None: ...
    def train(self, resume_from_checkpoint: str | None = ...) -> TrainOutput: ...
    def evaluate(self) -> dict[str, float]: ...
    def save_model(self, output_dir: str) -> None: ...
    def save_state(self) -> None: ...
    def _prepare_inputs(self, inputs: MutableMapping[str, torch.Tensor]) -> dict[str, object]: ...
    def compute_loss_context_manager(self) -> AbstractContextManager[None]: ...
    def compute_loss(
        self, model: torch.nn.Module, inputs: dict[str, object], return_outputs: bool = ...,
        num_items_in_batch: torch.Tensor | None = ...,
    ) -> torch.Tensor | tuple[torch.Tensor, object]: ...


class CatanTrainerFactory(Protocol):
    def __call__(
        self, *, model: torch.nn.Module, args: object, train_dataset: object,
        processing_class: object, eval_dataset: object = ...,
    ) -> CatanTrainer: ...


@dataclass(frozen=True)
class TrainConfig:
    train_jsonl: str
    image_root: str | None = None
    token_inventory: str = ""
    output_dir: str = ""
    eval_jsonl: str | None = None
    eval_image_root: str | None = None
    model_id: str = MODEL_ID
    profile: str = PROFILE_VISION_TOKENS_LORA
    hub_model_id: str = HUB_MODEL_ID
    publish_to_hub: bool = False
    require_curriculum: bool = True
    resume_from_checkpoint: str | None = None
    initial_bundle: str | None = None
    max_steps: int | None = None
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-4
    language_lora_learning_rate: float = 1e-4
    vision_learning_rate: float = 5e-6
    merger_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    image_min_pixels: int = 256 * 256
    image_max_pixels: int = 1024 * 1024
    save_steps: int = 256
    eval_steps: int = 128
    save_total_limit: int = 2
    dataloader_num_workers: int = 2
    seed: int = 42
    patch_loss_weight: float = 0.0
    patch_temperature: float = 0.07
    spatial_target_mode: str = "correct"
    token_init: str = "mean_noise"
    frozen_bundle: str | None = None
    visual_delta_factors: str | None = None
    orthogonal_lambda: float = 0.5
    vision_lora_learning_rate: float = 1e-4
    input_mode: str = "vision"
    max_sequence_length: int | None = None

    def validate(self) -> None:
        for name in ("train_jsonl", "token_inventory", "output_dir"):
            if not getattr(self, name) or not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}")
        if not self.text_only and not (self.image_root and str(self.image_root).strip()):
            raise ValueError("image_root is required in vision mode")
        if self.text_only:
            if (
                self.profile != PROFILE_VISION_TOKENS_LORA
                or type(self.lora_rank) is not int or self.lora_rank not in SUPPORTED_TEXT_LORA_RANKS
            ):
                raise ValueError("text mode requires vision_tokens_lora with rank 8 or 16")
            if self.resume_from_checkpoint:
                raise ValueError(
                    "text-mode resume is not supported; use initial_bundle for a fresh optimizer"
                )
            if not self.initial_bundle or self.token_init != "keep":
                raise ValueError("text mode requires initial_bundle and token_init=keep")
            if self.patch_loss_weight != 0 or self.spatial_target_mode != "correct":
                raise ValueError("text mode does not support patch objectives or shuffled targets")
            validate_text_budget(self.max_sequence_length)
        if self.resume_from_checkpoint:
            validate_resume_mode(self.resume_from_checkpoint, self.input_mode)
        if self.profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}; received {self.profile!r}")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.publish_to_hub and not self.hub_model_id.strip():
            raise ValueError("hub_model_id is required when publishing")
        if not self.text_only and bool(self.eval_jsonl) != bool(self.eval_image_root):
            raise ValueError("eval_jsonl and eval_image_root must be supplied together")
        if self.text_only and self.eval_image_root and not self.eval_jsonl:
            raise ValueError("eval_image_root requires eval_jsonl")
        if self.resume_from_checkpoint and self.initial_bundle:
            raise ValueError("resume_from_checkpoint and initial_bundle are mutually exclusive")
        if self.olora:
            if not self.frozen_bundle:
                raise ValueError("the olora profile needs frozen_bundle")
            if self.initial_bundle:
                raise ValueError("frozen_bundle and initial_bundle are mutually exclusive")
            if self.token_init != "keep":
                raise ValueError("the olora profile keeps the frozen bundle's rows; use token_init=keep")
            if self.orthogonal_lambda < 0:
                raise ValueError("orthogonal_lambda must be non-negative")
        elif self.frozen_bundle:
            raise ValueError("frozen_bundle needs the olora profile")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or None")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.per_device_train_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch size and gradient accumulation must be positive")
        for name in (
            "learning_rate",
            "language_lora_learning_rate",
            "vision_learning_rate",
            "merger_learning_rate",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("weight_decay must be nonnegative and warmup_ratio must be in [0, 1)")
        if self.lora_rank <= 0 or self.lora_alpha <= 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid LoRA settings")
        if self.image_min_pixels <= 0 or self.image_max_pixels < self.image_min_pixels:
            raise ValueError("invalid image pixel bounds")
        if self.save_steps <= 0 or self.eval_steps <= 0 or self.save_total_limit <= 0:
            raise ValueError("checkpoint settings must be positive")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be nonnegative")
        if self.patch_loss_weight < 0:
            raise ValueError("patch_loss_weight must be nonnegative")
        if self.patch_temperature <= 0:
            raise ValueError("patch_temperature must be positive")
        if self.token_init not in TOKEN_INIT_MODES:
            raise ValueError(f"token_init must be one of {TOKEN_INIT_MODES}; got {self.token_init!r}")
        if self.spatial_target_mode not in SPATIAL_TARGET_MODES:
            raise ValueError(
                f"spatial_target_mode must be one of {SPATIAL_TARGET_MODES}; "
                f"received {self.spatial_target_mode!r}"
            )

    @property
    def language_lora(self) -> bool:
        return self.profile in LORA_PROFILES

    @property
    def text_only(self) -> bool:
        return self.input_mode == "text"

    @property
    def olora(self) -> bool:
        return self.profile == PROFILE_OLORA_FROZEN_BUNDLE


@dataclass(frozen=True)
class ModelComponents:
    input_embedding: str
    output_head: str
    language: str
    vision: str
    merger: str
    vocab_size: int
    hidden_size: int

    def as_dict(self) -> JsonDict:
        return asdict(self)


def normalize_training_config(payload: JsonDict) -> JsonDict:
    """Fill only the two new input defaults for semantic comparisons.

    Never hash or persist this view in place of an original historical receipt.
    Other missing fields, unknown fields, and explicit nondefaults stay distinct.
    """
    return {"input_mode": "vision", "max_sequence_length": None, **payload}


@dataclass(frozen=True)
class TokenSetup:
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    tokenizer_size: int
    model_vocab_size: int
    added_tokens: int
    family_word_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["tokens"] = list(self.tokens)
        payload["token_ids"] = list(self.token_ids)
        return payload


def native_tokenizer(processor: object) -> PreTrainedTokenizerBase:
    """The text tokenizer of a multimodal processor, or `processor` itself."""
    return cast("PreTrainedTokenizerBase", getattr(processor, "tokenizer", processor))


def validate_text_budget(max_sequence_length: int | None) -> int:
    if (
        not isinstance(max_sequence_length, int) or isinstance(max_sequence_length, bool)
        or max_sequence_length <= 0
    ):
        raise ValueError("text mode requires a positive max_sequence_length (no truncation)")
    return max_sequence_length


def validate_resume_mode(checkpoint: str | Path, input_mode: str) -> None:
    path = Path(checkpoint) / RUN_CONFIG_FILE
    if path.is_file():
        saved = json.loads(path.read_text()).get("input_mode", "vision")
        if saved != input_mode:
            raise ValueError("cross-mode resume is forbidden; use initial_bundle for a fresh optimizer")
