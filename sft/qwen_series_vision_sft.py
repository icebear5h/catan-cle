"""LEGACY configuration and launch guards for native Qwen vision SFT.

The active contract is ``sft/scripts/train_trl_catan_vision.py``. The existing
Modal smoke launcher intentionally keeps the complete model path
frozen and quantized. This module defines a separate BF16 path that trains the
full vision tower and multimodal merger while keeping base language weights
frozen. It has no Modal dependency so launch plans can be tested locally.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals.catan_board_bench.tokens import recognition_token_inventory
from sft.paths import (
    repository_relative_path,
    resolve_dataset_asset,
    resolve_dataset_image,
)
from sft.scripts.convert_to_qwen_series_sft import iter_jsonl


VISION_ONLY = "vision_only"
VISION_LANGUAGE_LORA = "vision_language_lora"
VISION_SFT_PROFILES = (VISION_ONLY, VISION_LANGUAGE_LORA)

L40S_PROFILE = "l40s"
H200_PROFILE = "h200"
HARDWARE_PROFILES = (L40S_PROFILE, H200_PROFILE)

DEFAULT_4B_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_27B_MODEL_ID = "Qwen/Qwen3.8-27B"
MAX_PROMPT_CHARACTERS = 4096
MAX_SHORT_ANSWER_CHARACTERS = 512


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

    def as_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
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


def build_vision_sft_command(
    config: VisionSftConfig,
    *,
    train_json: str,
    image_folder: str,
    output_dir: str,
    token_inventory: str | None = None,
) -> list[str]:
    """Build the exact pinned-upstream command for one vision SFT profile."""

    command = [
        "python",
        "-m",
        "sft.scripts.train_qwen_series_with_catan_tokens",
        "--catan_patch_no_deepspeed_savers",
        "--catan_trainable_profile",
        config.profile,
        "--model_id",
        config.model_id,
        "--data_path",
        train_json,
        "--image_folder",
        image_folder,
        "--output_dir",
        output_dir,
        "--remove_unused_columns",
        "False",
        # The pinned trainer uses the PEFT lifecycle to save selective token
        # rows and full non-LoRA visual weights for both profiles.
        "--lora_enable",
        "True",
        "--vision_lora",
        "False",
        "--lora_namespan_exclude",
        '["lm_head", "embed_tokens", "embed_token"]',
        "--lora_rank",
        str(config.lora_rank),
        "--lora_alpha",
        str(config.lora_alpha),
        "--lora_dropout",
        str(config.lora_dropout),
        "--num_lora_modules",
        "-1",
        # Full vision-tower updates are intentionally BF16, never QLoRA.
        "--bits",
        "16",
        "--freeze_llm",
        "True",
        "--freeze_vision_tower",
        "False",
        "--freeze_merger",
        "False",
        "--enable_reasoning",
        "False",
        "--bf16",
        "True",
        "--fp16",
        "False",
        "--disable_flash_attn2",
        "True",
        "--use_liger_kernel",
        "False",
        "--num_train_epochs",
        str(config.num_train_epochs),
        "--per_device_train_batch_size",
        str(config.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation_steps),
        "--image_min_pixels",
        str(config.image_min_pixels),
        "--image_max_pixels",
        str(config.image_max_pixels),
        "--max_seq_length",
        str(config.max_seq_length),
        "--learning_rate",
        str(config.learning_rate),
        "--vision_lr",
        str(config.vision_lr),
        "--merger_lr",
        str(config.merger_lr),
        "--weight_decay",
        str(config.weight_decay),
        "--warmup_ratio",
        str(config.warmup_ratio),
        "--lr_scheduler_type",
        "cosine",
        "--max_grad_norm",
        "1.0",
        "--optim",
        "adamw_torch",
        "--logging_steps",
        "1",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(config.save_steps),
        "--save_total_limit",
        str(config.save_total_limit),
        "--report_to",
        "none",
        "--lazy_preprocess",
        "True",
        "--gradient_checkpointing",
        "True",
        "--tf32",
        "True",
        "--dataloader_num_workers",
        str(config.dataloader_num_workers),
        "--seed",
        str(config.seed),
        "--data_seed",
        str(config.seed),
    ]
    if token_inventory is not None:
        command.extend(["--catan_token_inventory", token_inventory])
    if config.profile == VISION_ONLY:
        command.append("--catan_token_adapter_only")
    if config.max_steps is not None:
        command.extend(["--max_steps", str(config.max_steps)])
    return command


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_text(content: Any) -> tuple[str, int]:
    if isinstance(content, str):
        return content, content.count("<image>")
    if not isinstance(content, list):
        raise TypeError(f"unsupported message content type: {type(content)!r}")

    text_parts = []
    image_count = 0
    for item in content:
        if item.get("type") == "image":
            image_count += 1
        elif item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))
    return "\n".join(text_parts), image_count


def _short_answer_pair(row: dict[str, Any]) -> tuple[str, str]:
    row_id = row.get("id")
    if "messages" in row:
        messages = row["messages"]
        if len(messages) != 2:
            raise ValueError(f"row {row_id!r} must contain exactly one user/assistant pair")
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise ValueError(f"row {row_id!r} must be ordered user then assistant")
        prompt, image_count = _content_text(messages[0].get("content", ""))
        answer, answer_images = _content_text(messages[1].get("content", ""))
        image_count += answer_images
    elif "conversations" in row:
        conversations = row["conversations"]
        if len(conversations) != 2:
            raise ValueError(f"row {row_id!r} must contain exactly one human/gpt pair")
        if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
            raise ValueError(f"row {row_id!r} must be ordered human then gpt")
        prompt, image_count = _content_text(conversations[0].get("value", ""))
        answer, answer_images = _content_text(conversations[1].get("value", ""))
        image_count += answer_images
    else:
        raise ValueError(f"row {row_id!r} has neither messages nor conversations")

    if image_count != 1:
        raise ValueError(f"row {row_id!r} must contain exactly one image placeholder")
    prompt = prompt.strip()
    answer = answer.strip()
    if not prompt or not answer:
        raise ValueError(f"row {row_id!r} has an empty prompt or answer")
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError(f"row {row_id!r} prompt exceeds {MAX_PROMPT_CHARACTERS} characters")
    if len(answer) > MAX_SHORT_ANSWER_CHARACTERS:
        raise ValueError(
            f"row {row_id!r} answer exceeds the short-answer limit of "
            f"{MAX_SHORT_ANSWER_CHARACTERS} characters"
        )
    return prompt, answer


def load_recognition_token_inventory(path: Path) -> dict[str, Any]:
    """Load and fail closed on the exact 154+6+60 recognition inventory."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text())
    expected = recognition_token_inventory()
    if payload != expected or payload.get("counts") != {
        "atlas": 154,
        "query": 6,
        "answer": 60,
        "total": 220,
    }:
        raise ValueError("token inventory must be exactly the 220 replay_v1 tokens")
    return payload


def fingerprint_training_dataset(
    train_jsonl: Path,
    *,
    image_root: Path | None = None,
    token_inventory: Path | None = None,
) -> dict[str, Any]:
    """Hash annotations, explicit image-root contents, and token inventory."""

    train_jsonl = train_jsonl.expanduser().resolve()
    if not train_jsonl.is_file():
        raise FileNotFoundError(train_jsonl)

    image_records: dict[str, str] = {}
    row_count = 0
    max_prompt_characters = 0
    max_answer_characters = 0
    for _, row in iter_jsonl(train_jsonl):
        row_count += 1
        prompt, answer = _short_answer_pair(row)
        max_prompt_characters = max(max_prompt_characters, len(prompt))
        max_answer_characters = max(max_answer_characters, len(answer))
        image_ref = row.get("image")
        if not image_ref:
            raise ValueError(f"row {row.get('id')!r} has no image")
        image_path = (
            resolve_dataset_image(image_root, image_ref)
            if image_root is not None
            else resolve_dataset_asset(train_jsonl, image_ref)
        )
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_records.setdefault(str(image_ref), _sha256_file(image_path))

    if row_count == 0:
        raise ValueError(f"training dataset is empty: {train_jsonl}")

    inventory_payload = None
    if token_inventory is not None:
        inventory = load_recognition_token_inventory(token_inventory)
        inventory_payload = {
            "reference": repository_relative_path(token_inventory),
            "sha256": _sha256_file(token_inventory.expanduser().resolve()),
            "counts": inventory["counts"],
        }
    payload = {
        "source_sha256": _sha256_file(train_jsonl),
        "annotation_root": repository_relative_path(train_jsonl.parent),
        "image_root": (
            repository_relative_path(image_root)
            if image_root is not None
            else "legacy_annotation_or_repository_resolution"
        ),
        "token_inventory": inventory_payload,
        "rows": row_count,
        "unique_images": len(image_records),
        "max_prompt_characters": max_prompt_characters,
        "max_answer_characters": max_answer_characters,
        "images": [
            {"reference": reference, "sha256": digest}
            for reference, digest in sorted(image_records.items())
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["combined_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def launch_identity(payload: dict[str, Any]) -> str:
    """Hash the immutable portion of one remote launch manifest."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
