"""Native Hugging Face TRL/PEFT training for Catan board grounding.

This module owns the complete model-side contract: dataset normalization,
curriculum-order validation, semantic token insertion, PEFT wrapping, optimizer
groups, checkpoint serialization, reload validation, and optional Hub upload.
It intentionally has no Modal or ms-swift dependency.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import types
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, TrainableTokensConfig, get_peft_model
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils.other import TrainableTokensWrapper
from safetensors.torch import load_file, save_file

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory
from sft.board_state_readout import score_board_state
from sft.lora_expansion import (
    SUPPORTED_TEXT_LORA_RANKS, expected_adapter_shapes, standard_lora_rank, tensor_headers,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
HUB_MODEL_ID = "TetraCorp/catan-qwen3.8-27b-spatial-sft"
PROFILE_VISION_TOKENS = "vision_tokens"
PROFILE_VISION_TOKENS_LORA = "vision_tokens_lora"
# O-LoRA rung: a finished bundle is merged into the base as a frozen task, the
# vision tower stays at that bundle's exact weights, and fresh rank-r adapters
# on the language layers and the vision tower learn the next task with their
# input rows held orthogonal to the frozen task's subspaces.
PROFILE_OLORA_FROZEN_BUNDLE = "olora_frozen_bundle"
PROFILES = (PROFILE_VISION_TOKENS, PROFILE_VISION_TOKENS_LORA, PROFILE_OLORA_FROZEN_BUNDLE)
LORA_PROFILES = (PROFILE_VISION_TOKENS_LORA, PROFILE_OLORA_FROZEN_BUNDLE)
FROZEN_BUNDLE_FILE = "frozen_bundle.json"
FROZEN_ADAPTER_DIR = "frozen_adapter"
VISION_LORA_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2", "merger.linear_fc1", "merger.linear_fc2")
CURRICULUM_STAGES = (
    "spatial_grounding",
    "clean_board_grounding",
    "pieces_and_colors",
    "real_game_distribution",
)

TRANSFORMERS_VERSION = "5.16.1"
TRL_VERSION = "1.12.0"
PEFT_VERSION = "0.20.0"
DATASETS_VERSION = "5.0.1"
ACCELERATE_VERSION = "1.14.0"
TORCH_VERSION = "2.13.0"
TORCHVISION_VERSION = "0.28.0"
HUGGINGFACE_HUB_VERSION = "1.29.0"
SAFETENSORS_VERSION = "0.8.0"
PILLOW_VERSION = "12.3.0"

VISUAL_STATE_FILE = "visual_model.safetensors"
TRAINABLE_SCOPE_FILE = "trainable_parameters.json"
OPTIMIZER_COVERAGE_FILE = "optimizer_coverage.json"
DATASET_REPORT_FILE = "dataset_contract.json"
RELOAD_REPORT_FILE = "reload_validation.json"
RUN_CONFIG_FILE = "training_config.json"
INITIAL_BUNDLE_FILE = "initial_bundle.json"
# Transformers 5 stores image/video settings in processor_config.json; older
# bundles use separate preprocessor files. Carry their bytes without instantiating
# a multimodal processor in the text pipeline.
PROCESSOR_ASSET_FILES = (
    "processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
)
# Row guard: one-phrase heads answer in a few words, and a full-board readout (54
# nodes or 72 edges, empties explicit) runs to about 1,500 characters. Anything
# past this is a broken row, not a long one.
MAX_PROMPT_CHARACTERS = 4096
MAX_ANSWER_CHARACTERS = 2048
PATCH_METRICS_FILE = "patch_localization_config.json"
SPATIAL_TARGET_MODES = ("correct", "shuffled")
INPUT_MODES = ("vision", "text")
TEXT_MEDIA_KEYS = frozenset({
    "image", "images", "image_url", "video", "videos", "audio", "audios",
    "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
})
# Completion tokens that every row shares; they are excluded from the
# answer-only metrics so a plateau cannot hide behind end-of-turn accuracy.
ANSWER_METRIC_TRIVIAL_TOKENS = ("<|im_end|>", "\n")
SEMANTIC_ROW_NOISE_SCALE = 0.1
TOKEN_INIT_MODES = ("mean_noise", "vocab_gaussian", "family_words", "keep")
# Base-vocabulary words whose embeddings seed each atlas family under the
# family_words initialization; the leading space matches how the words appear
# mid-sentence in the training prompts.
FAMILY_WORDS = {"N": " node", "E": " edge", "T": " tile", "P": " port"}
IMAGE_HASH_WORKERS = 16


JsonDict = dict[str, Any]


class _ChunkedNLLTrainableTokensHead:
    """Expose PEFT TrainableTokens as one differentiable output-head weight.

    TRL's chunked NLL intentionally reads ``lm_head.weight`` instead of calling
    the module so it can project only supervised positions. PEFT's
    ``TrainableTokensLayer.weight`` is the frozen base weight, while
    ``get_merged_weights`` is the exact functional weight with the selected
    trainable rows replaced. This view gives TRL that functional weight without
    merging the adapter in-place or making the full output head trainable.
    """

    def __init__(self, head: torch.nn.Module) -> None:
        self._head = head

    @property
    def weight(self) -> torch.Tensor:
        head = self._head
        if hasattr(head, "token_adapter"):
            # LoraConfig(trainable_token_indices=...) uses PEFT's auxiliary
            # TrainableTokensWrapper. Its weight property is already the
            # differentiable merged view from the inner token adapter.
            return head.weight
        active_adapters = list(head.active_adapters)
        if head.disable_adapters or head.merged or not active_adapters:
            return head.get_base_layer().weight
        return head.get_merged_weights(active_adapters)

    @property
    def bias(self) -> torch.Tensor | None:
        head = self._head
        adapter = head.token_adapter if hasattr(head, "token_adapter") else head
        return getattr(adapter.get_base_layer(), "bias", None)


@contextmanager
def expose_trainable_tokens_head_to_chunked_nll(model: torch.nn.Module) -> Iterator[None]:
    """Let TRL capture a PEFT-aware head while installing chunked NLL.

    The override exists only during ``SFTTrainer.__init__``. TRL's patched
    forward retains the lightweight view in its closure; the model immediately
    regains its normal ``get_output_embeddings`` method for saving, generation,
    and reload validation.
    """

    if not isinstance(model, PeftModel):
        yield
        return
    target = model.get_base_model()
    head = target.get_output_embeddings()
    if not isinstance(head, (BaseTunerLayer, TrainableTokensWrapper)):
        yield
        return
    if not isinstance(head, (TrainableTokensLayer, TrainableTokensWrapper)):
        raise TypeError(
            "chunked NLL only supports a PEFT-wrapped output head when the wrapper is "
            f"TrainableTokensLayer; received {type(head).__name__}"
        )

    view = _ChunkedNLLTrainableTokensHead(head)
    sentinel = object()
    previous = target.__dict__.get("get_output_embeddings", sentinel)

    def get_output_embeddings(_self: torch.nn.Module) -> _ChunkedNLLTrainableTokensHead:
        return view

    target.get_output_embeddings = types.MethodType(get_output_embeddings, target)
    try:
        yield
    finally:
        if previous is sentinel:
            delattr(target, "get_output_embeddings")
        else:
            target.get_output_embeddings = previous


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


def write_json_atomic(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, JsonDict]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def load_token_inventory(path: str | Path) -> JsonDict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text())
    expected = semantic_recognition_token_inventory()
    if payload != expected or payload.get("counts") != {
        "atlas": 154,
        "edge": 72,
        "node": 54,
        "port": 9,
        "tile": 19,
        "total": 154,
    }:
        raise ValueError("token inventory must be the exact 154-token semantic atlas")
    return payload


def _content_text(content: Any) -> tuple[str, int]:
    if isinstance(content, str):
        return content.replace("<image>", "").strip(), content.count("<image>")
    if not isinstance(content, list):
        raise TypeError(f"unsupported message content: {type(content)!r}")
    text = []
    images = 0
    for part in content:
        if part.get("type") in {"image", "image_url"}:
            images += 1
        elif part.get("type") == "text":
            text.append(str(part.get("text", "")))
    return "\n".join(text).strip(), images


def _text_content(content: Any, *, line_number: int) -> str:
    if isinstance(content, list):
        if not content or any(
            not isinstance(part, dict) or set(part) != {"type", "text"}
            or part["type"] != "text" or not isinstance(part["text"], str)
            for part in content
        ):
            raise ValueError(f"line {line_number} text mode allows only text content parts")
        content = "\n".join(part["text"] for part in content)
    if not isinstance(content, str):
        raise ValueError(f"line {line_number} text content must be a string")
    if any(marker in content for marker in ("<image>", "<video>", "<audio>", "<|")):
        raise ValueError(f"line {line_number} text content contains media or chat control tokens")
    return content.strip()


def _message_pair(
    row: JsonDict, *, line_number: int, input_mode: str = "vision",
) -> tuple[str, str]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    if input_mode == "text":
        if TEXT_MEDIA_KEYS.intersection(row):
            raise ValueError(f"line {line_number} text row must not contain media fields")
        if row.get("spatial_targets"):
            raise ValueError(f"line {line_number} text row must not contain spatial patch targets")
        if "messages" in row and "conversations" in row:
            raise ValueError(f"line {line_number} has ambiguous conversation formats")
    if "messages" in row:
        messages = row["messages"]
        role_key, user_role, assistant_role, content_key = "role", "user", "assistant", "content"
    elif "conversations" in row:
        messages = row["conversations"]
        role_key, user_role, assistant_role, content_key = "from", "human", "gpt", "value"
    else:
        raise ValueError(f"line {line_number} has neither messages nor conversations")
    if not isinstance(messages, list) or len(messages) != 2 or not all(isinstance(m, dict) for m in messages):
        raise ValueError(f"line {line_number} must contain one user/assistant pair")
    if messages[0].get(role_key) != user_role or messages[1].get(role_key) != assistant_role:
        raise ValueError(f"line {line_number} has invalid message ordering")
    if input_mode == "text":
        if any(TEXT_MEDIA_KEYS.intersection(message) for message in messages):
            raise ValueError(f"line {line_number} text messages must not contain media fields")
        prompt, answer = (
            _text_content(message.get(content_key, ""), line_number=line_number)
            for message in messages
        )
    else:
        prompt, prompt_images = _content_text(messages[0].get(content_key, ""))
        answer, answer_images = _content_text(messages[1].get(content_key, ""))
        if prompt_images + answer_images != 1:
            raise ValueError(f"line {line_number} must contain exactly one image placeholder")
    if not prompt or not answer:
        raise ValueError(f"line {line_number} contains an empty prompt or answer")
    answer_limit = MAX_ANSWER_CHARACTERS
    if row.get("task_type") == "full_board_readout":
        answer_limit = 4096
        score_board_state(answer, answer)  # Reject incomplete or repeated target addresses.
    if input_mode == "vision" and (len(prompt) > MAX_PROMPT_CHARACTERS or len(answer) > answer_limit):
        raise ValueError(f"line {line_number} exceeds the answer-length contract")
    return prompt, answer


def native_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def validate_text_budget(max_sequence_length: int | None) -> None:
    if (
        not isinstance(max_sequence_length, int) or isinstance(max_sequence_length, bool)
        or max_sequence_length <= 0
    ):
        raise ValueError("text mode requires a positive max_sequence_length (no truncation)")


def text_chat_ids(tokenizer: Any, messages: list[JsonDict], *, generation: bool) -> list[int]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("text mode requires the checkpoint's saved native chat template")
    return list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation,
        enable_thinking=False, preserve_thinking=False, return_dict=False,
    ))


def encode_text_pair(
    tokenizer: Any, prompt: str, answer: str, *, max_sequence_length: int,
) -> JsonDict:
    """Check the native token boundary; supervise the answer AND end-of-turn."""
    validate_text_budget(max_sequence_length)
    prompt = _text_content(prompt, line_number=0)
    answer = _text_content(answer, line_number=0)
    if not prompt or not answer:
        raise ValueError("text prompt and answer must be nonempty")
    messages = [{"role": "user", "content": prompt}]
    prefix = text_chat_ids(tokenizer, messages, generation=True)
    complete = text_chat_ids(
        tokenizer, messages + [{"role": "assistant", "content": answer}], generation=False,
    )
    if not prefix or complete[:len(prefix)] != prefix:
        raise ValueError("native chat template changed the prompt/completion token boundary")
    eot = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    completion = complete[len(prefix):]
    if (
        len(eot) != 1 or eot[0] not in tokenizer.all_special_ids
        or completion.count(eot[0]) != 1 or completion.index(eot[0]) == 0
    ):
        raise ValueError("native completion must contain answer tokens and one supervised end-of-turn")
    if len(complete) > max_sequence_length:
        raise ValueError(
            f"text sequence has {len(complete)} tokens, exceeds max_sequence_length="
            f"{max_sequence_length}; truncation is forbidden"
        )
    return {"input_ids": complete, "labels": [-100] * len(prefix) + completion}


def pad_text_inputs(tokenizer: Any, features: list[JsonDict], *, left: bool) -> JsonDict:
    """Pad by position, never by token ID (pad and EOT may share an ID)."""
    if not features or tokenizer.pad_token_id is None:
        raise ValueError("text batches require rows and a native padding token")
    width = max(len(item["input_ids"]) for item in features)
    ids, masks, labels = [], [], []
    for item in features:
        size = len(item["input_ids"])
        padding = width - size
        before, after = (padding, 0) if left else (0, padding)
        ids.append([tokenizer.pad_token_id] * before + item["input_ids"] + [tokenizer.pad_token_id] * after)
        masks.append([0] * before + [1] * size + [0] * after)
        if "labels" in item:
            labels.append([-100] * before + item["labels"] + [-100] * after)
    result = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}
    if labels:
        if len(labels) != len(features):
            raise ValueError("mixed labeled and unlabeled text rows")
        result["labels"] = torch.tensor(labels)
    return result


class TextCompletionCollator:
    """Explicit native-template collation, independent of TRL's VLM detection."""

    def __init__(self, tokenizer: Any, *, max_sequence_length: int) -> None:
        validate_text_budget(max_sequence_length)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, examples: list[JsonDict]) -> JsonDict:
        features = []
        for example in examples:
            prompt, answer = _message_pair(example, line_number=0, input_mode="text")
            features.append(encode_text_pair(
                self.tokenizer, prompt, answer, max_sequence_length=self.max_sequence_length,
            ))
        return pad_text_inputs(self.tokenizer, features, left=False)


def _image_reference(row: JsonDict, *, line_number: int) -> str:
    if row.get("image"):
        return str(row["image"])
    images = row.get("images")
    if isinstance(images, list) and len(images) == 1:
        return str(images[0])
    raise ValueError(f"line {line_number} must reference exactly one image")


def resolve_image_path(root: Path, row: JsonDict, *, line_number: int) -> Path:
    image_path = (root / _image_reference(row, line_number=line_number)).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"line {line_number} image escapes image_root: {image_path}") from exc
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return image_path


def validate_spatial_targets(row: JsonDict, *, line_number: int) -> list[JsonDict]:
    """Validate optional normalized localization targets without changing them."""

    raw_targets = row.get("spatial_targets", [])
    if raw_targets is None:
        return []
    if not isinstance(raw_targets, list) or len(raw_targets) > 1:
        raise ValueError(f"line {line_number} spatial_targets must contain zero or one target")
    validated: list[JsonDict] = []
    for target in raw_targets:
        if not isinstance(target, dict):
            raise ValueError(f"line {line_number} spatial target must be an object")
        token = str(target.get("token", ""))
        entity_type = str(target.get("entity_type", ""))
        expected_prefix = {"node": "<N", "edge": "<E", "tile": "<T", "port": "<P"}.get(
            entity_type
        )
        if expected_prefix is None or not token.startswith(expected_prefix) or not token.endswith(">"):
            raise ValueError(f"line {line_number} has invalid spatial token/type: {target!r}")
        for key in ("bbox", "control_bbox"):
            bbox = target.get(key)
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"line {line_number} spatial {key} must have four values")
            if not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in bbox):
                raise ValueError(f"line {line_number} spatial {key} must be normalized")
            if not bbox[0] < bbox[2] or not bbox[1] < bbox[3]:
                raise ValueError(f"line {line_number} spatial {key} has invalid corners")
        center = target.get("center")
        if (
            not isinstance(center, list)
            or len(center) != 2
            or not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in center)
        ):
            raise ValueError(f"line {line_number} spatial center must be normalized")
        validated.append(target)
    return validated


def inspect_jsonl_contract(
    train_jsonl: str | Path,
    image_root: str | Path | None = None,
    *,
    require_curriculum: bool,
    input_mode: str = "vision",
    tokenizer: Any = None,
    max_sequence_length: int | None = None,
) -> JsonDict:
    source = Path(train_jsonl).expanduser().resolve()
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    text_only = input_mode == "text"
    if text_only:
        validate_text_budget(max_sequence_length)
        if tokenizer is None:
            raise ValueError("text dataset inspection requires the checkpoint tokenizer")
    elif image_root is None:
        raise ValueError("image_root is required in vision mode")
    root = None if text_only else Path(image_root).expanduser().resolve()
    if not source.is_file() or (root is not None and not root.is_dir()):
        raise FileNotFoundError(source if not source.is_file() else root)
    rows = 0
    stages: dict[str, int] = {}
    stage_spans: dict[str, dict[str, int]] = {}
    previous_stage = -1
    image_paths: set[Path] = set()
    max_prompt = 0
    max_answer = 0
    max_tokens = total_tokens = supervised_tokens = 0
    spatial_target_rows = 0
    spatial_target_types: dict[str, int] = {}
    for line_number, row in iter_jsonl(source):
        prompt, answer = _message_pair(row, line_number=line_number, input_mode=input_mode)
        if text_only:
            try:
                encoded = encode_text_pair(
                    tokenizer, prompt, answer, max_sequence_length=max_sequence_length,
                )
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
            total_tokens += len(encoded["input_ids"])
            max_tokens = max(max_tokens, len(encoded["input_ids"]))
            supervised_tokens += sum(value != -100 for value in encoded["labels"])
            spatial_targets = []
        else:
            image_paths.add(resolve_image_path(root, row, line_number=line_number))
            spatial_targets = validate_spatial_targets(row, line_number=line_number)
        if spatial_targets:
            spatial_target_rows += 1
            entity_type = spatial_targets[0]["entity_type"]
            spatial_target_types[entity_type] = spatial_target_types.get(entity_type, 0) + 1
        stage = row.get("curriculum_stage")
        if stage is None:
            if require_curriculum:
                raise ValueError(f"line {line_number} is missing curriculum_stage")
            stage = CURRICULUM_STAGES[0]
        if stage not in CURRICULUM_STAGES:
            raise ValueError(f"line {line_number} has invalid curriculum_stage={stage!r}")
        stage_index = CURRICULUM_STAGES.index(stage)
        if stage_index < previous_stage:
            raise ValueError(
                f"curriculum order regressed at line {line_number}: {stage!r}"
            )
        previous_stage = stage_index
        stages[stage] = stages.get(stage, 0) + 1
        span = stage_spans.setdefault(
            stage,
            {"start_row": rows, "end_row_exclusive": rows + 1},
        )
        span["end_row_exclusive"] = rows + 1
        max_prompt = max(max_prompt, len(prompt))
        max_answer = max(max_answer, len(answer))
        rows += 1
    if rows == 0:
        raise ValueError("training dataset is empty")
    if require_curriculum and tuple(stages) != CURRICULUM_STAGES:
        raise ValueError(
            "production curriculum must contain all four stages in order; "
            f"received {tuple(stages)}"
        )
    # Hash images concurrently: on a cold Modal volume a serial loop over a few
    # thousand files runs at roughly 1 MB/s and keeps the GPU idle for minutes.
    ordered_paths = sorted(image_paths)
    digests = []
    if not text_only:
        with ThreadPoolExecutor(max_workers=IMAGE_HASH_WORKERS) as pool:
            digests = list(pool.map(sha256_file, ordered_paths))
    image_manifest = [
        {"path": path.relative_to(root).as_posix(), "sha256": digest}
        for path, digest in zip(ordered_paths, digests, strict=True)
    ]
    image_manifest_sha256 = hashlib.sha256(
        json.dumps(image_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema": "catan_trl_dataset_contract/v1",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "image_root": str(root) if root is not None else None,
        "rows": rows,
        "unique_images": len(image_paths),
        "stages": [
            {
                "name": stage,
                "rows": stages[stage],
                **stage_spans[stage],
            }
            for stage in stages
        ],
        "image_manifest_sha256": image_manifest_sha256,
        "max_prompt_characters": max_prompt,
        "max_answer_characters": max_answer,
        "spatial_target_rows": spatial_target_rows,
        "spatial_target_types": dict(sorted(spatial_target_types.items())),
        "require_curriculum": require_curriculum,
    }
    if text_only:
        report.update({
            "input_mode": "text", "image_manifest_sha256": None,
            "max_sequence_length": max_sequence_length, "max_sequence_tokens": max_tokens,
            "total_sequence_tokens": total_tokens, "supervised_tokens": supervised_tokens,
            "truncation": False,
        })
    return report


def load_training_dataset(config: TrainConfig, *, tokenizer: Any = None) -> tuple[Any, JsonDict]:
    if config.text_only:
        return load_text_dataset(
            config.train_jsonl, tokenizer=tokenizer,
            max_sequence_length=config.max_sequence_length,
            require_curriculum=config.require_curriculum,
        )
    return load_vision_dataset(
        config.train_jsonl,
        config.image_root,
        require_curriculum=config.require_curriculum,
    )


def load_text_dataset(
    jsonl_path: str | Path, *, tokenizer: Any, max_sequence_length: int,
    require_curriculum: bool,
) -> tuple[Any, JsonDict]:
    datasets = importlib.import_module("datasets")
    report = inspect_jsonl_contract(
        jsonl_path, require_curriculum=require_curriculum, input_mode="text",
        tokenizer=tokenizer, max_sequence_length=max_sequence_length,
    )
    examples = []
    for line_number, row in iter_jsonl(Path(jsonl_path).expanduser().resolve()):
        prompt, answer = _message_pair(row, line_number=line_number, input_mode="text")
        examples.append({"messages": [
            {"role": "user", "content": prompt}, {"role": "assistant", "content": answer},
        ]})
    return datasets.Dataset.from_list(examples), report


def load_vision_dataset(
    jsonl_path: str | Path,
    image_root: str | Path,
    *,
    require_curriculum: bool,
) -> tuple[Any, JsonDict]:
    datasets = importlib.import_module("datasets")

    report = inspect_jsonl_contract(
        jsonl_path,
        image_root,
        require_curriculum=require_curriculum,
    )
    root = Path(image_root).expanduser().resolve()
    examples = []
    for line_number, row in iter_jsonl(Path(jsonl_path).expanduser().resolve()):
        prompt, answer = _message_pair(row, line_number=line_number)
        image_path = resolve_image_path(root, row, line_number=line_number)
        stage = row.get("curriculum_stage", CURRICULUM_STAGES[0])
        examples.append(
            {
                "image": str(image_path),
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "completion": [{"role": "assistant", "content": answer}],
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                },
                "curriculum_stage": stage,
                "curriculum_stage_index": CURRICULUM_STAGES.index(stage),
                "spatial_targets": validate_spatial_targets(row, line_number=line_number),
            }
        )
    dataset = datasets.Dataset.from_list(examples).cast_column("image", datasets.Image(decode=True))
    return dataset, report


class SpatialTargetCollator:
    """Preserve normalized localization targets around TRL's VLM collator."""

    def __init__(
        self,
        base_collator: Callable[[list[JsonDict]], JsonDict],
        setup: TokenSetup,
        *,
        target_mode: str,
    ) -> None:
        if target_mode not in SPATIAL_TARGET_MODES:
            raise ValueError(f"unsupported target mode: {target_mode}")
        self.base_collator = base_collator
        self.token_to_id = dict(zip(setup.tokens, setup.token_ids, strict=True))
        self.target_mode = target_mode

    def __call__(self, examples: list[JsonDict]) -> JsonDict:
        targets = [example.get("spatial_targets") or [] for example in examples]
        cleaned = []
        for example in examples:
            item = dict(example)
            item.pop("spatial_targets", None)
            cleaned.append(item)
        output = self.base_collator(cleaned)
        token_ids = []
        bboxes = []
        mask = []
        for row_targets in targets:
            if not row_targets:
                token_ids.append(0)
                bboxes.append([0.0, 0.0, 1.0, 1.0])
                mask.append(False)
                continue
            if len(row_targets) != 1:
                raise ValueError("the patch objective currently requires exactly one target per row")
            target = row_targets[0]
            token = target["token"]
            if token not in self.token_to_id:
                raise ValueError(f"spatial target is outside the 154-token inventory: {token}")
            bbox_key = "bbox" if self.target_mode == "correct" else "control_bbox"
            token_ids.append(self.token_to_id[token])
            bboxes.append([float(value) for value in target[bbox_key]])
            mask.append(True)
        output["spatial_target_token_ids"] = torch.tensor(token_ids, dtype=torch.long)
        output["spatial_target_bboxes"] = torch.tensor(bboxes, dtype=torch.float32)
        output["spatial_target_mask"] = torch.tensor(mask, dtype=torch.bool)
        return output


def _merged_patch_counts(image_grid_thw: torch.Tensor, merge_size: int) -> list[int]:
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(f"image_grid_thw must have shape [batch, 3], got {image_grid_thw.shape}")
    counts = []
    for temporal, height, width in image_grid_thw.detach().cpu().tolist():
        if height % merge_size or width % merge_size:
            raise ValueError(
                f"vision grid {(temporal, height, width)} is not divisible by merge size {merge_size}"
            )
        counts.append(int(temporal * (height // merge_size) * (width // merge_size)))
    return counts


def split_merged_visual_features(
    pooled_output: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    merge_size: int,
) -> list[torch.Tensor]:
    """Split Qwen's post-merger patch stream into one tensor per image."""

    counts = _merged_patch_counts(image_grid_thw, merge_size)
    if pooled_output.ndim == 2:
        if pooled_output.shape[0] != sum(counts):
            raise ValueError(
                f"pooled visual rows {pooled_output.shape[0]} do not match merged grids {counts}"
            )
        return list(pooled_output.split(counts, dim=0))
    if pooled_output.ndim == 3 and pooled_output.shape[0] == len(counts):
        if any(count > pooled_output.shape[1] for count in counts):
            raise ValueError("batched pooled visual output is shorter than its grid")
        return [pooled_output[index, :count] for index, count in enumerate(counts)]
    raise ValueError(f"unsupported pooled visual output shape: {pooled_output.shape}")


def soft_patch_target(
    bbox: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a bbox-aware soft distribution over the merged patch grid."""

    device = bbox.device
    ys = (torch.arange(grid_height, device=device, dtype=torch.float32) + 0.5) / grid_height
    xs = (torch.arange(grid_width, device=device, dtype=torch.float32) + 0.5) / grid_width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x1, y1, x2, y2 = bbox.float()
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    sigma_x = torch.clamp((x2 - x1) / 2, min=1.0 / grid_width)
    sigma_y = torch.clamp((y2 - y1) / 2, min=1.0 / grid_height)
    gaussian = torch.exp(
        -0.5 * (((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2)
    )
    patch_x1, patch_x2 = xx - 0.5 / grid_width, xx + 0.5 / grid_width
    patch_y1, patch_y2 = yy - 0.5 / grid_height, yy + 0.5 / grid_height
    overlap_x = torch.clamp(torch.minimum(patch_x2, x2) - torch.maximum(patch_x1, x1), min=0)
    overlap_y = torch.clamp(torch.minimum(patch_y2, y2) - torch.maximum(patch_y1, y1), min=0)
    overlap = overlap_x * overlap_y
    weights = gaussian + (overlap > 0).float()
    target = (weights / weights.sum()).flatten()
    return target, xx.flatten(), yy.flatten()


def spatial_patch_loss(
    pooled_output: torch.Tensor,
    image_grid_thw: torch.Tensor,
    token_embeddings: torch.Tensor,
    bboxes: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    merge_size: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return normalized soft CE, tolerant top-1 accuracy, and target count."""

    features_by_image = split_merged_visual_features(
        pooled_output,
        image_grid_thw,
        merge_size=merge_size,
    )
    losses = []
    correct = []
    grids = image_grid_thw.detach().cpu().tolist()
    for index, (features, grid) in enumerate(zip(features_by_image, grids, strict=True)):
        if not bool(target_mask[index].item()):
            continue
        temporal, raw_height, raw_width = (int(value) for value in grid)
        if temporal != 1:
            raise ValueError("Catan patch localization supports one image frame per row")
        grid_height, grid_width = raw_height // merge_size, raw_width // merge_size
        feature_vectors = torch.nn.functional.normalize(features.float(), dim=-1)
        token_vector = torch.nn.functional.normalize(token_embeddings[index].float(), dim=-1)
        if feature_vectors.shape[-1] != token_vector.shape[-1]:
            raise ValueError(
                "post-merger visual features and semantic token embeddings must share LM space"
            )
        scores = feature_vectors @ token_vector / temperature
        target, xs, ys = soft_patch_target(
            bboxes[index],
            grid_height=grid_height,
            grid_width=grid_width,
        )
        target = target.to(scores.device)
        normalized = -(target * torch.log_softmax(scores, dim=0)).sum()
        normalized = normalized / max(math.log(scores.numel()), 1.0)
        losses.append(normalized)

        top = int(scores.detach().argmax().item())
        x1, y1, x2, y2 = bboxes[index].float()
        tolerance_x, tolerance_y = 1.0 / grid_width, 1.0 / grid_height
        correct.append(
            (xs[top] >= x1 - tolerance_x)
            & (xs[top] <= x2 + tolerance_x)
            & (ys[top] >= y1 - tolerance_y)
            & (ys[top] <= y2 + tolerance_y)
        )
    if not losses:
        zero = pooled_output.sum() * 0.0
        return zero, zero.detach(), 0
    return torch.stack(losses).mean(), torch.stack(correct).float().mean(), len(losses)


class VisionPoolerCapture:
    """Hold the differentiable post-merger output from the current forward pass."""

    def __init__(self, vision_module: torch.nn.Module) -> None:
        self.output: torch.Tensor | None = None
        self.handle = vision_module.register_forward_hook(self._capture)

    def _capture(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        pooled = getattr(output, "pooler_output", None)
        if pooled is None and isinstance(output, (tuple, list)) and len(output) > 1:
            pooled = output[1]
        if not isinstance(pooled, torch.Tensor):
            raise RuntimeError("Qwen vision forward did not expose a tensor pooler_output")
        self.output = pooled

    def take(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("vision pooler output was not captured for this batch")
        output = self.output
        self.output = None
        return output


def module_name_for_instance(model: torch.nn.Module, target: torch.nn.Module) -> str:
    names = [name for name, module in model.named_modules() if module is target and name]
    if len(names) != 1:
        raise ValueError(f"expected one module path for {type(target).__name__}; found {names}")
    return names[0]


def discover_components(model: torch.nn.Module) -> ModelComponents:
    input_embedding = model.get_input_embeddings()
    output_head = model.get_output_embeddings()
    visual = model.model.visual
    merger = visual.merger
    language = model.model.language_model
    if not isinstance(input_embedding, torch.nn.Embedding):
        raise TypeError("input embedding must be torch.nn.Embedding")
    if not isinstance(output_head, torch.nn.Linear) or output_head.bias is not None:
        raise TypeError("untied output head must be a biasless torch.nn.Linear")
    if input_embedding.weight is output_head.weight:
        raise ValueError("Catan selective input/output rows require untied weights")
    vocab_size, hidden_size = input_embedding.weight.shape
    if tuple(output_head.weight.shape) != (vocab_size, hidden_size):
        raise ValueError("input embedding and output head shapes disagree")
    components = ModelComponents(
        input_embedding=module_name_for_instance(model, input_embedding),
        output_head=module_name_for_instance(model, output_head),
        language=module_name_for_instance(model, language),
        vision=module_name_for_instance(model, visual),
        merger=module_name_for_instance(model, merger),
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
    )
    if getattr(model.config, "model_type", None) == "qwen3_5":
        expected = {
            "input_embedding": "model.language_model.embed_tokens",
            "output_head": "lm_head",
            "language": "model.language_model",
            "vision": "model.visual",
            "merger": "model.visual.merger",
        }
        actual = {key: getattr(components, key) for key in expected}
        if actual != expected:
            raise ValueError(f"Qwen3.5 component paths changed: {actual}")
    return components


def prepare_semantic_tokens(
    processor: Any,
    model: torch.nn.Module,
    inventory: JsonDict,
) -> tuple[TokenSetup, ModelComponents]:
    tokenizer = native_tokenizer(processor)
    tokens = tuple(inventory["tokens"])
    if len(tokens) != 154 or len(set(tokens)) != 154:
        raise ValueError("semantic inventory must contain 154 unique tokens")
    before = tokenizer.get_vocab()
    present = [token for token in tokens if token in before]
    if present and len(present) != len(tokens):
        raise ValueError("tokenizer contains only part of the semantic inventory")
    added = tokenizer.add_tokens(list(tokens), special_tokens=False)
    if added != (0 if present else 154):
        raise ValueError(f"tokenizer added {added} tokens unexpectedly")
    current_vocab = int(model.get_input_embeddings().weight.shape[0])
    requested_vocab = max(len(tokenizer), current_vocab)
    model.resize_token_embeddings(
        requested_vocab,
        pad_to_multiple_of=128,
        mean_resizing=False,
    )
    components = discover_components(model)
    token_ids = tuple(int(value) for value in tokenizer.convert_tokens_to_ids(list(tokens)))
    if len(set(token_ids)) != 154 or min(token_ids) < 0 or max(token_ids) >= components.vocab_size:
        raise ValueError("semantic token IDs do not address 154 unique model rows")
    if set(token_ids) & set(getattr(tokenizer, "all_special_ids", [])):
        raise ValueError("semantic atlas tokens must be regular tokens")
    for token, token_id in zip(tokens, token_ids, strict=True):
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"semantic token is not atomic: {token}")
    family_word_ids = {
        family: tuple(int(value) for value in tokenizer.encode(word, add_special_tokens=False))
        for family, word in FAMILY_WORDS.items()
    }
    if any(not ids or max(ids) >= min(token_ids) for ids in family_word_ids.values()):
        raise ValueError("family words must tokenize to base-vocabulary ids")
    setup = TokenSetup(
        tokens=tokens,
        token_ids=token_ids,
        tokenizer_size=len(tokenizer),
        model_vocab_size=components.vocab_size,
        added_tokens=added,
        family_word_ids=family_word_ids,
    )
    return setup, components


def initialize_semantic_token_rows(
    model: torch.nn.Module,
    components: ModelComponents,
    setup: TokenSetup,
    *,
    seed: int,
    mode: str = "mean_noise",
) -> JsonDict:
    """Seed the 154 atlas rows from the base vocabulary before PEFT copies them.

    Qwen's embedding matrix is already padded past the tokenizer, so
    ``resize_token_embeddings`` never touches the new ids and they would
    otherwise start from the checkpoint's untrained padding rows. Under
    ``mean_noise`` each row becomes the mean of the original vocabulary plus
    small seeded noise so the rows are distinct from the first step. Under
    ``family_words`` each row starts from the base embedding of its family
    word (" node", " edge", " tile", " port") plus the same noise, so nodes,
    edges, tiles, and ports carry a shared per-family direction from the
    first step instead of one undifferentiated atlas direction. Under
    ``vocab_gaussian`` each row is drawn from the base vocabulary's own
    per-dimension mean and standard deviation, so the 154 rows start as far
    apart as random real words rather than as one direction plus a tenth of
    that spread.
    """

    if mode not in TOKEN_INIT_MODES:
        raise ValueError(f"unknown token init mode: {mode}")
    if mode == "keep":
        return {"mode": mode, "note": "rows left as loaded; the frozen bundle already merged its atlas rows"}
    reference_rows = min(setup.token_ids)
    if reference_rows <= 0:
        raise ValueError("semantic token ids must follow the base vocabulary")
    if mode == "family_words" and not setup.family_word_ids:
        raise ValueError("family_words init needs family_word_ids on the token setup")
    ids = torch.tensor(setup.token_ids, dtype=torch.long)
    families = [token[1] for token in setup.tokens]
    generator = torch.Generator().manual_seed(int(seed))
    report: JsonDict = {
        "mode": mode,
        "reference_rows": reference_rows,
        "noise_scale": 1.0 if mode == "vocab_gaussian" else SEMANTIC_ROW_NOISE_SCALE,
    }
    for name, path in (
        ("input_embedding", components.input_embedding),
        ("output_head", components.output_head),
    ):
        weight = resolve_wrapped_module(model, path).weight
        if weight.shape[0] <= max(setup.token_ids):
            raise ValueError(f"{path} does not contain the semantic rows")
        ids = ids.to(weight.device)
        with torch.no_grad():
            reference = weight[:reference_rows].float()
            mean = reference.mean(dim=0)
            noise_std = reference.std(dim=0) * (1.0 if mode == "vocab_gaussian" else SEMANTIC_ROW_NOISE_SCALE)
            del reference
            before = weight[ids].float().norm(dim=1)
            noise = torch.randn((len(setup.token_ids), weight.shape[1]), generator=generator)
            if mode == "family_words":
                family_base = {
                    family: weight[torch.tensor(word_ids, device=weight.device)].float().mean(dim=0)
                    for family, word_ids in setup.family_word_ids.items()
                }
                base = torch.stack([family_base[family] for family in families])
            else:
                base = mean.unsqueeze(0).expand(len(setup.token_ids), -1)
            rows = base + noise.to(mean.device) * noise_std.unsqueeze(0)
            weight[ids] = rows.to(weight.dtype)
            after = weight[ids].float().norm(dim=1)
        report[name] = {
            "mean_row_norm": float(mean.norm()),
            "row_norm_before": float(before.mean()),
            "row_norm_after": float(after.mean()),
            "row_norm_std_after": float(after.std()),
        }
        if mode == "family_words":
            report[name]["family_base_norms"] = {family: float(vec.norm()) for family, vec in family_base.items()}
    return report


def promote_visual_master_weights(model: torch.nn.Module, components: ModelComponents) -> JsonDict:
    """Keep fp32 master weights for the full visual path.

    The base model loads in bf16. AdamW steps at the vision and merger rates are
    smaller than half a bf16 ulp for most weights, so leaving those parameters
    in bf16 silently discards nearly every update. Autocast still runs the
    matmuls in bf16; only the stored parameters and optimizer states widen.
    """

    visual = resolve_wrapped_module(model, components.vision)
    visual.float()
    visual.requires_grad_(True)
    dtypes = Counter(str(parameter.dtype) for parameter in visual.parameters())
    if set(dtypes) != {"torch.float32"}:
        raise RuntimeError(f"visual module is not fully fp32 after promotion: {dict(dtypes)}")
    return {"module": components.vision, "dtypes": dict(dtypes), "tensors": sum(dtypes.values())}


def trivial_completion_token_ids(tokenizer: Any) -> tuple[int, ...]:
    ids = []
    for text in ANSWER_METRIC_TRIVIAL_TOKENS:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"expected one token id for {text!r}; received {encoded}")
        ids.append(int(encoded[0]))
    return tuple(ids)


def output_head_weight(head: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the functional output weight for a plain or PEFT-wrapped head."""

    if hasattr(head, "token_adapter") or hasattr(head, "get_merged_weights"):
        view = _ChunkedNLLTrainableTokensHead(head)
        return view.weight, view.bias
    return head.weight, getattr(head, "bias", None)


def answer_token_metrics(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    trivial_token_ids: Sequence[int],
) -> dict[str, float]:
    """Teacher-forced accuracy over answer tokens only.

    ``hidden_states`` is the final language hidden state ``[batch, seq, hidden]``
    and ``labels`` the unshifted label tensor with ``-100`` on unsupervised
    positions. Trivial completion tokens are excluded, so the result is the
    accuracy of the content the evaluator will score. ``answer_row_exact`` is
    the fraction of rows whose answer tokens are all argmax-correct.
    """

    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels disagree on batch or sequence length")
    shifted = labels[:, 1:]
    hidden = hidden_states[:, :-1]
    mask = shifted != -100
    for token_id in trivial_token_ids:
        mask &= shifted != int(token_id)
    if not bool(mask.any()):
        return {"answer_token_accuracy": 0.0, "answer_row_exact": 0.0, "answer_token_count": 0.0}
    with torch.no_grad():
        selected = hidden[mask].float()
        logits = selected @ weight.float().t()
        if bias is not None:
            logits = logits + bias.float()
        correct = logits.argmax(dim=-1) == shifted[mask]
        rows = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(shifted)[mask]
        row_correct = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
        row_correct[rows[~correct]] = False
        answered = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        answered[rows] = True
    return {
        "answer_token_accuracy": float(correct.float().mean()),
        "answer_row_exact": float((row_correct & answered).sum() / answered.sum()),
        "answer_token_count": float(mask.sum()),
    }


class LanguageHiddenCapture:
    """Hold the final language hidden state from the current forward pass."""

    def __init__(self, language_module: torch.nn.Module) -> None:
        self.output: torch.Tensor | None = None
        self.handle = language_module.register_forward_hook(self._capture)

    def _capture(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("language forward did not expose a tensor last_hidden_state")
        self.output = hidden

    def take(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("language hidden state was not captured for this batch")
        output = self.output
        self.output = None
        return output


def _matches_path(name: str, path: str) -> bool:
    return name == path or name.endswith(f".{path}") or f".{path}." in f".{name}."


def resolve_wrapped_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    matches = [
        module
        for name, module in model.named_modules()
        if name == path or name.endswith(f".{path}")
    ]
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise ValueError(f"expected one wrapped module ending in {path!r}; found {len(unique)}")
    return unique[0]


def language_linear_targets(model: torch.nn.Module, components: ModelComponents) -> list[str]:
    prefix = components.language + ".layers."
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith(prefix) and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise RuntimeError("no language linear modules were found for LoRA")
    return targets


def vision_linear_targets(model: torch.nn.Module, components: ModelComponents) -> list[str]:
    """Attention, MLP and merger projections of the vision tower, the O-LoRA vision targets."""

    prefix = components.vision + "."
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith(prefix) and isinstance(module, torch.nn.Linear) and name.endswith(VISION_LORA_SUFFIXES)
    ]
    if not targets:
        raise RuntimeError("no vision linear modules were found for LoRA")
    return targets


def wrap_trainable_model(
    model: torch.nn.Module,
    setup: TokenSetup,
    components: ModelComponents,
    config: TrainConfig,
) -> torch.nn.Module:
    model.requires_grad_(False)
    token_init = initialize_semantic_token_rows(model, components, setup, seed=config.seed, mode=config.token_init)
    token_targets = {
        components.input_embedding: list(setup.token_ids),
        components.output_head: list(setup.token_ids),
    }
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
    wrapped = get_peft_model(model, peft_config)
    visual = (
        freeze_visual_weights(wrapped, components) if config.text_only
        else promote_visual_master_weights(wrapped, components)
    )
    wrapped._catan_components = components
    wrapped._catan_token_setup = setup
    wrapped._catan_initialization = {"semantic_rows": token_init, "visual_master_weights": visual}
    return wrapped


def module_key(parameter_name: str) -> str:
    """The base-model module path an adapter or factor tensor belongs to.

    ``base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.default.weight``
    and ``model.visual.blocks.0.attn.qkv.lora_A.weight`` both map to their
    ``model....`` module path, so bases from a frozen adapter file and from the
    visual-delta factor file address the same modules as the live model.
    """

    name = parameter_name
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for marker in (".lora_A.", ".lora_B."):
        if marker in name:
            return name.split(marker, 1)[0]
    return name


def orthonormal_rows(matrix: torch.Tensor) -> torch.Tensor:
    """An orthonormal basis (rows) of the row space of ``matrix``, in fp32."""

    rows = matrix.detach().float()
    q, r = torch.linalg.qr(rows.T)
    keep = torch.abs(torch.diagonal(r)) > 1e-6
    return q[:, keep].T.contiguous()


def load_protected_bases(
    frozen_adapter_dir: Path,
    visual_delta_factors: Path | None,
) -> dict[str, torch.Tensor]:
    """Orthonormal input bases per module: the frozen adapter's ``lora_A`` rows, plus the visual delta's."""

    bases: dict[str, torch.Tensor] = {}
    adapter = load_file(frozen_adapter_dir / "adapter_model.safetensors")
    for name, tensor in adapter.items():
        if ".lora_A." in name:
            bases[module_key(name)] = orthonormal_rows(tensor)
    if visual_delta_factors is not None:
        factors = load_file(visual_delta_factors)
        for name, tensor in factors.items():
            if ".lora_A." in name:
                key = module_key(name)
                if key in bases:
                    raise RuntimeError(f"protected basis defined twice: {key}")
                bases[key] = orthonormal_rows(tensor)
    if not bases:
        raise RuntimeError("no protected bases were found")
    return bases


def orthogonal_penalty(
    model: torch.nn.Module,
    bases: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, float, int]:
    """O-LoRA's L_orth: the squared projection of every trainable ``lora_A`` onto its module's protected basis.

    Returns the penalty, the fraction of the adapters' total squared norm that
    lies inside the protected subspaces (a scale-free progress number), and the
    count of trainable adapters with no basis (which are unconstrained).
    """

    penalty = None
    inside = 0.0
    total = 0.0
    unprotected = 0
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name or not parameter.requires_grad:
            continue
        key = module_key(name)
        basis = bases.get(key)
        if basis is None:
            unprotected += 1
            continue
        cached = cache.get(key)
        if cached is None or cached.device != parameter.device:
            cached = basis.to(parameter.device)
            cache[key] = cached
        rows = parameter.float()
        projection = rows @ cached.T
        term = (projection * projection).sum()
        penalty = term if penalty is None else penalty + term
        inside += float(term.detach())
        total += float((rows.detach() * rows.detach()).sum())
    if penalty is None:
        raise RuntimeError("no trainable lora_A parameter has a protected basis")
    return penalty, (inside / total if total else 0.0), unprotected


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
    report = {"path": str(frozen_dir), "adapter_sha256": sha256_file(frozen_dir / "adapter_model.safetensors")}
    if restore_visual:
        # Parent bundles save the PEFT-wrapped state_dict names. Restore while
        # that same wrapper is present, before merging it or adding vision LoRA
        # (which would introduce base_layer names). Promote before copying so
        # the parent's FP32 visual weights never round through BF16.
        components = base_model._catan_components
        frozen._catan_components = components
        resolve_wrapped_module(frozen, components.vision).float()
        report["visual_state"] = load_visual_state(frozen, frozen_dir)
    merged = frozen.merge_and_unload()
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


def validate_resume_mode(checkpoint: str | Path, input_mode: str) -> None:
    path = Path(checkpoint) / RUN_CONFIG_FILE
    if path.is_file():
        saved = json.loads(path.read_text()).get("input_mode", "vision")
        if saved != input_mode:
            raise ValueError("cross-mode resume is forbidden; use initial_bundle for a fresh optimizer")


def validate_checkpoint_tokenizer(
    tokenizer: Any, bundle: str | Path, tokens: Sequence[str],
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


def load_checkpoint_text_tokenizer(bundle: str | Path, tokens: Sequence[str]) -> Any:
    transformers = importlib.import_module("transformers")
    if not (Path(bundle) / "tokenizer_config.json").is_file():
        raise ValueError("text mode requires the checkpoint's saved tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(bundle, local_files_only=True)
    validate_checkpoint_tokenizer(tokenizer, bundle, tokens)
    return tokenizer


def validate_text_context_budget(model: Any, max_sequence_length: int) -> None:
    validate_text_budget(max_sequence_length)
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
    adapter_config = LoraConfig.from_pretrained(bundle, local_files_only=True)
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


def load_frozen_bundle(
    base_model: torch.nn.Module,
    setup: TokenSetup,
    components: ModelComponents,
    config: TrainConfig,
) -> tuple[torch.nn.Module, JsonDict]:
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

    base_model._catan_components = components
    merged, frozen_report = apply_frozen_adapter(base_model, bundle, restore_visual=True)
    visual = frozen_report["visual_state"]
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
    model = get_peft_model(merged, peft_config)
    visual_freeze = freeze_visual_except_lora(model, components)
    bases = load_protected_bases(bundle, factors)
    trainable_keys = {module_key(name) for name, parameter in model.named_parameters() if ".lora_A." in name and parameter.requires_grad}
    protected = sorted(trainable_keys & set(bases))
    unprotected = sorted(trainable_keys - set(bases))
    if not protected:
        raise RuntimeError("no trainable adapter module has a protected basis; check the frozen adapter and factor file")
    model._catan_components = components
    model._catan_token_setup = setup
    model._catan_initialization = {"semantic_rows": {"mode": "keep"}, "visual_master_weights": visual_freeze}
    model._catan_orthogonal_bases = bases
    model._catan_frozen_adapter_dir = bundle
    report = {
        "schema": "catan_trl_frozen_bundle/v1",
        "path": str(bundle),
        "frozen_adapter": frozen_report,
        "visual_sha256": visual["sha256"],
        "visual_delta_factors": str(factors) if factors else None,
        "visual_delta_factors_sha256": sha256_file(factors) if factors else None,
        "new_adapter": {"rank": config.lora_rank, "alpha": config.lora_alpha, "language_targets": len(language_targets), "vision_targets": len(vision_targets)},
        "protected_modules": len(protected),
        "unprotected_modules": unprotected,
        "orthogonal_lambda": config.orthogonal_lambda,
        "optimizer_state_restored": False,
    }
    return model, report


def load_initial_bundle(
    base_model: torch.nn.Module,
    setup: TokenSetup,
    components: ModelComponents,
    config: TrainConfig,
) -> tuple[torch.nn.Module, JsonDict]:
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
    model._catan_components = components
    model._catan_token_setup = setup
    promoted = (
        freeze_visual_weights(model, components) if config.text_only
        else promote_visual_master_weights(model, components)
    )
    visual = load_visual_state(model, bundle)
    model._catan_initialization = {"semantic_rows": None, "visual_master_weights": promoted}
    report = {
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
            "processor_assets_sha256": processor_assets,
            "lora_rank": config.lora_rank, "lora_alpha": config.lora_alpha,
        })
    return model, report


def parameter_category(name: str, components: ModelComponents) -> str:
    if "trainable_tokens_delta" in name:
        if _matches_path(name, components.input_embedding):
            return "atlas_input_rows"
        if _matches_path(name, components.output_head):
            return "atlas_output_rows"
    if "lora_A" in name or "lora_B" in name:
        if _matches_path(name, components.language):
            return "language_lora"
        if _matches_path(name, components.vision):
            return "vision_lora"
        return "forbidden"
    if _matches_path(name, components.merger):
        return "merger"
    if _matches_path(name, components.vision):
        return "vision"
    return "forbidden"


def audit_trainable_scope(
    model: torch.nn.Module,
    components: ModelComponents,
    setup: TokenSetup,
    config: TrainConfig,
    output_path: Path,
) -> JsonDict:
    groups: dict[str, JsonDict] = {
        name: {"tensors": 0, "parameters": 0, "names": []}
        for name in (
            "vision",
            "merger",
            "atlas_input_rows",
            "atlas_output_rows",
            "language_lora",
            "vision_lora",
            "forbidden",
        )
    }
    parameters = []
    errors = []
    dtypes: dict[str, Counter[str]] = defaultdict(Counter)
    expected_delta = (154, components.hidden_size)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = parameter_category(name, components)
        groups[category]["tensors"] += 1
        groups[category]["parameters"] += parameter.numel()
        groups[category]["names"].append(name)
        dtypes[category][str(parameter.dtype)] += 1
        parameters.append(
            {"name": name, "category": category, "shape": list(parameter.shape)}
        )
        if category.startswith("atlas_") and tuple(parameter.shape) != expected_delta:
            errors.append(f"{name} must have shape {expected_delta}")
        if config.text_only and category == "language_lora":
            rank_axis = 0 if ".lora_A." in name else 1
            if parameter.ndim != 2 or parameter.shape[rank_axis] != config.lora_rank:
                errors.append(f"{name} must have configured LoRA rank {config.lora_rank}")
    if config.text_only:
        required_groups = ("atlas_input_rows", "atlas_output_rows", "language_lora")
    elif config.olora:
        required_groups = ("atlas_input_rows", "atlas_output_rows", "language_lora", "vision_lora")
    else:
        required_groups = ("vision", "merger", "atlas_input_rows", "atlas_output_rows")
    for required in required_groups:
        if groups[required]["parameters"] == 0:
            errors.append(f"required trainable group is empty: {required}")
    if config.text_only:
        for frozen in ("vision", "merger", "vision_lora"):
            if groups[frozen]["parameters"]:
                errors.append(f"{frozen} weights must stay frozen in text mode")
        visual = resolve_wrapped_module(model, components.vision)
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in visual.state_dict().values()):
            errors.append("dormant visual state must stay FP32 in text mode")
    elif config.olora:
        for frozen in ("vision", "merger"):
            if groups[frozen]["parameters"]:
                errors.append(f"{frozen} weights must stay frozen under the olora profile")
    elif groups["vision_lora"]["parameters"]:
        errors.append("vision LoRA parameters need the olora profile")
    for category in ("atlas_input_rows", "atlas_output_rows"):
        if groups[category]["tensors"] != 1:
            errors.append(f"exactly one {category} tensor is required")
    if config.language_lora != bool(groups["language_lora"]["parameters"]):
        errors.append("language LoRA parameters disagree with the selected profile")
    if groups["forbidden"]["parameters"]:
        errors.append("base language or unknown parameters are trainable")
    for category in ("vision", "merger", "vision_lora"):
        if set(dtypes[category]) - {"torch.float32"}:
            errors.append(f"{category} trainable parameters must hold fp32 master weights")
    report = {
        "schema": "catan_trl_trainable_scope/v2",
        "profile": config.profile,
        "input_mode": config.input_mode,
        "components": components.as_dict(),
        "semantic_tokens": setup.as_dict(),
        "groups": groups,
        "dtypes": {name: dict(counter) for name, counter in dtypes.items()},
        "initialization": getattr(model, "_catan_initialization", None),
        "parameters": parameters,
        "errors": errors,
    }
    if config.text_only:
        report["lora"] = {
            "rank": config.lora_rank, "alpha": config.lora_alpha,
            "scaling": config.lora_alpha / config.lora_rank, "use_rslora": False,
        }
    write_json_atomic(output_path, report)
    if errors:
        raise RuntimeError("invalid trainable scope: " + "; ".join(errors))
    return report


def build_optimizer(
    model: torch.nn.Module,
    components: ModelComponents,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    grouped: dict[str, list[torch.nn.Parameter]] = {
        "vision": [],
        "merger": [],
        "token_rows": [],
        "language_lora": [],
        "vision_lora": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = parameter_category(name, components)
        if config.text_only and category not in {"atlas_input_rows", "atlas_output_rows", "language_lora"}:
            raise RuntimeError(f"text optimizer cannot contain {category}: {name}")
        if category == "vision":
            grouped["vision"].append(parameter)
        elif category == "merger":
            grouped["merger"].append(parameter)
        elif category in {"atlas_input_rows", "atlas_output_rows"}:
            grouped["token_rows"].append(parameter)
        elif category == "language_lora":
            grouped["language_lora"].append(parameter)
        elif category == "vision_lora":
            grouped["vision_lora"].append(parameter)
        else:
            raise RuntimeError(f"cannot route trainable parameter to optimizer: {name}")
    if config.text_only:
        required = ("token_rows", "language_lora")
    elif config.olora:
        required = ("token_rows", "language_lora", "vision_lora")
    else:
        required = ("vision", "merger", "token_rows")
    if any(not grouped[name] for name in required):
        raise RuntimeError({name: len(values) for name, values in grouped.items()})
    if config.language_lora != bool(grouped["language_lora"]):
        raise RuntimeError("language LoRA optimizer group disagrees with the selected profile")
    optimizer_groups = [
        {
            "params": grouped["vision"],
            "lr": config.vision_learning_rate,
            "weight_decay": config.weight_decay,
            "catan_name": "vision",
        },
        {
            "params": grouped["merger"],
            "lr": config.merger_learning_rate,
            "weight_decay": config.weight_decay,
            "catan_name": "merger",
        },
        {
            "params": grouped["token_rows"],
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "catan_name": "token_rows",
        },
    ]
    optimizer_groups = [group for group in optimizer_groups if group["params"]]
    if grouped["vision_lora"]:
        optimizer_groups.append(
            {
                "params": grouped["vision_lora"],
                "lr": config.vision_lora_learning_rate,
                "weight_decay": config.weight_decay,
                "catan_name": "vision_lora",
            }
        )
    if grouped["language_lora"]:
        optimizer_groups.append(
            {
                "params": grouped["language_lora"],
                "lr": config.language_lora_learning_rate,
                "weight_decay": config.weight_decay,
                "catan_name": "language_lora",
            }
        )
    return torch.optim.AdamW(optimizer_groups)


def audit_optimizer_coverage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
) -> JsonDict:
    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    occurrences: dict[int, int] = {}
    groups = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            occurrences[parameter_id] = occurrences.get(parameter_id, 0) + 1
            names.append(trainable.get(parameter_id, f"<unknown:{parameter_id}>"))
        groups.append(
            {
                "name": group.get("catan_name"),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameters": names,
            }
        )
    missing = sorted(name for key, name in trainable.items() if key not in occurrences)
    duplicate = sorted(
        trainable.get(key, f"<unknown:{key}>")
        for key, count in occurrences.items()
        if count != 1
    )
    unknown = sorted(
        name for group in groups for name in group["parameters"] if name.startswith("<unknown:")
    )
    errors = []
    if missing:
        errors.append(f"missing trainable tensors: {missing[:8]}")
    if duplicate:
        errors.append(f"duplicate tensors: {duplicate[:8]}")
    if unknown:
        errors.append(f"unknown tensors: {unknown[:8]}")
    report = {
        "schema": "catan_trl_optimizer_coverage/v1",
        "trainable_tensors": len(trainable),
        "covered_tensors": len(occurrences),
        "groups": groups,
        "errors": errors,
    }
    write_json_atomic(output_path, report)
    if errors:
        raise RuntimeError("invalid optimizer coverage: " + "; ".join(errors))
    return report


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
        name for name in model.state_dict() if _matches_path(name, model._catan_components.vision)
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


def _trainer_class(config: TrainConfig, components: ModelComponents, setup: TokenSetup):
    TRAINING_ARGS_NAME = importlib.import_module("transformers.trainer").TRAINING_ARGS_NAME
    SFTTrainer = importlib.import_module("trl").SFTTrainer

    class CatanSFTTrainer(SFTTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            trainer_model = kwargs.get("model", args[0] if args else None)
            if trainer_model is None:
                raise TypeError("CatanSFTTrainer requires a model")
            if config.text_only:
                kwargs["data_collator"] = TextCompletionCollator(
                    native_tokenizer(kwargs["processing_class"]),
                    max_sequence_length=config.max_sequence_length,
                )
            with expose_trainable_tokens_head_to_chunked_nll(trainer_model):
                super().__init__(*args, **kwargs)
            self._catan_merge_size = None
            if not config.text_only:
                self.data_collator = SpatialTargetCollator(
                    self.data_collator,
                    setup,
                    target_mode=config.spatial_target_mode,
                )
                vision_module = resolve_wrapped_module(self.model, components.vision)
                self._catan_vision_capture = VisionPoolerCapture(vision_module)
                vision_config = getattr(getattr(self.model, "config", None), "vision_config", None)
                merge_size = getattr(vision_config, "spatial_merge_size", None)
                if merge_size is None:
                    image_processor = getattr(self.processing_class, "image_processor", None)
                    merge_size = getattr(image_processor, "merge_size", 2)
                self._catan_merge_size = int(merge_size)
                if self._catan_merge_size <= 0:
                    raise ValueError("vision spatial merge size must be positive")
            self._catan_hidden_capture = LanguageHiddenCapture(
                resolve_wrapped_module(self.model, components.language)
            )
            self._catan_trivial_token_ids = trivial_completion_token_ids(
                native_tokenizer(self.processing_class)
            )
            self._catan_metrics: dict[str, list[float]] = defaultdict(list)
            self._catan_basis_cache: dict[str, torch.Tensor] = {}

        def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: JsonDict,
            return_outputs: bool = False,
            num_items_in_batch: torch.Tensor | None = None,
        ) -> Any:
            if config.text_only:
                # TRL 1.12 prediction_step sets this flag; its compute_loss must
                # receive and pop it before forwarding model kwargs. Do not admit
                # arbitrary underscore-prefixed fields or drop the eval intent.
                if "_prediction_loss_only" in inputs and not isinstance(inputs["_prediction_loss_only"], bool):
                    raise ValueError("_prediction_loss_only must be a trainer-owned boolean")
                if set(inputs) - {"input_ids", "attention_mask", "labels", "_prediction_loss_only"}:
                    raise ValueError("text loss accepts only input_ids, attention_mask, and labels")
            else:
                token_ids = inputs.pop("spatial_target_token_ids")
                bboxes = inputs.pop("spatial_target_bboxes")
                target_mask = inputs.pop("spatial_target_mask")
                image_grid_thw = inputs.get("image_grid_thw")
                self._catan_vision_capture.output = None
            labels = inputs.get("labels")
            self._catan_hidden_capture.output = None
            nll_loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            unwrapped = self.accelerator.unwrap_model(model, keep_torch_compile=False)
            total_loss = nll_loss
            if not config.text_only:
                if image_grid_thw is None:
                    raise RuntimeError("Qwen processor did not return image_grid_thw")
                pooled = self._catan_vision_capture.take()
                token_embeddings = unwrapped.get_input_embeddings()(token_ids)
                patch_loss, patch_accuracy, target_count = spatial_patch_loss(
                    pooled,
                    image_grid_thw,
                    token_embeddings,
                    bboxes,
                    target_mask,
                    merge_size=self._catan_merge_size,
                    temperature=config.patch_temperature,
                )
                total_loss = nll_loss + config.patch_loss_weight * patch_loss
                self._catan_metrics["patch_loss"].append(float(patch_loss.detach()))
                self._catan_metrics["patch_top1_tolerant_accuracy"].append(float(patch_accuracy.detach()))
                self._catan_metrics["patch_target_count"].append(float(target_count))
            bases = getattr(unwrapped, "_catan_orthogonal_bases", None)
            if bases is not None and config.orthogonal_lambda > 0:
                orth, fraction, unprotected = orthogonal_penalty(unwrapped, bases, self._catan_basis_cache)
                total_loss = total_loss + config.orthogonal_lambda * orth
                self._catan_metrics["orth_loss"].append(float(orth.detach()))
                self._catan_metrics["orth_fraction"].append(fraction)
                self._catan_metrics["orth_unprotected_modules"].append(float(unprotected))
            if labels is not None:
                weight, bias = output_head_weight(unwrapped.get_base_model().get_output_embeddings())
                answer = answer_token_metrics(
                    self._catan_hidden_capture.take(),
                    labels,
                    weight,
                    bias,
                    self._catan_trivial_token_ids,
                )
                for name, value in answer.items():
                    self._catan_metrics[name].append(value)
            self._catan_metrics["nll_loss"].append(float(nll_loss.detach()))
            return (total_loss, outputs) if return_outputs else total_loss

        def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
            for name, values in self._catan_metrics.items():
                if values:
                    logs[name] = sum(values) / len(values)
            self._catan_metrics.clear()
            super().log(logs, *args, **kwargs)

        def _clip_grad_norm(self, model: torch.nn.Module) -> Any:
            # Record the pre-clip norm of every optimizer group so a dominant
            # group cannot hide behind the single clipped scalar Trainer logs.
            with torch.no_grad():
                for group in self.optimizer.param_groups:
                    grads = [p.grad.detach().float().norm() for p in group["params"] if p.grad is not None]
                    if grads:
                        norm = torch.norm(torch.stack(grads))
                        self._catan_metrics[f"grad_norm_{group['catan_name']}"].append(float(norm))
            return super()._clip_grad_norm(model)

        def create_optimizer(self, model: Any = None) -> torch.optim.Optimizer:
            if self.optimizer is None:
                target = self.model if model is None else model
                self.optimizer = build_optimizer(target, components, config)
                audit_optimizer_coverage(
                    target,
                    self.optimizer,
                    Path(self.args.output_dir) / OPTIMIZER_COVERAGE_FILE,
                )
            return self.optimizer

        def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
            target_dir = Path(output_dir or self.args.output_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            unwrapped.save_pretrained(
                target_dir,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            self.processing_class.save_pretrained(target_dir)
            if config.text_only:
                carry_processor_assets(config.initial_bundle, target_dir)
            torch.save(self.args, target_dir / TRAINING_ARGS_NAME)
            is_checkpoint = Path(self.args.output_dir).resolve() in target_dir.resolve().parents
            save_visual_state(
                unwrapped,
                components,
                target_dir,
                dtype=None if is_checkpoint or config.text_only else torch.bfloat16,
            )
            audit_trainable_scope(
                unwrapped,
                components,
                setup,
                config,
                target_dir / TRAINABLE_SCOPE_FILE,
            )
            write_json_atomic(target_dir / RUN_CONFIG_FILE, asdict(config))
            frozen_dir = getattr(unwrapped, "_catan_frozen_adapter_dir", None)
            if frozen_dir is not None:
                # A bundle from this profile only reproduces its model on top of the
                # merged parent, so the parent adapter travels inside every save.
                carried = target_dir / FROZEN_ADAPTER_DIR
                carried.mkdir(exist_ok=True)
                for source in frozen_adapter_files(Path(frozen_dir)):
                    shutil.copy2(source, carried / source.name)
                write_json_atomic(target_dir / FROZEN_BUNDLE_FILE, {"schema": "catan_trl_frozen_bundle_pointer/v1", "path": str(frozen_dir), "carried_adapter": str(carried), "orthogonal_lambda": config.orthogonal_lambda})
            if config.text_only:
                return
            write_json_atomic(
                target_dir / PATCH_METRICS_FILE,
                {
                    "schema": "catan_patch_localization_config/v1",
                    "loss_weight": config.patch_loss_weight,
                    "temperature": config.patch_temperature,
                    "target_mode": config.spatial_target_mode,
                    "spatial_merge_size": self._catan_merge_size,
                    "feature_space": "post_merger_lm_hidden",
                    "normalization": "soft_cross_entropy_div_log_patch_count",
                },
            )

        def _load_from_checkpoint(
            self,
            resume_from_checkpoint: str,
            model: torch.nn.Module | None = None,
        ) -> None:
            validate_resume_mode(resume_from_checkpoint, config.input_mode)
            super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            target = self.model if model is None else model
            if config.text_only:
                freeze_visual_weights(target, components)
            elif config.olora:
                freeze_visual_except_lora(target, components)
            else:
                promote_visual_master_weights(target, components)
            load_visual_state(target, resume_from_checkpoint)

    return CatanSFTTrainer


def _load_base_model_and_processor(
    config: TrainConfig, *, processor: Any = None,
) -> tuple[Any, torch.nn.Module]:
    transformers = importlib.import_module("transformers")
    if processor is None:
        if config.text_only:
            processor = load_checkpoint_text_tokenizer(
                config.initial_bundle, load_token_inventory(config.token_inventory)["tokens"],
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
) -> Any:
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
        load_checkpoint_text_tokenizer(final_dir, inventory["tokens"]) if config.text_only
        else transformers.AutoProcessor.from_pretrained(final_dir)
    )
    if config.text_only:
        processor_assets = processor_asset_hashes(final_dir)
        if processor_assets != processor_asset_hashes(config.initial_bundle):
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
    report = {
        "schema": "catan_trl_reload_validation/v1",
        "valid": True,
        "model_id": config.model_id,
        "input_mode": config.input_mode,
        "components": components.as_dict(),
        "semantic_tokens": setup.as_dict(),
        "visual_state": visual,
    }
    if config.text_only:
        report["processor_assets_sha256"] = processor_assets
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
    report = {
        "schema": "catan_trl_hub_publish/v1",
        "repo_id": config.hub_model_id,
        "commit_url": str(commit.commit_url),
        "oid": commit.oid,
    }
    write_json_atomic(final_dir / "hub_publish.json", report)
    return report


def run_training(config: TrainConfig, *, extra_callbacks: Sequence[Any] | None = None) -> JsonDict:
    config.validate()
    versions = assert_runtime_versions()
    inventory = load_token_inventory(config.token_inventory)
    processor = (
        load_checkpoint_text_tokenizer(config.initial_bundle, inventory["tokens"])
        if config.text_only else None
    )
    dataset, dataset_report = load_training_dataset(config, tokenizer=processor)
    eval_dataset = None
    eval_dataset_report = None
    if config.text_only and config.eval_jsonl is not None:
        eval_dataset, eval_dataset_report = load_text_dataset(
            config.eval_jsonl, tokenizer=processor,
            max_sequence_length=config.max_sequence_length, require_curriculum=False,
        )
    elif config.eval_jsonl is not None and config.eval_image_root is not None:
        eval_dataset, eval_dataset_report = load_vision_dataset(
            config.eval_jsonl,
            config.eval_image_root,
            require_curriculum=False,
        )
    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    final_dir = output_dir / "final"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / DATASET_REPORT_FILE,
        {"train": dataset_report, "eval": eval_dataset_report},
    )
    write_json_atomic(output_dir / RUN_CONFIG_FILE, asdict(config))

    processor, base_model = _load_base_model_and_processor(config, processor=processor)
    setup, components = prepare_semantic_tokens(processor, base_model, inventory)
    initial_bundle_report = None
    frozen_bundle_report = None
    if config.olora:
        model, frozen_bundle_report = load_frozen_bundle(base_model, setup, components, config)
        write_json_atomic(output_dir / FROZEN_BUNDLE_FILE, frozen_bundle_report)
    elif config.initial_bundle:
        model, initial_bundle_report = load_initial_bundle(
            base_model,
            setup,
            components,
            config,
        )
        write_json_atomic(output_dir / INITIAL_BUNDLE_FILE, initial_bundle_report)
    else:
        model = wrap_trainable_model(base_model, setup, components, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    audit_trainable_scope(
        model,
        components,
        setup,
        config,
        output_dir / TRAINABLE_SCOPE_FILE,
    )

    args = build_sft_config(
        config,
        checkpoints_dir,
        has_eval_dataset=eval_dataset is not None,
    )
    trainer_type = _trainer_class(config, components, setup)
    trainer = trainer_type(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )
    # Append after constructor-installed callbacks and the existing bundle save machinery.
    for callback in extra_callbacks or ():
        trainer.add_callback(callback)
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    # The periodic eval already ran on the last step whenever max_steps is a
    # multiple of eval_steps; only evaluate again when it did not.
    final_eval_metrics = None
    if eval_dataset is not None:
        last_eval = next(
            (row for row in reversed(trainer.state.log_history) if "eval_loss" in row),
            None,
        )
        if last_eval is not None and last_eval.get("step") == trainer.state.global_step:
            final_eval_metrics = {key: value for key, value in last_eval.items() if key != "step"}
        else:
            final_eval_metrics = trainer.evaluate()
    trainer.save_model(str(final_dir))
    trainer.save_state()
    _write_model_card(final_dir, config)
    metrics = dict(train_result.metrics)
    write_json_atomic(
        final_dir / "train_metrics.json",
        {"train": metrics, "eval": final_eval_metrics},
    )
    if initial_bundle_report is not None:
        write_json_atomic(final_dir / INITIAL_BUNDLE_FILE, initial_bundle_report)
    if frozen_bundle_report is not None:
        write_json_atomic(final_dir / FROZEN_BUNDLE_FILE, {**frozen_bundle_report, "carried_adapter": str(final_dir / FROZEN_ADAPTER_DIR)})

    del trainer, model, base_model, processor, dataset, eval_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reload_report = validate_saved_bundle(config, final_dir, inventory, setup)
    publish_report = publish_bundle(config, final_dir) if config.publish_to_hub else None
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "final_dir": str(final_dir),
        "versions": versions,
        "dataset": {"train": dataset_report, "eval": eval_dataset_report},
        "metrics": metrics,
        "eval_metrics": final_eval_metrics,
        "initial_bundle": initial_bundle_report,
        "frozen_bundle": frozen_bundle_report,
        "reload_validation": reload_report,
        "hub_publish": publish_report,
    }


def parse_args(argv: Iterable[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="vision")
    parser.add_argument("--max-sequence-length", type=int, help="Required text token budget; never truncates")
    parser.add_argument("--token-inventory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-jsonl")
    parser.add_argument("--eval-image-root")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE_VISION_TOKENS_LORA)
    parser.add_argument("--hub-model-id", default=HUB_MODEL_ID)
    parser.add_argument("--publish-to-hub", action="store_true")
    parser.add_argument("--allow-unstaged", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--initial-bundle")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--language-lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vision-learning-rate", type=float, default=5e-6)
    parser.add_argument("--merger-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--image-min-pixels", type=int, default=256 * 256)
    parser.add_argument("--image-max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--save-steps", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-loss-weight", type=float, default=0.0)
    parser.add_argument("--patch-temperature", type=float, default=0.07)
    parser.add_argument(
        "--spatial-target-mode",
        choices=SPATIAL_TARGET_MODES,
        default="correct",
    )
    parser.add_argument("--token-init", choices=TOKEN_INIT_MODES, default="mean_noise")
    parser.add_argument("--frozen-bundle", help="Finished bundle merged into the base as the frozen task (olora profile)")
    parser.add_argument("--visual-delta-factors", help="safetensors of lora_A/lora_B factors of the frozen task's visual delta; their lora_A rows are the protected vision subspaces")
    parser.add_argument("--orthogonal-lambda", type=float, default=0.5)
    parser.add_argument("--vision-lora-learning-rate", type=float, default=1e-4)
    args = parser.parse_args(argv)
    return TrainConfig(
        train_jsonl=args.train_jsonl,
        image_root=args.image_root,
        input_mode=args.input_mode,
        max_sequence_length=args.max_sequence_length,
        token_inventory=args.token_inventory,
        output_dir=args.output_dir,
        eval_jsonl=args.eval_jsonl,
        eval_image_root=args.eval_image_root,
        model_id=args.model_id,
        profile=args.profile,
        hub_model_id=args.hub_model_id,
        publish_to_hub=args.publish_to_hub,
        require_curriculum=not args.allow_unstaged,
        resume_from_checkpoint=args.resume_from_checkpoint,
        initial_bundle=args.initial_bundle,
        max_steps=None if args.max_steps == 0 else args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        language_lora_learning_rate=args.language_lora_learning_rate,
        vision_learning_rate=args.vision_learning_rate,
        merger_learning_rate=args.merger_learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        patch_loss_weight=args.patch_loss_weight,
        patch_temperature=args.patch_temperature,
        spatial_target_mode=args.spatial_target_mode,
        token_init=args.token_init,
        frozen_bundle=args.frozen_bundle,
        visual_delta_factors=args.visual_delta_factors,
        orthogonal_lambda=args.orthogonal_lambda,
        vision_lora_learning_rate=args.vision_lora_learning_rate,
    )


def main() -> int:
    result = run_training(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
