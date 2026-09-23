"""Shared constants, aliases, and JSON/hash primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory
from sft.json_types import JsonDict as JsonDict
from sft.json_types import JsonLike, as_dict, as_list, as_str, loads_json

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


def write_json_atomic(path: Path, payload: Mapping[str, JsonLike]) -> None:
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
                yield line_number, as_dict(loads_json(line))


def load_token_inventory(path: str | Path) -> JsonDict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = as_dict(loads_json(resolved.read_text()))
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


def inventory_tokens(inventory: JsonDict) -> list[str]:
    """The ordered atlas tokens of a semantic token inventory."""
    return [as_str(token) for token in as_list(inventory["tokens"])]
