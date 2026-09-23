"""Profile names and answer/prompt budgets for vision SFT runs."""

from __future__ import annotations

import hashlib as hashlib
import json as json
import re as re
from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from pathlib import Path as Path
from typing import Any as Any

from evals.catan_board_bench.tokens import (
    recognition_token_inventory as recognition_token_inventory,
)
from sft.paths import repository_relative_path as repository_relative_path
from sft.paths import resolve_dataset_asset as resolve_dataset_asset
from sft.paths import resolve_dataset_image as resolve_dataset_image
from sft.scripts.builders.convert_to_qwen_series_sft import iter_jsonl as iter_jsonl

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
