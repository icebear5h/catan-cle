"""LEGACY configuration and launch guards for native Qwen vision SFT.

The active contract is ``sft/scripts/train/train_trl_catan_vision.py``. The existing
Modal smoke launcher intentionally keeps the complete model path
frozen and quantized. This module defines a separate BF16 path that trains the
full vision tower and multimodal merger while keeping base language weights
frozen. It has no Modal dependency so launch plans can be tested locally.
"""

from __future__ import annotations

from ._base import DEFAULT_4B_MODEL_ID as DEFAULT_4B_MODEL_ID
from ._base import DEFAULT_27B_MODEL_ID as DEFAULT_27B_MODEL_ID
from ._base import H200_PROFILE as H200_PROFILE
from ._base import HARDWARE_PROFILES as HARDWARE_PROFILES
from ._base import L40S_PROFILE as L40S_PROFILE
from ._base import MAX_PROMPT_CHARACTERS as MAX_PROMPT_CHARACTERS
from ._base import MAX_SHORT_ANSWER_CHARACTERS as MAX_SHORT_ANSWER_CHARACTERS
from ._base import VISION_LANGUAGE_LORA as VISION_LANGUAGE_LORA
from ._base import VISION_ONLY as VISION_ONLY
from ._base import VISION_SFT_PROFILES as VISION_SFT_PROFILES
from ._base import Any as Any
from ._base import Path as Path
from ._base import asdict as asdict
from ._base import dataclass as dataclass
from ._base import hashlib as hashlib
from ._base import iter_jsonl as iter_jsonl
from ._base import json as json
from ._base import re as re
from ._base import recognition_token_inventory as recognition_token_inventory
from ._base import repository_relative_path as repository_relative_path
from ._base import resolve_dataset_asset as resolve_dataset_asset
from ._base import resolve_dataset_image as resolve_dataset_image
from ._command import build_vision_sft_command as build_vision_sft_command
from ._config import VisionSftConfig as VisionSftConfig
from ._config import default_run_name as default_run_name
from ._config import validate_hardware_profile as validate_hardware_profile
from ._fingerprint import _content_text as _content_text
from ._fingerprint import _sha256_file as _sha256_file
from ._fingerprint import _short_answer_pair as _short_answer_pair
from ._fingerprint import fingerprint_training_dataset as fingerprint_training_dataset
from ._fingerprint import launch_identity as launch_identity
from ._fingerprint import load_recognition_token_inventory as load_recognition_token_inventory

__all__ = [
    "Any",
    "DEFAULT_27B_MODEL_ID",
    "DEFAULT_4B_MODEL_ID",
    "H200_PROFILE",
    "HARDWARE_PROFILES",
    "L40S_PROFILE",
    "MAX_PROMPT_CHARACTERS",
    "MAX_SHORT_ANSWER_CHARACTERS",
    "Path",
    "VISION_LANGUAGE_LORA",
    "VISION_ONLY",
    "VISION_SFT_PROFILES",
    "VisionSftConfig",
    "asdict",
    "build_vision_sft_command",
    "dataclass",
    "default_run_name",
    "fingerprint_training_dataset",
    "hashlib",
    "iter_jsonl",
    "json",
    "launch_identity",
    "load_recognition_token_inventory",
    "re",
    "recognition_token_inventory",
    "repository_relative_path",
    "resolve_dataset_asset",
    "resolve_dataset_image",
    "validate_hardware_profile",
]
