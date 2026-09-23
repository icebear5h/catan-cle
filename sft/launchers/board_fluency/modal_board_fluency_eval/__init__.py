"""One bounded text-only review or 400-row matched coordinate comparison.

CLI: MODAL_PROFILE=tetracorp .venv/bin/python -m modal run -m sft.launchers.board_fluency.modal_board_fluency_eval \
    --hf-repo "$HF_REPO" --hf-revision "$HF_SHA" --run-name "$RUN_NAME" --prepare-only
Omit --prepare-only for CPU preparation followed by exactly one H200 call.
Alternatively, use --adapter-dir /runs/catan-vision-sft/.../checkpoints/checkpoint-128
instead of the HF adapter arguments. CATAN_HF_SECRET_NAME selects the HF secret.
Every invocation needs a fresh run name; preparation reuses the shared HF cache.
"""

from __future__ import annotations

__all__ = [
    "AddedToken",
    "Any",
    "AutoConfig",
    "BATCH",
    "CACHE",
    "COMPARISON_GPU_DEADLINE",
    "CONTEXT",
    "Counter",
    "DEFAULT_INVENTORY",
    "DEFAULT_REVIEW",
    "GPU_DEADLINE",
    "HF_SECRET_NAME",
    "HfApi",
    "INFERENCE_SIDECARS",
    "LoraConfig",
    "MODEL_ID",
    "MODEL_REVISION",
    "NEW_TOKENS",
    "ORIGINAL_INVENTORY",
    "Path",
    "PurePosixPath",
    "REQUIRED_BUNDLE",
    "ROWS",
    "RUN_CONFIG_FILE",
    "TRAINABLE_SCOPE_FILE",
    "VISUAL_STATE_FILE",
    "VOLUMES",
    "reload_volumes",
    "ModalCommon",
    "ModalGpu",
    "ModalStage",
    "_eval_bounded",
    "_message_pair",
    "adapter_preflight",
    "annotations",
    "app",
    "assert_runtime_versions",
    "base_preflight",
    "check_target_module_exists",
    "coordinate_comparison",
    "datetime",
    "digest",
    "discover_components",
    "download_file",
    "encode_text_pair",
    "error_record",
    "eval_command",
    "eval_coordinate_h200",
    "eval_h200",
    "eval_image",
    "evaluator",
    "expected_adapter_shapes",
    "file_manifest",
    "hashlib",
    "hf_cache",
    "hf_secret",
    "inspect_inputs",
    "io",
    "json",
    "language_linear_targets",
    "load_checkpoint_text_tokenizer",
    "load_token_inventory",
    "main",
    "modal",
    "now",
    "open_tensors",
    "os",
    "prepare_cpu",
    "re",
    "read_json",
    "safe_open",
    "sft_data",
    "sft_runs",
    "sha256_file",
    "snapshot",
    "snapshot_download",
    "standard_lora_rank",
    "subprocess",
    "sys",
    "tensor_headers",
    "time",
    "timezone",
    "torch",
    "training_base_image",
    "uuid",
    "validate_cli",
    "verify_inputs",
    "verify_outputs",
    "volume_bundle",
    "write_json_atomic",
]

import hashlib as hashlib
import io as io
import json as json
import os as os
import re as re
import subprocess as subprocess
import sys as sys
import time as time
import uuid as uuid
from collections import Counter as Counter
from datetime import datetime as datetime
from datetime import timezone as timezone
from pathlib import Path as Path
from pathlib import PurePosixPath as PurePosixPath
from typing import Any as Any

import modal as modal

from sft.board import coordinate_comparison as coordinate_comparison
from sft.launchers.modal_catan_vision_sft import HF_SECRET_NAME as HF_SECRET_NAME
from sft.launchers.modal_catan_vision_sft import hf_cache as hf_cache
from sft.launchers.modal_catan_vision_sft import sft_data as sft_data
from sft.launchers.modal_catan_vision_sft import sft_runs as sft_runs
from sft.launchers.modal_catan_vision_sft import training_base_image as training_base_image
from sft.lora_expansion import expected_adapter_shapes as expected_adapter_shapes
from sft.lora_expansion import standard_lora_rank as standard_lora_rank
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import RUN_CONFIG_FILE as RUN_CONFIG_FILE
from sft.scripts.train.train_trl_catan_vision import TRAINABLE_SCOPE_FILE as TRAINABLE_SCOPE_FILE
from sft.scripts.train.train_trl_catan_vision import VISUAL_STATE_FILE as VISUAL_STATE_FILE
from sft.scripts.train.train_trl_catan_vision import _message_pair as _message_pair
from sft.scripts.train.train_trl_catan_vision import (
    assert_runtime_versions as assert_runtime_versions,
)
from sft.scripts.train.train_trl_catan_vision import discover_components as discover_components
from sft.scripts.train.train_trl_catan_vision import encode_text_pair as encode_text_pair
from sft.scripts.train.train_trl_catan_vision import (
    language_linear_targets as language_linear_targets,
)
from sft.scripts.train.train_trl_catan_vision import (
    load_checkpoint_text_tokenizer as load_checkpoint_text_tokenizer,
)
from sft.scripts.train.train_trl_catan_vision import load_token_inventory as load_token_inventory
from sft.scripts.train.train_trl_catan_vision import sha256_file as sha256_file
from sft.scripts.train.train_trl_catan_vision import write_json_atomic as write_json_atomic

from ._cli import download_file as download_file
from ._cli import main as main
from ._config import BATCH as BATCH
from ._config import CACHE as CACHE
from ._config import COMPARISON_GPU_DEADLINE as COMPARISON_GPU_DEADLINE
from ._config import CONTEXT as CONTEXT
from ._config import DEFAULT_INVENTORY as DEFAULT_INVENTORY
from ._config import DEFAULT_REVIEW as DEFAULT_REVIEW
from ._config import GPU_DEADLINE as GPU_DEADLINE
from ._config import INFERENCE_SIDECARS as INFERENCE_SIDECARS
from ._config import MODEL_ID as MODEL_ID
from ._config import MODEL_REVISION as MODEL_REVISION
from ._config import NEW_TOKENS as NEW_TOKENS
from ._config import ORIGINAL_INVENTORY as ORIGINAL_INVENTORY
from ._config import REQUIRED_BUNDLE as REQUIRED_BUNDLE
from ._config import ROWS as ROWS
from ._config import VOLUMES as VOLUMES
from ._config import ModalCommon as ModalCommon
from ._config import ModalGpu as ModalGpu
from ._config import ModalStage as ModalStage
from ._config import app as app
from ._config import eval_image as eval_image
from ._config import hf_secret as hf_secret
from ._config import reload_volumes as reload_volumes
from ._remote import _eval_bounded as _eval_bounded
from ._remote import eval_command as eval_command
from ._remote import eval_coordinate_h200 as eval_coordinate_h200
from ._remote import eval_h200 as eval_h200
from ._remote import prepare_cpu as prepare_cpu
from ._remote import verify_outputs as verify_outputs
from ._snapshots import AddedToken as AddedToken
from ._snapshots import AutoConfig as AutoConfig
from ._snapshots import HfApi as HfApi
from ._snapshots import LoraConfig as LoraConfig
from ._snapshots import adapter_preflight as adapter_preflight
from ._snapshots import base_preflight as base_preflight
from ._snapshots import check_target_module_exists as check_target_module_exists
from ._snapshots import file_manifest as file_manifest
from ._snapshots import open_tensors as open_tensors
from ._snapshots import safe_open as safe_open
from ._snapshots import snapshot as snapshot
from ._snapshots import snapshot_download as snapshot_download
from ._snapshots import tensor_headers as tensor_headers
from ._snapshots import torch as torch
from ._snapshots import volume_bundle as volume_bundle
from ._validation import digest as digest
from ._validation import error_record as error_record
from ._validation import inspect_inputs as inspect_inputs
from ._validation import now as now
from ._validation import read_json as read_json
from ._validation import validate_cli as validate_cli
from ._validation import verify_inputs as verify_inputs
