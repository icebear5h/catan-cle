"""Pinned paths, budgets and Modal options for the continuation run."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import uuid
from collections import Counter as Counter
from dataclasses import asdict as asdict
from dataclasses import replace as replace
from datetime import datetime as datetime
from datetime import timezone as timezone
from pathlib import Path as Path
from pathlib import PurePosixPath
from typing import TypedDict

import modal
import torch
from safetensors import safe_open as safe_open
from transformers import AddedToken as AddedToken
from transformers import AutoTokenizer as AutoTokenizer

from sft.launchers.full_board.modal_full_board_pilot import GPU_OPTIONS as GPU_OPTIONS
from sft.launchers.full_board.modal_full_board_pilot import SNAPSHOT as SNAPSHOT
from sft.launchers.full_board.modal_full_board_pilot import VOLUMES as VOLUMES
from sft.launchers.full_board.modal_full_board_pilot import ModalGpuOptions as ModalGpuOptions
from sft.launchers.full_board.modal_full_board_pilot import app as app
from sft.launchers.full_board.modal_full_board_pilot import train as pilot_train
from sft.launchers.modal_catan_vision_sft import hf_cache as hf_cache
from sft.launchers.modal_catan_vision_sft import sft_data as sft_data
from sft.launchers.modal_catan_vision_sft import sft_runs as sft_runs
from sft.launchers.modal_catan_vision_sft import training_image as training_image
from sft.launchers.modal_catan_vision_sft import (
    upload_training_bundle as upload_training_bundle,
)
from sft.launchers.qwen_series._eval_support import upload_eval_jsonl as upload_eval_jsonl
from sft.paths import PROJECT_ROOT as PROJECT_ROOT
from sft.paths import resolve_dataset_asset as resolve_dataset_asset
from sft.paths import resolve_dataset_image as resolve_dataset_image
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import TrainConfig as TrainConfig
from sft.scripts.train.train_trl_catan_vision import (
    inspect_jsonl_contract as inspect_jsonl_contract,
)
from sft.scripts.train.train_trl_catan_vision import iter_jsonl as iter_jsonl
from sft.scripts.train.train_trl_catan_vision import load_token_inventory as load_token_inventory
from sft.scripts.train.train_trl_catan_vision import (
    normalize_training_config as normalize_training_config,
)
from sft.scripts.train.train_trl_catan_vision import sha256_file as sha256_file
from sft.scripts.train.train_trl_catan_vision import write_json_atomic as write_json_atomic

__all__ = [
    "AddedToken",
    "AutoTokenizer",
    "BOARD_RECEIPT",
    "CONTINUATION_GPU_OPTIONS",
    "COORDINATOR_TIMEOUT",
    "CPU_OPTIONS",
    "Counter",
    "DEFAULT_INPUTS",
    "FAMILY_STEPS",
    "FIXED_CONFIG",
    "GPU_OPTIONS",
    "GPU_TIMEOUT",
    "LOCAL_RUN_ROOT",
    "NEW_LABELS",
    "NEW_PANEL_TASKS",
    "OLD_PANELS",
    "PANEL_BUDGETS",
    "PARENT_CHECKPOINT",
    "PARENT_RESULT",
    "PREFLIGHT_TIMEOUT",
    "PROJECT_ROOT",
    "Path",
    "RUN_ROOT",
    "SNAPSHOT",
    "SPATIAL_RECEIPT",
    "STARTUP_TIMEOUT",
    "TrainConfig",
    "VISUAL_SHA256",
    "VOLUMES",
    "WAIT_GRACE",
    "app",
    "argparse",
    "asdict",
    "continuation_image",
    "datetime",
    "evaluator",
    "hashlib",
    "hf_cache",
    "inspect",
    "inspect_jsonl_contract",
    "iter_jsonl",
    "json",
    "load_token_inventory",
    "modal",
    "normalize_training_config",
    "os",
    "pilot_train",
    "replace",
    "resolve_dataset_asset",
    "resolve_dataset_image",
    "safe_open",
    "sft_data",
    "sft_runs",
    "sha256_file",
    "timezone",
    "torch",
    "training_image",
    "upload_eval_jsonl",
    "upload_training_bundle",
    "uuid",
    "write_json_atomic",
]

RUN_ROOT = Path("/runs/catan-vision-sft")

LOCAL_RUN_ROOT = PROJECT_ROOT / "artifacts/runs/sft"

DEFAULT_INPUTS = PROJECT_ROOT / "artifacts/generated/board_recognition/spatial_continuation_v1/dataset_inputs.json"

PARENT_RESULT = LOCAL_RUN_ROOT / "full-board-new-layouts-20260907/result.json"

PARENT_CHECKPOINT = str(RUN_ROOT / "full-board-new-layouts-20260907/checkpoints/checkpoint-128")

VISUAL_SHA256 = "3535adfa86dbc4675f612f98995222f2f15619d151b5aff2f7537eab147b7183"

SPATIAL_RECEIPT = LOCAL_RUN_ROOT / "full-board-new-layouts-ck128-spatial-answer-only-v1-20260908-r01/summary.json"

BOARD_RECEIPT = LOCAL_RUN_ROOT / "full-board-new-layouts-ck128-spatial-choice-order-v1-20260908-r01/prior_full_board_summary.json"

OLD_PANELS = {
    "spatial": {
        "eval_jsonl": str(PROJECT_ROOT / "artifacts/generated/board_recognition/replay_v1/evals/spatial_choice_order_answer_only_validation_v1.jsonl"),
        "image_root": str(PROJECT_ROOT / "artifacts/generated/board_recognition/replay_v1/images"),
        "sha256": "53c4330d955d369833ca6024307b7b725a4cc46117fe6c1efc6170e7e7486786",
    },
    "fullboard": {
        "eval_jsonl": str(LOCAL_RUN_ROOT / "full-board-epoch2-20260907/validation64.jsonl"),
        "image_root": str(PROJECT_ROOT / "artifacts/generated/board_recognition/full_board_diverse_v1/full_board_readout_v1/images"),
        "sha256": "902d2833b9fe705888e834822488593c14c1c08e9e4c9939464d1501655cbc8e",
    },
}

PANEL_BUDGETS = {"spatial": (120, 48, 16), "fullboard": (64, 4, 1280),
                 "node_tiles": (54, 16, 128), "paths": (64, 16, 128),
                 "local": (64, 16, 128), "production": (64, 16, 128)}

FAMILY_STEPS = dict.fromkeys(("directions", "adjacency_connectivity", "node_tiles", "shortest_node_path",
                            "local_node_tiles", "dice_production"), 16) | {"full_board_readout": 32}

NEW_PANEL_TASKS = {"node_tiles": "node_tiles", "paths": "shortest_node_path",
                   "local": "local_node_tiles", "production": "dice_production"}

NEW_LABELS = tuple(NEW_PANEL_TASKS)

GPU_TIMEOUT = 3600

STARTUP_TIMEOUT = 300

PREFLIGHT_TIMEOUT = 1200

WAIT_GRACE = 60

COORDINATOR_TIMEOUT = 14400

continuation_image = training_image.add_local_python_source("data_pipeline")


class ContinuationGpuOptions(ModalGpuOptions):
    """The pilot's GPU options with this launcher's image and per-stage timeout."""

    timeout: int


class ModalCpuOptions(TypedDict):
    """`modal.App.function` CPU options, kept checkable through `**` unpacking."""

    image: modal.Image
    volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
    retries: int
    startup_timeout: int
    max_containers: int
    scaledown_window: int


CONTINUATION_GPU_OPTIONS: ContinuationGpuOptions = {
    **GPU_OPTIONS, "image": continuation_image, "timeout": GPU_TIMEOUT}

CPU_OPTIONS: ModalCpuOptions = dict(image=continuation_image, volumes=VOLUMES, retries=0,
                                    startup_timeout=STARTUP_TIMEOUT, max_containers=1,
                                    scaledown_window=2)

class FixedConfig(TypedDict):
    """The approved `TrainConfig` overrides, checkable through `**` unpacking."""

    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    require_curriculum: bool
    token_init: str
    resume_from_checkpoint: str | None
    publish_to_hub: bool
    profile: str
    frozen_bundle: str | None
    visual_delta_factors: str | None
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    language_lora_learning_rate: float
    vision_learning_rate: float
    merger_learning_rate: float
    warmup_ratio: float
    patch_loss_weight: float
    spatial_target_mode: str
    model_id: str


FIXED_CONFIG: FixedConfig = dict(max_steps=128, per_device_train_batch_size=4, gradient_accumulation_steps=2,
                    save_steps=32, eval_steps=32, save_total_limit=4, require_curriculum=False,
                    token_init="keep", resume_from_checkpoint=None, publish_to_hub=False,
                    profile="vision_tokens_lora", frozen_bundle=None, visual_delta_factors=None,
                    lora_rank=8, lora_alpha=16, lora_dropout=0.05, learning_rate=5e-4,
                    language_lora_learning_rate=1e-4, vision_learning_rate=5e-6,
                    merger_learning_rate=5e-5, warmup_ratio=0.1, patch_loss_weight=0.0,
                    spatial_target_mode="correct", model_id=SNAPSHOT)
