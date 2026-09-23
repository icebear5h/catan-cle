"""Pinned parent run, checkpoint steps and Modal options."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from dataclasses import asdict as asdict
from dataclasses import replace as replace
from pathlib import Path as Path
from typing import TypedDict

import modal
import torch
from transformers import AddedToken as AddedToken
from transformers import AutoTokenizer as AutoTokenizer

from sft.launchers.spatial.modal_spatial_continuation import CPU_OPTIONS as CPU_OPTIONS
from sft.launchers.spatial.modal_spatial_continuation import DEFAULT_INPUTS as DEFAULT_INPUTS
from sft.launchers.spatial.modal_spatial_continuation import GPU_OPTIONS as GPU_OPTIONS
from sft.launchers.spatial.modal_spatial_continuation import LOCAL_RUN_ROOT as LOCAL_RUN_ROOT
from sft.launchers.spatial.modal_spatial_continuation import NEW_PANEL_TASKS as NEW_PANEL_TASKS
from sft.launchers.spatial.modal_spatial_continuation import OLD_PANELS as OLD_PANELS
from sft.launchers.spatial.modal_spatial_continuation import PANEL_BUDGETS as PANEL_BUDGETS
from sft.launchers.spatial.modal_spatial_continuation import PREFLIGHT_TIMEOUT as PREFLIGHT_TIMEOUT
from sft.launchers.spatial.modal_spatial_continuation import RUN_ROOT as RUN_ROOT
from sft.launchers.spatial.modal_spatial_continuation import SNAPSHOT as SNAPSHOT
from sft.launchers.spatial.modal_spatial_continuation import STARTUP_TIMEOUT as STARTUP_TIMEOUT
from sft.launchers.spatial.modal_spatial_continuation import WAIT_GRACE as WAIT_GRACE
from sft.launchers.spatial.modal_spatial_continuation import check_identity as check_identity
from sft.launchers.spatial.modal_spatial_continuation import checkpoint_audit as checkpoint_audit
from sft.launchers.spatial.modal_spatial_continuation import compare_panels as compare_panels
from sft.launchers.spatial.modal_spatial_continuation import completion_audit as completion_audit
from sft.launchers.spatial.modal_spatial_continuation import (
    continuation_image as continuation_image,
)
from sft.launchers.spatial.modal_spatial_continuation import dataset_identity as dataset_identity
from sft.launchers.spatial.modal_spatial_continuation import digest as digest
from sft.launchers.spatial.modal_spatial_continuation import (
    evaluation_conditions as evaluation_conditions,
)
from sft.launchers.spatial.modal_spatial_continuation import evaluator as evaluator
from sft.launchers.spatial.modal_spatial_continuation import matched_baseline as matched_baseline
from sft.launchers.spatial.modal_spatial_continuation import now as now
from sft.launchers.spatial.modal_spatial_continuation import panel_args as panel_args
from sft.launchers.spatial.modal_spatial_continuation import read_json as read_json
from sft.launchers.spatial.modal_spatial_continuation import reload_volumes as reload_volumes
from sft.launchers.spatial.modal_spatial_continuation import sft_runs as sft_runs
from sft.launchers.spatial.modal_spatial_continuation import source_hashes as source_hashes
from sft.launchers.spatial.modal_spatial_continuation import validate_mixture as validate_mixture
from sft.launchers.spatial.modal_spatial_continuation import validate_run_name as validate_run_name
from sft.launchers.spatial.modal_spatial_continuation._base import ContinuationGpuOptions
from sft.scripts.train.train_trl_catan_vision import TrainConfig as TrainConfig
from sft.scripts.train.train_trl_catan_vision import (
    inspect_jsonl_contract as inspect_jsonl_contract,
)
from sft.scripts.train.train_trl_catan_vision import iter_jsonl as iter_jsonl
from sft.scripts.train.train_trl_catan_vision import load_token_inventory as load_token_inventory
from sft.scripts.train.train_trl_catan_vision import (
    normalize_training_config as normalize_training_config,
)
from sft.scripts.train.train_trl_catan_vision import run_training as run_training
from sft.scripts.train.train_trl_catan_vision import sha256_file as sha256_file
from sft.scripts.train.train_trl_catan_vision import write_json_atomic as write_json_atomic

__all__ = [
    "AddedToken",
    "AutoTokenizer",
    "CHECKPOINT_STEPS",
    "CONFIG_CHANGES",
    "COORDINATOR_TIMEOUT",
    "CPU_OPTIONS",
    "DEFAULT_INPUTS",
    "DEFAULT_RUN_NAME",
    "GPU_OPTIONS",
    "LOCAL_RUN_ROOT",
    "NEW_PANEL_TASKS",
    "OLD_PANELS",
    "PANEL_BUDGETS",
    "PARENT_CHECKPOINT",
    "PARENT_RESULT",
    "PARENT_RUN",
    "POST_GPU_OPTIONS",
    "POST_TIMEOUT",
    "PREFLIGHT_TIMEOUT",
    "Path",
    "RUN_ROOT",
    "SNAPSHOT",
    "STARTUP_TIMEOUT",
    "TRAIN_GPU_OPTIONS",
    "TRAIN_TIMEOUT",
    "TrainConfig",
    "WAIT_GRACE",
    "app",
    "argparse",
    "asdict",
    "check_identity",
    "checkpoint_audit",
    "compare_panels",
    "completion_audit",
    "continuation_image",
    "dataset_identity",
    "digest",
    "evaluation_conditions",
    "evaluator",
    "inspect_jsonl_contract",
    "iter_jsonl",
    "json",
    "load_token_inventory",
    "matched_baseline",
    "math",
    "modal",
    "normalize_training_config",
    "now",
    "os",
    "panel_args",
    "read_json",
    "reload_volumes",
    "replace",
    "run_training",
    "sft_runs",
    "sha256_file",
    "source_hashes",
    "torch",
    "uuid",
    "validate_mixture",
    "validate_run_name",
    "write_json_atomic",
]

PARENT_RUN = "spatial-continuation-20260909-r01"

PARENT_RESULT = LOCAL_RUN_ROOT / PARENT_RUN / "result.json"

PARENT_CHECKPOINT = str(RUN_ROOT / PARENT_RUN / "checkpoints/checkpoint-128")

DEFAULT_RUN_NAME = "spatial-continuation-20260912-r01"

TRAIN_TIMEOUT = 7200

POST_TIMEOUT = 3600

COORDINATOR_TIMEOUT = 14400

CHECKPOINT_STEPS = tuple(range(32, 257, 32))

class ConfigChanges(TypedDict):
    """The `TrainConfig` overrides applied to the parent, checkable through `**`."""

    max_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    resume_from_checkpoint: str | None


CONFIG_CHANGES: ConfigChanges = dict(max_steps=256, save_steps=32, eval_steps=32, save_total_limit=8,
                                     resume_from_checkpoint=None)

app = modal.App("catan-spatial-extension")

TRAIN_GPU_OPTIONS: ContinuationGpuOptions = {**GPU_OPTIONS, "image": continuation_image, "timeout": TRAIN_TIMEOUT}

POST_GPU_OPTIONS: ContinuationGpuOptions = {**GPU_OPTIONS, "image": continuation_image, "timeout": POST_TIMEOUT}
