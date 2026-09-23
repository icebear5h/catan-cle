"""Opt-in 256-additional-update extension of the completed spatial continuation.

Dry run: python -B -m sft.launchers.spatial.modal_spatial_extension --run-name spatial-continuation-20260912-r01
Add --execute to reserve a new run and detach its bounded CPU coordinator.
The unchanged, already-uploaded parent datasets and all six saved post panels
are pinned by local receipts and checked on CPU before the only training call.
"""

from __future__ import annotations

from ._base import CHECKPOINT_STEPS as CHECKPOINT_STEPS
from ._base import CONFIG_CHANGES as CONFIG_CHANGES
from ._base import COORDINATOR_TIMEOUT as COORDINATOR_TIMEOUT
from ._base import CPU_OPTIONS as CPU_OPTIONS
from ._base import DEFAULT_INPUTS as DEFAULT_INPUTS
from ._base import DEFAULT_RUN_NAME as DEFAULT_RUN_NAME
from ._base import GPU_OPTIONS as GPU_OPTIONS
from ._base import LOCAL_RUN_ROOT as LOCAL_RUN_ROOT
from ._base import NEW_PANEL_TASKS as NEW_PANEL_TASKS
from ._base import OLD_PANELS as OLD_PANELS
from ._base import PANEL_BUDGETS as PANEL_BUDGETS
from ._base import PARENT_CHECKPOINT as PARENT_CHECKPOINT
from ._base import PARENT_RESULT as PARENT_RESULT
from ._base import PARENT_RUN as PARENT_RUN
from ._base import POST_GPU_OPTIONS as POST_GPU_OPTIONS
from ._base import POST_TIMEOUT as POST_TIMEOUT
from ._base import PREFLIGHT_TIMEOUT as PREFLIGHT_TIMEOUT
from ._base import RUN_ROOT as RUN_ROOT
from ._base import SNAPSHOT as SNAPSHOT
from ._base import STARTUP_TIMEOUT as STARTUP_TIMEOUT
from ._base import TRAIN_GPU_OPTIONS as TRAIN_GPU_OPTIONS
from ._base import TRAIN_TIMEOUT as TRAIN_TIMEOUT
from ._base import WAIT_GRACE as WAIT_GRACE
from ._base import AddedToken as AddedToken
from ._base import AutoTokenizer as AutoTokenizer
from ._base import Path as Path
from ._base import TrainConfig as TrainConfig
from ._base import app as app
from ._base import argparse as argparse
from ._base import asdict as asdict
from ._base import check_identity as check_identity
from ._base import checkpoint_audit as checkpoint_audit
from ._base import compare_panels as compare_panels
from ._base import completion_audit as completion_audit
from ._base import continuation_image as continuation_image
from ._base import dataset_identity as dataset_identity
from ._base import digest as digest
from ._base import evaluation_conditions as evaluation_conditions
from ._base import evaluator as evaluator
from ._base import inspect_jsonl_contract as inspect_jsonl_contract
from ._base import iter_jsonl as iter_jsonl
from ._base import json as json
from ._base import load_token_inventory as load_token_inventory
from ._base import matched_baseline as matched_baseline
from ._base import math as math
from ._base import modal as modal
from ._base import normalize_training_config as normalize_training_config
from ._base import now as now
from ._base import os as os
from ._base import panel_args as panel_args
from ._base import read_json as read_json
from ._base import reload_volumes as reload_volumes
from ._base import replace as replace
from ._base import run_training as run_training
from ._base import sft_runs as sft_runs
from ._base import sha256_file as sha256_file
from ._base import source_hashes as source_hashes
from ._base import torch as torch
from ._base import uuid as uuid
from ._base import validate_mixture as validate_mixture
from ._base import validate_run_name as validate_run_name
from ._base import write_json_atomic as write_json_atomic
from ._coordinate import coordinate as coordinate
from ._coordinate import reserve as reserve
from ._launch import launch as launch
from ._launch import main as main
from ._plan import audit_inputs as audit_inputs
from ._plan import audit_parent as audit_parent
from ._plan import build_plan as build_plan
from ._plan import validate_config as validate_config
from ._plan import verify_files as verify_files
from ._plan import verify_runtime as verify_runtime
from ._post import evaluate_bounded as evaluate_bounded
from ._preflight import extension_preflight as extension_preflight
from ._training import train_bounded as train_bounded
from ._training import training_history as training_history

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
    "audit_inputs",
    "audit_parent",
    "build_plan",
    "check_identity",
    "checkpoint_audit",
    "compare_panels",
    "completion_audit",
    "continuation_image",
    "coordinate",
    "dataset_identity",
    "digest",
    "evaluate_bounded",
    "evaluation_conditions",
    "evaluator",
    "extension_preflight",
    "inspect_jsonl_contract",
    "iter_jsonl",
    "json",
    "launch",
    "load_token_inventory",
    "main",
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
    "reserve",
    "run_training",
    "sft_runs",
    "sha256_file",
    "source_hashes",
    "torch",
    "train_bounded",
    "training_history",
    "uuid",
    "validate_config",
    "validate_mixture",
    "validate_run_name",
    "verify_files",
    "verify_runtime",
    "write_json_atomic",
]
