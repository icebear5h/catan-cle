"""One opt-in, bounded spatial continuation. Importing/dry-running never contacts Modal.

Run ``python -m sft.launchers.spatial.modal_spatial_continuation --help`` locally. Only --execute
reserves remote paths, uploads inputs, and spawns the detached CPU coordinator.
Saved original-image generations are independently rescored before training;
there are no new blank controls, probes, sweeps, or conditional training stages.
"""

from __future__ import annotations

from ._audits import checkpoint_audit as checkpoint_audit
from ._audits import completion_audit as completion_audit
from ._audits import matched_baseline as matched_baseline
from ._base import BOARD_RECEIPT as BOARD_RECEIPT
from ._base import CONTINUATION_GPU_OPTIONS as CONTINUATION_GPU_OPTIONS
from ._base import COORDINATOR_TIMEOUT as COORDINATOR_TIMEOUT
from ._base import CPU_OPTIONS as CPU_OPTIONS
from ._base import DEFAULT_INPUTS as DEFAULT_INPUTS
from ._base import FAMILY_STEPS as FAMILY_STEPS
from ._base import FIXED_CONFIG as FIXED_CONFIG
from ._base import GPU_OPTIONS as GPU_OPTIONS
from ._base import GPU_TIMEOUT as GPU_TIMEOUT
from ._base import LOCAL_RUN_ROOT as LOCAL_RUN_ROOT
from ._base import NEW_LABELS as NEW_LABELS
from ._base import NEW_PANEL_TASKS as NEW_PANEL_TASKS
from ._base import OLD_PANELS as OLD_PANELS
from ._base import PANEL_BUDGETS as PANEL_BUDGETS
from ._base import PARENT_CHECKPOINT as PARENT_CHECKPOINT
from ._base import PARENT_RESULT as PARENT_RESULT
from ._base import PREFLIGHT_TIMEOUT as PREFLIGHT_TIMEOUT
from ._base import PROJECT_ROOT as PROJECT_ROOT
from ._base import RUN_ROOT as RUN_ROOT
from ._base import SNAPSHOT as SNAPSHOT
from ._base import SPATIAL_RECEIPT as SPATIAL_RECEIPT
from ._base import STARTUP_TIMEOUT as STARTUP_TIMEOUT
from ._base import VISUAL_SHA256 as VISUAL_SHA256
from ._base import VOLUMES as VOLUMES
from ._base import WAIT_GRACE as WAIT_GRACE
from ._base import AddedToken as AddedToken
from ._base import AutoTokenizer as AutoTokenizer
from ._base import Counter as Counter
from ._base import Path as Path
from ._base import TrainConfig as TrainConfig
from ._base import app as app
from ._base import argparse as argparse
from ._base import asdict as asdict
from ._base import continuation_image as continuation_image
from ._base import datetime as datetime
from ._base import evaluator as evaluator
from ._base import hashlib as hashlib
from ._base import hf_cache as hf_cache
from ._base import inspect as inspect
from ._base import inspect_jsonl_contract as inspect_jsonl_contract
from ._base import iter_jsonl as iter_jsonl
from ._base import json as json
from ._base import load_token_inventory as load_token_inventory
from ._base import modal as modal
from ._base import normalize_training_config as normalize_training_config
from ._base import os as os
from ._base import pilot_train as pilot_train
from ._base import replace as replace
from ._base import resolve_dataset_asset as resolve_dataset_asset
from ._base import resolve_dataset_image as resolve_dataset_image
from ._base import safe_open as safe_open
from ._base import sft_data as sft_data
from ._base import sft_runs as sft_runs
from ._base import sha256_file as sha256_file
from ._base import timezone as timezone
from ._base import torch as torch
from ._base import training_image as training_image
from ._base import upload_eval_jsonl as upload_eval_jsonl
from ._base import upload_training_bundle as upload_training_bundle
from ._base import uuid as uuid
from ._base import write_json_atomic as write_json_atomic
from ._coordinate import coordinate as coordinate
from ._coordinate import reserve as reserve
from ._launch import launch as launch
from ._launch import main as main
from ._plan import build_plan as build_plan
from ._plan import check_identity as check_identity
from ._plan import dataset_identity as dataset_identity
from ._plan import digest as digest
from ._plan import now as now
from ._plan import read_json as read_json
from ._plan import source_hashes as source_hashes
from ._plan import validate_config as validate_config
from ._plan import validate_mixture as validate_mixture
from ._plan import validate_run_name as validate_run_name
from ._runtime import continuation_preflight as continuation_preflight
from ._runtime import evaluation_conditions as evaluation_conditions
from ._runtime import panel_args as panel_args
from ._runtime import reload_volumes as reload_volumes
from ._runtime import verify_runtime as verify_runtime
from ._workers import compare_panels as compare_panels
from ._workers import evaluate_bounded as evaluate_bounded
from ._workers import train_bounded as train_bounded

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
    "build_plan",
    "check_identity",
    "checkpoint_audit",
    "compare_panels",
    "completion_audit",
    "continuation_image",
    "continuation_preflight",
    "coordinate",
    "dataset_identity",
    "datetime",
    "digest",
    "evaluate_bounded",
    "evaluation_conditions",
    "evaluator",
    "hashlib",
    "hf_cache",
    "inspect",
    "inspect_jsonl_contract",
    "iter_jsonl",
    "json",
    "launch",
    "load_token_inventory",
    "main",
    "matched_baseline",
    "modal",
    "normalize_training_config",
    "now",
    "os",
    "panel_args",
    "pilot_train",
    "read_json",
    "reload_volumes",
    "replace",
    "reserve",
    "resolve_dataset_asset",
    "resolve_dataset_image",
    "safe_open",
    "sft_data",
    "sft_runs",
    "sha256_file",
    "source_hashes",
    "timezone",
    "torch",
    "train_bounded",
    "training_image",
    "upload_eval_jsonl",
    "upload_training_bundle",
    "uuid",
    "validate_config",
    "validate_mixture",
    "validate_run_name",
    "verify_runtime",
    "write_json_atomic",
]
