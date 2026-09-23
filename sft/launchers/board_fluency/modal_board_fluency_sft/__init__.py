"""Opt-in, single-run board-fluency continuation; dry runs never contact Modal.

MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2 \
    .venv/bin/python -m sft.launchers.board_fluency.modal_board_fluency_sft --run-name NAME --budget-usd 15
Add --execute only after reviewing the local admission/configuration receipt.
The detached CPU coordinator owns preparation, gate, training and post-evaluation.
"""

from __future__ import annotations

__all__ = [
    "ABSOLUTE_SECONDS",
    "ALLOWED",
    "ATOL",
    "Any",
    "BASE",
    "BASELINE_ROOT",
    "BASELINE_SHA256",
    "COMMON",
    "CONTROL_CPU",
    "COORDINATOR_SECONDS",
    "CheckpointCallback",
    "Counter",
    "DATASET",
    "DATA_FILES",
    "EVALUATOR_FILES",
    "FLUENCY_BUILDER_FILES",
    "GPU",
    "GateCallback",
    "HF_SECRET_NAME",
    "OFFLINE",
    "PARENT",
    "PRIOR_ATTEMPTS_USD",
    "PROJECT_ROOT",
    "Path",
    "REVIEW",
    "REVIEW_SHA256",
    "RTOL",
    "SOURCE_FILES",
    "STAGE_SECONDS",
    "STARTUP",
    "TRAINER_FILES",
    "TrainConfig",
    "TrainerCallback",
    "VOLUMES",
    "_message_pair",
    "annotations",
    "app",
    "argparse",
    "asdict",
    "atlas_tokens",
    "bounded_stage",
    "budget_plan",
    "build_plan",
    "cached_base_files",
    "check_deadline",
    "check_manifest",
    "compare_probes",
    "configuration",
    "contextmanager",
    "coordinate",
    "coordinate_work",
    "deadline_alarm",
    "defaultdict",
    "deque",
    "encode_text_pair",
    "eval_image",
    "eval_panel",
    "evaluator",
    "expand_lora_bundle",
    "gate_h200",
    "gate_work",
    "gc",
    "hashlib",
    "inspect",
    "inspect_data",
    "io",
    "json",
    "launch",
    "load_eval",
    "load_token_inventory",
    "main",
    "math",
    "modal",
    "os",
    "pad_text_inputs",
    "posteval_h200",
    "posteval_work",
    "prepare_cpu",
    "prepare_work",
    "prepared_for",
    "probe",
    "progress",
    "replace",
    "require_dependencies",
    "reserve",
    "retained_prevalidation",
    "row_contract",
    "rows_at",
    "safe_open",
    "save_file",
    "secret",
    "sft_data",
    "sft_runs",
    "sha256_file",
    "shared",
    "shutil",
    "signal",
    "source_hashes",
    "stop_app",
    "stop_run",
    "subprocess",
    "sys",
    "teacher_ids",
    "threading",
    "time",
    "torch",
    "train_h200",
    "train_work",
    "trainer",
    "uuid",
    "validate_dataset",
    "verify_plan",
    "verify_uploaded",
    "visual_digest",
    "visual_file_digest",
    "worker",
    "write_json_atomic",
    "write_rows",
]

import argparse as argparse
import gc as gc
import hashlib as hashlib
import inspect as inspect
import io as io
import json as json
import math as math
import os as os
import shutil as shutil
import signal as signal
import subprocess as subprocess
import sys as sys
import threading as threading
import time as time
import uuid as uuid
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from collections import deque as deque
from contextlib import contextmanager as contextmanager
from dataclasses import asdict as asdict
from dataclasses import replace as replace
from pathlib import Path as Path
from typing import Any as Any

import modal as modal
import torch as torch
from safetensors import safe_open as safe_open
from safetensors.torch import save_file as save_file
from transformers import TrainerCallback as TrainerCallback

from evals.catan_board_bench.tokens import atlas_tokens as atlas_tokens
from sft.launchers.board_fluency import modal_board_fluency_eval as shared
from sft.launchers.board_fluency.modal_board_fluency_eval import VOLUMES as VOLUMES
from sft.launchers.board_fluency.modal_board_fluency_eval import eval_image as eval_image
from sft.launchers.modal_catan_vision_sft import HF_SECRET_NAME as HF_SECRET_NAME
from sft.launchers.modal_catan_vision_sft import sft_data as sft_data
from sft.launchers.modal_catan_vision_sft import sft_runs as sft_runs
from sft.lora_expansion import expand_lora_bundle as expand_lora_bundle
from sft.paths import PROJECT_ROOT as PROJECT_ROOT
from sft.scripts.builders.build_board_fluency_dataset import validate_dataset as validate_dataset
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train import train_trl_catan_vision as trainer
from sft.scripts.train.train_trl_catan_vision import TrainConfig as TrainConfig
from sft.scripts.train.train_trl_catan_vision import _message_pair as _message_pair
from sft.scripts.train.train_trl_catan_vision import encode_text_pair as encode_text_pair
from sft.scripts.train.train_trl_catan_vision import load_token_inventory as load_token_inventory
from sft.scripts.train.train_trl_catan_vision import pad_text_inputs as pad_text_inputs
from sft.scripts.train.train_trl_catan_vision import sha256_file as sha256_file
from sft.scripts.train.train_trl_catan_vision import write_json_atomic as write_json_atomic

from ._callbacks import CheckpointCallback as CheckpointCallback
from ._callbacks import GateCallback as GateCallback
from ._config import ABSOLUTE_SECONDS as ABSOLUTE_SECONDS
from ._config import ALLOWED as ALLOWED
from ._config import ATOL as ATOL
from ._config import BASE as BASE
from ._config import BASELINE_ROOT as BASELINE_ROOT
from ._config import BASELINE_SHA256 as BASELINE_SHA256
from ._config import COMMON as COMMON
from ._config import CONTROL_CPU as CONTROL_CPU
from ._config import COORDINATOR_SECONDS as COORDINATOR_SECONDS
from ._config import DATA_FILES as DATA_FILES
from ._config import DATASET as DATASET
from ._config import EVALUATOR_FILES as EVALUATOR_FILES
from ._config import FLUENCY_BUILDER_FILES as FLUENCY_BUILDER_FILES
from ._config import GPU as GPU
from ._config import OFFLINE as OFFLINE
from ._config import PARENT as PARENT
from ._config import PRIOR_ATTEMPTS_USD as PRIOR_ATTEMPTS_USD
from ._config import REVIEW as REVIEW
from ._config import REVIEW_SHA256 as REVIEW_SHA256
from ._config import RTOL as RTOL
from ._config import SOURCE_FILES as SOURCE_FILES
from ._config import STAGE_SECONDS as STAGE_SECONDS
from ._config import STARTUP as STARTUP
from ._config import TRAINER_FILES as TRAINER_FILES
from ._config import app as app
from ._config import secret as secret
from ._data import budget_plan as budget_plan
from ._data import configuration as configuration
from ._data import inspect_data as inspect_data
from ._data import progress as progress
from ._data import require_dependencies as require_dependencies
from ._data import row_contract as row_contract
from ._data import rows_at as rows_at
from ._data import source_hashes as source_hashes
from ._data import teacher_ids as teacher_ids
from ._gate import gate_work as gate_work
from ._gate import prepared_for as prepared_for
from ._gate import retained_prevalidation as retained_prevalidation
from ._launch import coordinate as coordinate
from ._launch import coordinate_work as coordinate_work
from ._launch import launch as launch
from ._launch import main as main
from ._launch import stop_app as stop_app
from ._launch import stop_run as stop_run
from ._planning import build_plan as build_plan
from ._planning import cached_base_files as cached_base_files
from ._planning import check_deadline as check_deadline
from ._planning import check_manifest as check_manifest
from ._planning import deadline_alarm as deadline_alarm
from ._planning import prepare_work as prepare_work
from ._planning import verify_plan as verify_plan
from ._planning import verify_uploaded as verify_uploaded
from ._planning import write_rows as write_rows
from ._probes import compare_probes as compare_probes
from ._probes import eval_panel as eval_panel
from ._probes import load_eval as load_eval
from ._probes import probe as probe
from ._probes import visual_digest as visual_digest
from ._probes import visual_file_digest as visual_file_digest
from ._workers import bounded_stage as bounded_stage
from ._workers import gate_h200 as gate_h200
from ._workers import posteval_h200 as posteval_h200
from ._workers import posteval_work as posteval_work
from ._workers import prepare_cpu as prepare_cpu
from ._workers import reserve as reserve
from ._workers import train_h200 as train_h200
from ._workers import train_work as train_work
from ._workers import worker as worker
