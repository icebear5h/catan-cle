"""Opt-in 128-update extension of r06; local dry runs never contact Modal.

MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2 \
    .venv/bin/python -B -m sft.launchers.board_fluency.modal_board_fluency_extension \
    --run-name board-fluency-extension-20260915-r01 --budget-usd 15
Add --execute to reserve and supervise one run, or --stop to stop its entire app.
"""

from __future__ import annotations

__all__ = [
    "ABSOLUTE_SECONDS",
    "ANALYSIS_SHA256",
    "BASE",
    "BUDGET_CLAIMS",
    "CHECKPOINT_STEPS",
    "COMMON",
    "COORDINATOR_SECONDS",
    "COST_KEY",
    "COST_SHA256",
    "CheckpointCallback",
    "Counter",
    "DATA_DIR",
    "DEFAULT_RUN_NAME",
    "Decimal",
    "ExtensionCallback",
    "LIMITS",
    "LOCAL_PARENT",
    "LOCAL_PARENT_SHA256",
    "LOCAL_ROOT",
    "OFFLINE",
    "PARENT",
    "PARENT_ROOT",
    "PARENT_RUN",
    "PARENT_SHA256",
    "POLICY",
    "PRIOR_USD",
    "Path",
    "RESOURCES",
    "SOURCE_FILES",
    "STAGE_SECONDS",
    "STARTUP",
    "TrainConfig",
    "_message_pair",
    "annotations",
    "app",
    "argparse",
    "asdict",
    "bounded_stage",
    "budget_plan",
    "build_plan",
    "check_deadline",
    "check_manifest",
    "completed_stage",
    "configuration",
    "coordinate",
    "coordinate_work",
    "deadline_alarm",
    "encode_text_pair",
    "eval_panel",
    "evaluator",
    "full_checkpoint_manifest",
    "hashlib",
    "inspect_original",
    "json",
    "launch",
    "load_eval",
    "load_token_inventory",
    "main",
    "math",
    "modal",
    "normalized",
    "original",
    "os",
    "posteval_h200",
    "posteval_work",
    "prepare_cpu",
    "prepare_work",
    "prepared_for",
    "progress",
    "read_parent",
    "replace",
    "reserve",
    "resource_options",
    "retained_baselines",
    "rows_at",
    "sha256_file",
    "shared",
    "shutil",
    "signal",
    "source_hashes",
    "stop_run",
    "subprocess",
    "suffix_identity",
    "suppress",
    "sys",
    "text_boundaries",
    "threading",
    "time",
    "torch",
    "train_h200",
    "train_work",
    "trainer",
    "uuid",
    "verify_hashes",
    "verify_initialization",
    "verify_plan",
    "visual_file_digest",
    "worker",
    "write_json_atomic",
]

import argparse as argparse
import hashlib as hashlib
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
from contextlib import suppress as suppress
from dataclasses import asdict as asdict
from dataclasses import replace as replace
from decimal import Decimal as Decimal
from pathlib import Path as Path

import modal as modal
import torch as torch

from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_sft import BASE as BASE
from sft.launchers.board_fluency.modal_board_fluency_sft import OFFLINE as OFFLINE
from sft.launchers.board_fluency.modal_board_fluency_sft import (
    CheckpointCallback as CheckpointCallback,
)
from sft.launchers.board_fluency.modal_board_fluency_sft import check_deadline as check_deadline
from sft.launchers.board_fluency.modal_board_fluency_sft import check_manifest as check_manifest
from sft.launchers.board_fluency.modal_board_fluency_sft import deadline_alarm as deadline_alarm
from sft.launchers.board_fluency.modal_board_fluency_sft import eval_panel as eval_panel
from sft.launchers.board_fluency.modal_board_fluency_sft import evaluator as evaluator
from sft.launchers.board_fluency.modal_board_fluency_sft import load_eval as load_eval
from sft.launchers.board_fluency.modal_board_fluency_sft import progress as progress
from sft.launchers.board_fluency.modal_board_fluency_sft import rows_at as rows_at
from sft.launchers.board_fluency.modal_board_fluency_sft import shared as shared
from sft.launchers.board_fluency.modal_board_fluency_sft import trainer as trainer
from sft.launchers.board_fluency.modal_board_fluency_sft import (
    visual_file_digest as visual_file_digest,
)
from sft.scripts.train.train_trl_catan_vision import TrainConfig as TrainConfig
from sft.scripts.train.train_trl_catan_vision import _message_pair as _message_pair
from sft.scripts.train.train_trl_catan_vision import encode_text_pair as encode_text_pair
from sft.scripts.train.train_trl_catan_vision import load_token_inventory as load_token_inventory
from sft.scripts.train.train_trl_catan_vision import sha256_file as sha256_file
from sft.scripts.train.train_trl_catan_vision import write_json_atomic as write_json_atomic

from ._baselines import build_plan as build_plan
from ._baselines import inspect_original as inspect_original
from ._baselines import retained_baselines as retained_baselines
from ._baselines import suffix_identity as suffix_identity
from ._baselines import verify_plan as verify_plan
from ._config import ABSOLUTE_SECONDS as ABSOLUTE_SECONDS
from ._config import ANALYSIS_SHA256 as ANALYSIS_SHA256
from ._config import BUDGET_CLAIMS as BUDGET_CLAIMS
from ._config import CHECKPOINT_STEPS as CHECKPOINT_STEPS
from ._config import COMMON as COMMON
from ._config import COORDINATOR_SECONDS as COORDINATOR_SECONDS
from ._config import COST_KEY as COST_KEY
from ._config import COST_SHA256 as COST_SHA256
from ._config import DATA_DIR as DATA_DIR
from ._config import DEFAULT_RUN_NAME as DEFAULT_RUN_NAME
from ._config import LIMITS as LIMITS
from ._config import LOCAL_PARENT as LOCAL_PARENT
from ._config import LOCAL_PARENT_SHA256 as LOCAL_PARENT_SHA256
from ._config import LOCAL_ROOT as LOCAL_ROOT
from ._config import PARENT as PARENT
from ._config import PARENT_ROOT as PARENT_ROOT
from ._config import PARENT_RUN as PARENT_RUN
from ._config import PARENT_SHA256 as PARENT_SHA256
from ._config import POLICY as POLICY
from ._config import PRIOR_USD as PRIOR_USD
from ._config import RESOURCES as RESOURCES
from ._config import SOURCE_FILES as SOURCE_FILES
from ._config import STAGE_SECONDS as STAGE_SECONDS
from ._config import STARTUP as STARTUP
from ._config import app as app
from ._coordination import coordinate as coordinate
from ._coordination import coordinate_work as coordinate_work
from ._coordination import posteval_h200 as posteval_h200
from ._coordination import prepare_cpu as prepare_cpu
from ._coordination import reserve as reserve
from ._coordination import stop_run as stop_run
from ._coordination import train_h200 as train_h200
from ._launch import launch as launch
from ._launch import main as main
from ._planning import budget_plan as budget_plan
from ._planning import configuration as configuration
from ._planning import normalized as normalized
from ._planning import read_parent as read_parent
from ._planning import resource_options as resource_options
from ._planning import source_hashes as source_hashes
from ._planning import verify_hashes as verify_hashes
from ._prepare import completed_stage as completed_stage
from ._prepare import full_checkpoint_manifest as full_checkpoint_manifest
from ._prepare import prepare_work as prepare_work
from ._prepare import prepared_for as prepared_for
from ._prepare import text_boundaries as text_boundaries
from ._prepare import verify_initialization as verify_initialization
from ._training import ExtensionCallback as ExtensionCallback
from ._training import bounded_stage as bounded_stage
from ._training import posteval_work as posteval_work
from ._training import train_work as train_work
from ._training import worker as worker
