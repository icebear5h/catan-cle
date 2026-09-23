"""Shared imports, constants and types for this package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path as Path

import modal

from sft.launchers.spatial.modal_spatial_continuation import LOCAL_RUN_ROOT as LOCAL_RUN_ROOT
from sft.launchers.spatial.modal_spatial_continuation import NEW_PANEL_TASKS as NEW_PANEL_TASKS
from sft.launchers.spatial.modal_spatial_continuation import OLD_PANELS as OLD_PANELS
from sft.launchers.spatial.modal_spatial_continuation import PANEL_BUDGETS as PANEL_BUDGETS
from sft.launchers.spatial.modal_spatial_continuation import PROJECT_ROOT as PROJECT_ROOT
from sft.launchers.spatial.modal_spatial_continuation import check_identity as check_identity
from sft.launchers.spatial.modal_spatial_continuation import dataset_identity as dataset_identity
from sft.launchers.spatial.modal_spatial_continuation import digest as digest
from sft.launchers.spatial.modal_spatial_continuation import (
    evaluation_conditions as evaluation_conditions,
)
from sft.launchers.spatial.modal_spatial_continuation import read_json as read_json
from sft.launchers.spatial.modal_spatial_extension import CHECKPOINT_STEPS as CHECKPOINT_STEPS
from sft.launchers.spatial.modal_spatial_extension import DEFAULT_RUN_NAME as DEFAULT_RUN_NAME
from sft.launchers.spatial.modal_spatial_extension import PARENT_CHECKPOINT as PARENT_CHECKPOINT
from sft.launchers.spatial.modal_spatial_extension import PARENT_RUN as PARENT_RUN
from sft.launchers.spatial.modal_spatial_extension import validate_config as validate_config
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import sha256_file as sha256_file

__all__ = [
    "CHECKPOINT_FILES",
    "CHECKPOINT_STEPS",
    "DEFAULT_RUN_NAME",
    "EVALUATOR_FILES",
    "LOCAL_RUN_ROOT",
    "NEW_PANEL_TASKS",
    "OLD_PANELS",
    "PANEL_BUDGETS",
    "PARENT_CHECKPOINT",
    "PARENT_RUN",
    "PROJECT_ROOT",
    "Path",
    "SCORER_FILES",
    "STATE_PATH",
    "argparse",
    "check_identity",
    "dataset_identity",
    "digest",
    "evaluation_conditions",
    "evaluator",
    "hashlib",
    "json",
    "math",
    "modal",
    "os",
    "read_json",
    "sha256_file",
    "sys",
    "tempfile",
    "time",
    "validate_config",
]

EVALUATOR_FILES = (
    "sft/scripts/eval/eval_qwen_vl_adapter/__init__.py",
    "sft/scripts/eval/eval_qwen_vl_adapter/__main__.py",
    "sft/scripts/eval/eval_qwen_vl_adapter/_scoring.py",
    "sft/scripts/eval/eval_qwen_vl_adapter/_candidates.py",
    "sft/scripts/eval/eval_qwen_vl_adapter/_summaries.py",
    "sft/scripts/eval/eval_qwen_vl_adapter/_reporting.py",
    "sft/scripts/eval/eval_qwen_vl_adapter/_model.py",
    "sft/scripts/eval/eval_qwen_vl_adapter/_jobs.py",
    "sft/scripts/eval/eval_qwen_vl_adapter/_cli.py",
)

SCORER_FILES = (
    *EVALUATOR_FILES, "sft/board/spatial_tasks/__init__.py",
    "sft/board/spatial_tasks/_topology.py", "sft/board/spatial_tasks/_contracts.py",
    "sft/board/spatial_tasks/_scoring.py",
    "sft/board_state_readout.py", "sft/analysis/behavior_diagnostics/__init__.py",
    "sft/analysis/behavior_diagnostics/_types.py",
    "sft/analysis/behavior_diagnostics/_summaries.py",
    "sft/analysis/behavior_diagnostics/_history.py",
    "sft/analysis/behavior_diagnostics/_markdown.py",
    "evals/catan_board_bench/tokens/__init__.py",
    "evals/catan_board_bench/tokens/atlas.py",
    "evals/catan_board_bench/tokens/manifest.py",
    "evals/catan_board_bench/tokens/vocabulary.py",
    "cle/game_engine/board_tokens.py",
    "cle/game_engine/models/map.py", "cle/game_engine/models/enums.py",
    "cle/game_engine/models/player.py",
)

CHECKPOINT_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "visual_model.safetensors",
    "trainable_parameters.json", "training_config.json", "tokenizer_config.json",
    "tokenizer.json", "trainer_state.json",
)

STATE_PATH = "checkpoints/checkpoint-256/trainer_state.json"
