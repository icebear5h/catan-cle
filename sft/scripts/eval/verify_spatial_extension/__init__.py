"""Read extension receipts from Modal volumes, or independently verify them offline.

python -B -m sft.scripts.eval.verify_spatial_extension --download --status-only
python -B -m sft.scripts.eval.verify_spatial_extension --download
python -B -m sft.scripts.eval.verify_spatial_extension
"""

from __future__ import annotations

from ._base import CHECKPOINT_FILES as CHECKPOINT_FILES
from ._base import CHECKPOINT_STEPS as CHECKPOINT_STEPS
from ._base import DEFAULT_RUN_NAME as DEFAULT_RUN_NAME
from ._base import EVALUATOR_FILES as EVALUATOR_FILES
from ._base import LOCAL_RUN_ROOT as LOCAL_RUN_ROOT
from ._base import NEW_PANEL_TASKS as NEW_PANEL_TASKS
from ._base import OLD_PANELS as OLD_PANELS
from ._base import PANEL_BUDGETS as PANEL_BUDGETS
from ._base import PARENT_CHECKPOINT as PARENT_CHECKPOINT
from ._base import PARENT_RUN as PARENT_RUN
from ._base import PROJECT_ROOT as PROJECT_ROOT
from ._base import SCORER_FILES as SCORER_FILES
from ._base import STATE_PATH as STATE_PATH
from ._base import Path as Path
from ._base import argparse as argparse
from ._base import check_identity as check_identity
from ._base import dataset_identity as dataset_identity
from ._base import digest as digest
from ._base import evaluation_conditions as evaluation_conditions
from ._base import evaluator as evaluator
from ._base import hashlib as hashlib
from ._base import json as json
from ._base import math as math
from ._base import modal as modal
from ._base import os as os
from ._base import read_json as read_json
from ._base import sha256_file as sha256_file
from ._base import sys as sys
from ._base import tempfile as tempfile
from ._base import time as time
from ._base import validate_config as validate_config
from ._panels import verify_panel as verify_panel
from ._panels import verify_training as verify_training
from ._receipts import atomic_write as atomic_write
from ._receipts import completed_post as completed_post
from ._receipts import download_receipts as download_receipts
from ._receipts import expected_checkpoint as expected_checkpoint
from ._receipts import full_history as full_history
from ._receipts import json_bytes as json_bytes
from ._receipts import load_launch as load_launch
from ._receipts import local_path as local_path
from ._receipts import require as require
from ._receipts import status_report as status_report
from ._receipts import validate_result as validate_result
from ._verify import main as main
from ._verify import verify_local as verify_local

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
    "atomic_write",
    "check_identity",
    "completed_post",
    "dataset_identity",
    "digest",
    "download_receipts",
    "evaluation_conditions",
    "evaluator",
    "expected_checkpoint",
    "full_history",
    "hashlib",
    "json",
    "json_bytes",
    "load_launch",
    "local_path",
    "main",
    "math",
    "modal",
    "os",
    "read_json",
    "require",
    "sha256_file",
    "status_report",
    "sys",
    "tempfile",
    "time",
    "validate_config",
    "validate_result",
    "verify_local",
    "verify_panel",
    "verify_training",
]
