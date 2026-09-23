from __future__ import annotations

import modal

from sft.launchers.board_fluency import modal_board_fluency_eval as shared
from sft.launchers.board_fluency.modal_board_fluency_eval import (
    VOLUMES,
    ModalCommon,
    ModalGpu,
    eval_image,
)
from sft.paths import PROJECT_ROOT

PARENT = "/runs/catan-vision-sft/spatial-continuation-20260912-r01/checkpoints/checkpoint-128"
BASE = ("/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
DATASET = PROJECT_ROOT / "artifacts/generated/sft/symbolic_board_fluency_sft_v1"
REVIEW = PROJECT_ROOT / shared.DEFAULT_REVIEW
REVIEW_SHA256 = "40469f58d060dd9a1d24354df42d959971172ba6e0de5e70b153cdda747e5b12"
DATA_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "validation_eval.jsonl",
              "trainable_tokens.json", "metadata.json", "manifest.json")
STAGE_SECONDS = {"prepare": 1200, "gate": 450, "train": 3300, "posteval": 1800}
STARTUP = 300
COORDINATOR_SECONDS = 9000
CONTROL_CPU = 1.0
# Includes r01/r02 CPU starts, r03 parity check and r04's capped baseline run.
PRIOR_ATTEMPTS_USD = 2.50
BASELINE_ROOT = "/runs/catan-vision-sft/board-fluency-sft-20260915-r04"
BASELINE_SHA256 = "2fc6ce78b4f1620d1b186a8e4052cf38d3cad4841250d1522703c2e9edccdd00"
ABSOLUTE_SECONDS = 8700
ATOL, RTOL = 0.002, 0.0002
ALLOWED = {"language_lora", "atlas_input_rows", "atlas_output_rows"}
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "PYTHONUNBUFFERED": "1"}
# train_trl_catan_vision is a same-name package: pin its facade plus every impl module.
TRAINER_FILES = (
    "sft/scripts/train/train_trl_catan_vision/__init__.py",
    "sft/scripts/train/train_trl_catan_vision/_common.py",
    "sft/scripts/train/train_trl_catan_vision/_config.py",
    "sft/scripts/train/train_trl_catan_vision/_text_data.py",
    "sft/scripts/train/train_trl_catan_vision/_vision_data.py",
    "sft/scripts/train/train_trl_catan_vision/_datasets.py",
    "sft/scripts/train/train_trl_catan_vision/_visual.py",
    "sft/scripts/train/train_trl_catan_vision/_structure.py",
    "sft/scripts/train/train_trl_catan_vision/_model_tokens.py",
    "sft/scripts/train/train_trl_catan_vision/_frozen.py",
    "sft/scripts/train/train_trl_catan_vision/_bundles.py",
    "sft/scripts/train/train_trl_catan_vision/_optim.py",
    "sft/scripts/train/train_trl_catan_vision/_trainer.py",
    "sft/scripts/train/train_trl_catan_vision/_sft.py",
    "sft/scripts/train/train_trl_catan_vision/_run.py",
)
# eval_qwen_vl_adapter is a same-name package: pin its facade plus every impl module.
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
# build_board_fluency_dataset is a same-name package: pin its facade plus every impl module.
FLUENCY_BUILDER_FILES = (
    "sft/scripts/builders/build_board_fluency_dataset/__init__.py",
    "sft/scripts/builders/build_board_fluency_dataset/_sources.py",
    "sft/scripts/builders/build_board_fluency_dataset/_selection.py",
    "sft/scripts/builders/build_board_fluency_dataset/_select.py",
    "sft/scripts/builders/build_board_fluency_dataset/_rows.py",
    "sft/scripts/builders/build_board_fluency_dataset/_validate.py",
    "sft/scripts/builders/build_board_fluency_dataset/_build.py",
)
SOURCE_FILES = (
    "sft/launchers/board_fluency/modal_board_fluency_eval/__init__.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/__main__.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_config.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_validation.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_snapshots.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_remote.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_cli.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/__init__.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/__main__.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/_config.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/_data.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/_planning.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/_probes.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/_callbacks.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/_gate.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/_workers.py",
    "sft/launchers/board_fluency/modal_board_fluency_sft/_launch.py",
    "sft/launchers/_config_fields.py",
    "sft/launchers/_train_config.py",
    "sft/lora_expansion/__init__.py",
    "sft/lora_expansion/_constants.py",
    "sft/lora_expansion/_shapes.py",
    "sft/lora_expansion/_bundle.py",
    *FLUENCY_BUILDER_FILES, *TRAINER_FILES, *EVALUATOR_FILES,
    "sft/board/board_fluency_scoring.py",
)
app = modal.App("catan-board-fluency-sft")
secret = modal.Secret.from_name("huggingface-secret-2", required_keys=["HF_TOKEN"])
COMMON: ModalCommon = dict(image=eval_image, volumes=VOLUMES, secrets=[secret],
                           startup_timeout=STARTUP, retries=0, max_containers=1,
                           scaledown_window=2)
GPU: ModalGpu = dict(**COMMON, gpu="H200", cpu=(16.0, 16.0), memory=(128 * 1024, 128 * 1024))
