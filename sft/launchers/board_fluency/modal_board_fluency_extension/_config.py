from __future__ import annotations

from decimal import Decimal

import modal

from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_eval import ModalCommon

from ._types import Limits, Resources

DEFAULT_RUN_NAME = "board-fluency-extension-20260915-r01"
PARENT_RUN = "board-fluency-sft-20260915-r06"
PARENT_ROOT = f"/runs/catan-vision-sft/{PARENT_RUN}"
PARENT = PARENT_ROOT + "/training/checkpoints/checkpoint-128"
DATA_DIR = f"/data/board-fluency-sft/{PARENT_RUN}"
LOCAL_ROOT = original.PROJECT_ROOT / "artifacts/runs/sft"
LOCAL_PARENT = LOCAL_ROOT / PARENT_RUN
SOURCE_FILES = (*original.SOURCE_FILES,
    "sft/launchers/board_fluency/modal_board_fluency_extension/__init__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension/__main__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension/_config.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension/_planning.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension/_baselines.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension/_prepare.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension/_training.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension/_coordination.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension/_launch.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension/_types.py",)
STAGE_SECONDS = {"prepare": 300, "train": 1800, "posteval": 1050}
STARTUP, COORDINATOR_SECONDS, ABSOLUTE_SECONDS = 300, 4500, 4350
RESOURCES: dict[str, Resources] = {"gpu": {"gpu": "H200", "cpu": 16, "memory_gib": 128},
             "prepare": {"cpu": 4, "memory_gib": 16},
             "coordinator": {"cpu": 1, "memory_gib": 2}}
LIMITS: Limits = {"stage_seconds": STAGE_SECONDS, "startup_seconds": STARTUP,
          "coordinator_seconds": COORDINATOR_SECONDS, "absolute_seconds": ABSOLUTE_SECONDS,
          "reservation_seconds": 300, "resources": RESOURCES,
          "retries": 0, "max_containers": 1, "scaledown_seconds": 2}
CHECKPOINT_STEPS = [32, 64, 96, 128]
COST_KEY = "recorded_window_subtotal_plus_prior_allowance_plus_full_reserve_usd"
PRIOR_USD = Decimal("7.47126099194740")
COST_SHA256 = "dba85f90a58bf166030a66904f681312158045b381037a4ee2994dc8334cf9da"
BUDGET_CLAIMS = "catan-board-fluency-budget-claims"
ANALYSIS_SHA256 = "8e5aa27178f966ba58eeb27b2ea6bef7b7c309b40ce50847457614e5a9b755ec"
# Historical receipts pin completed work, not a historically unrecorded adapter hash.
PARENT_SHA256 = {
    "launch.json": "0bb1503be57b34c1391b2a1992a0096d8550f4a5e85322c84913cd01f74823f3",
    "coordinator.json": "2c0478cd1c2f9cc3b99ab546a2cf77cb71396cc2dd237583b458607201adcbc8",
    "prepare/result.json": "3cfa48698a01dda8d66eb4863d44acef4daf8d7e7259696416f8cc5702e46e70",
    "gate/result.json": "45d59210d19c3a201d9be3fe14e522301935e4dc3f8dc03b2c8941b68b2e22a2",
    "train/result.json": "1138963ad0b6b2b5b3f5d4f292c79ff795ba9f27f2887e6ab2598c2409936a10",
    "posteval/result.json": "8d8e9921de115e896aaa436ebd69a08b8d35cf3699748e8b4b272e13a99b1c55",
    "prepare/wrapper.json": "232ec5995552710686f7d07107e3c4c99dea18b9cfe3d3934ff009624d322571",
    "gate/wrapper.json": "a5c5454e8a5ebc04e48ecb0f67472214cbc83f8a5b8ae8788772211cf1e15280",
    "train/wrapper.json": "5e44f8aa342f1cf7ab6b12163f5b1e6bea1c7e1acd50df86a38f5248be861914",
    "posteval/wrapper.json": "36588d5bb1a20cc9eedf000e3d2585986cf31a4a5311d2048d2a07b7840ce93c",
    "prepare/teacher120.jsonl": "11e533d592566ae367fab87b3351a18a80d655aaa81a408250267bc5ecdc72b4",
    "training/checkpoints/checkpoint-128/trainer_state.json":
        "7c3e519de05b9147173ce43ec8bdc8740c9030a60ed817f51ae6235bd53f181c",
}
LOCAL_PARENT_SHA256 = {
    "cost_estimate.json": COST_SHA256, "analysis.json": ANALYSIS_SHA256,
    "orchestration.json": "ea714c141e82e259af1f6ae942c744bd9fde4df897775d47db5e31bb3bb8360d",
    "stop_receipt.json": "dec79f44f3281e128489870475639bdecc9d26c8d43b521dbc2c96ea8b9be8a9",
}
POLICY = {"stages": list(STAGE_SECONDS), "additional_optimizer_steps": 128,
          "parent_optimizer_steps": 128, "cumulative_optimizer_steps": 256,
          "additional_unique_examples": 1024, "cumulative_unique_examples": 2048,
          "original_corpus_rows": 3200, "cumulative_corpus_epoch": 0.64,
          "original_rows_one_based_inclusive": [1025, 2048],
          "checkpoint_steps": CHECKPOINT_STEPS, "fresh_optimizer_and_schedule": True,
          "sequential_sampling": True, "reuse_parent_uploaded_data": True,
          "automatic_retries_or_extensions": False}
app = modal.App("catan-board-fluency-extension")
COMMON: ModalCommon = dict(
    image=original.eval_image, volumes=original.VOLUMES, secrets=[original.secret],
    startup_timeout=STARTUP, retries=LIMITS["retries"],
    max_containers=LIMITS["max_containers"],
    scaledown_window=LIMITS["scaledown_seconds"])
