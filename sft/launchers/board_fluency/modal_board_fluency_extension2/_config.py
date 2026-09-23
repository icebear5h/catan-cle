from __future__ import annotations

from decimal import Decimal

import modal

from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_eval import ModalCommon
from sft.launchers.board_fluency.modal_board_fluency_extension._types import Limits, Resources

DEFAULT_RUN_NAME = "board-fluency-extension-20260915-r02"
PARENT_RUN = "board-fluency-extension-20260915-r01"
PARENT_ROOT = f"/runs/catan-vision-sft/{PARENT_RUN}"
PARENT = PARENT_ROOT + "/training/checkpoints/checkpoint-128"
R06_RUN = "board-fluency-sft-20260915-r06"
R06_ROOT = f"/runs/catan-vision-sft/{R06_RUN}"
DATA_DIR = f"/data/board-fluency-sft/{R06_RUN}"
LOCAL_ROOT = original.PROJECT_ROOT / "artifacts/runs/sft"
LOCAL_PARENT = LOCAL_ROOT / PARENT_RUN
LOCAL_R06 = LOCAL_ROOT / R06_RUN
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
    "sft/launchers/board_fluency/modal_board_fluency_extension/_types.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/__init__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/__main__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_config.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_planning.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_baselines.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_prepare.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_training.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_coordination.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_launch.py",)
STAGE_SECONDS = {"prepare": 300, "train": 3300, "posteval": 900}
STARTUP, COORDINATOR_SECONDS, ABSOLUTE_SECONDS = 300, 6000, 5400
RESOURCES: dict[str, Resources] = {"gpu": {"gpu": "H200", "cpu": 16, "memory_gib": 128},
             "prepare": {"cpu": 4, "memory_gib": 16},
             "coordinator": {"cpu": 1, "memory_gib": 2}}
LIMITS: Limits = {"stage_seconds": STAGE_SECONDS, "startup_seconds": STARTUP,
          "coordinator_seconds": COORDINATOR_SECONDS, "absolute_seconds": ABSOLUTE_SECONDS,
          "reservation_seconds": 300, "resources": RESOURCES,
          "retries": 0, "max_containers": 1, "scaledown_seconds": 2}
CHECKPOINT_STEPS = [32, 64, 96, 128, 160, 192, 224, 256]
PARENT_CHECKPOINT_STEPS = [32, 64, 96, 128]
BUDGET_CLAIMS = "catan-board-fluency-budget-claims"
COST_R06_SHA256 = "dba85f90a58bf166030a66904f681312158045b381037a4ee2994dc8334cf9da"
COST_R06_KEY = "recorded_window_subtotal_plus_prior_allowance_plus_full_reserve_usd"
PRIOR_R06_USD = Decimal("7.47126099194740")
R01_ADDENDUM_SHA256 = "3c8796509b01c8cd7eae7de1de231134e2c9d4e048329838ccdf9e123e2df7db"
R01_RECORDED_USD = Decimal("3.78449619101530")
R01_TOTAL_KEY = "cumulative_recorded_window_plus_prior_allowances_usd"
R01_SUBTOTAL_KEY = "extension_recorded_window_subtotal_usd"
PRIOR_USD = Decimal("11.25575718296270")
CLAIM_KEY = R01_ADDENDUM_SHA256
TEACHER_SHA256 = "11e533d592566ae367fab87b3351a18a80d655aaa81a408250267bc5ecdc72b4"
FROZEN_DIGEST = "4b5d8892dc201fb0f9cda1decca4d1350225e2a6f29f6a3b8c7ddad123b8dcfe"
PARENT_SHA256 = {
    "launch.json": "6106f86d8672dfa296d957d618d3a891713cddcdab1b4d632c4dfc7ec5c8bfcd",
    "coordinator.json": "b8a73cd231bd62ce656c22cb77eb4c51285e5ce4388325b45c50de684e0e4087",
    "prepare/result.json": "c93f70eaf830abce5fe38f7a3e4dffbc40e046798517464b4751a38991b217b8",
    "train/result.json": "be6d79a0477f5e18a2fc8f223f597120eb22e1299e82a8eb656c1c6980a712e1",
    "posteval/result.json": "10058bb77687eaedc22063d1459fccda98347d2d8352d687b5881dac69167440",
    "prepare/wrapper.json": "4cd796ecc87d838b25efaf59f0572ad27f1a84f23f99d992daa51f3bb28d6604",
    "train/wrapper.json": "153af12797534a68adfaa369fda38c3019146e287a995ba1ee48fb1b1adaf786",
    "posteval/wrapper.json": "8ab496a093ec58b3cf236ee60063939a66d7bf0111b97292b952e7b61fd187c3",
    "training/checkpoints/checkpoint-128/trainer_state.json":
        "60825229859817711cef8b1878b54c74ccfb7783c6659cae6a21be5cf495629b",
}
LOCAL_PARENT_SHA256 = {
    "analysis.json": "03de3d4179ead80ace55a21a1a9cefb829cd009795e4074300b1ed9a7933d4aa",
    "cost_addendum.json": R01_ADDENDUM_SHA256,
    "orchestration.json": "8e3eba249ca54f34acd85a3c0bc5869025506fd77f365deea324eeec9d08a005",
    "stop_receipt.json": "98077cf1d3bff435d077249021ca434a481e133fb6caa692038f93e4df6d09d5",
}
POLICY = {"stages": list(STAGE_SECONDS), "additional_optimizer_steps": 256,
          "parent_optimizer_steps": 256, "cumulative_optimizer_steps": 512,
          "additional_presentations": 2048, "cumulative_presentations": 4096,
          "additional_unique_examples": 1152, "cumulative_unique_examples": 3200,
          "original_corpus_rows": 3200, "cumulative_corpus_epoch": 1.0,
          "original_rows_one_based_inclusive": [2049, 3200],
          "repeat_rows_one_based_inclusive": [1, 896],
          "already_consumed_rows_one_based_inclusive": [1, 2048],
          "checkpoint_steps": CHECKPOINT_STEPS, "fresh_optimizer_and_schedule": True,
          "sequential_sampling": True, "reuse_parent_uploaded_data": True,
          "automatic_retries_or_extensions": False,
          "budget_approval": "user chose raise-the-ceiling to $21 over fitting $15"}
SCHEMA = "catan_board_fluency_extension_r02_launch/v1"
app = modal.App("catan-board-fluency-extension-r02")
COMMON: ModalCommon = dict(
    image=original.eval_image, volumes=original.VOLUMES, secrets=[original.secret],
    startup_timeout=STARTUP, retries=LIMITS["retries"],
    max_containers=LIMITS["max_containers"],
    scaledown_window=LIMITS["scaledown_seconds"])
