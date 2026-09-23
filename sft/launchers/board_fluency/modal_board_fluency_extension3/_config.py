from __future__ import annotations

from decimal import Decimal

import modal

from sft.launchers.board_fluency import modal_board_fluency_extension as ext1
from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_eval import ModalCommon

from ._types import Limits, Resource

DEFAULT_RUN_NAME = "board-fluency-extension-20260915-r03"
PARENT_RUN = "board-fluency-extension-20260915-r02"
PARENT_ROOT = f"/runs/catan-vision-sft/{PARENT_RUN}"
PARENT = PARENT_ROOT + "/training/checkpoints/checkpoint-256"
R01_RUN = ext1.DEFAULT_RUN_NAME
R01_ROOT = f"/runs/catan-vision-sft/{R01_RUN}"
R01_CHECKPOINT = R01_ROOT + "/training/checkpoints/checkpoint-128"
R06_RUN = "board-fluency-sft-20260915-r06"
R06_ROOT = f"/runs/catan-vision-sft/{R06_RUN}"
DATA_DIR = f"/data/board-fluency-sft/{R06_RUN}"
LOCAL_ROOT = original.PROJECT_ROOT / "artifacts/runs/sft"
LOCAL_PARENT = LOCAL_ROOT / PARENT_RUN
LOCAL_R06 = LOCAL_ROOT / R06_RUN
LOCAL_R01 = LOCAL_ROOT / R01_RUN
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
    "sft/launchers/board_fluency/modal_board_fluency_extension2/__init__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/__main__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_config.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_planning.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_baselines.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_prepare.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_training.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_coordination.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension2/_launch.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/__init__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/__main__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/_config.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/_planning.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/_baselines.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/_prepare.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/_training.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/_coordination.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/_launch.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension3/_types.py",)
STAGE_SECONDS = {"prepare": 300, "train": 6200, "posteval": 900}
STARTUP, COORDINATOR_SECONDS, ABSOLUTE_SECONDS = 300, 8000, 7500
RESOURCES: dict[str, Resource] = {"gpu": {"gpu": "H200", "cpu": 16, "memory_gib": 128},
             "prepare": {"cpu": 4, "memory_gib": 16},
             "coordinator": {"cpu": 1, "memory_gib": 2}}
LIMITS: Limits = {"stage_seconds": STAGE_SECONDS, "startup_seconds": STARTUP,
          "coordinator_seconds": COORDINATOR_SECONDS, "absolute_seconds": ABSOLUTE_SECONDS,
          "reservation_seconds": 300, "resources": RESOURCES,
          "retries": 0, "max_containers": 1, "scaledown_seconds": 2}
CHECKPOINT_STEPS = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
PARENT_CHECKPOINT_STEPS = [32, 64, 96, 128, 160, 192, 224, 256]
BUDGET_CLAIMS = "catan-board-fluency-budget-claims"
R01_ADDENDUM_SHA256 = "3c8796509b01c8cd7eae7de1de231134e2c9d4e048329838ccdf9e123e2df7db"
R01_TOTAL_KEY = "cumulative_recorded_window_plus_prior_allowances_usd"
PRIOR_R01_USD = Decimal("11.25575718296270")
R02_ADDENDUM_SHA256 = "69a840261b824f9f67d59f67bb04dea5e4d4ff1b429c85a0452aca15c78c59c2"
R02_RECORDED_USD = Decimal("5.17718958319660")
R02_SUBTOTAL_KEY = "extension_recorded_window_subtotal_usd"
R02_TOTAL_KEY = "cumulative_recorded_window_plus_prior_allowances_usd"
R02_PRIOR_KEY = "prior_allowances_usd"
PRIOR_USD = Decimal("16.43294676615930")
CLAIM_KEY = R02_ADDENDUM_SHA256
TEACHER_SHA256 = "11e533d592566ae367fab87b3351a18a80d655aaa81a408250267bc5ecdc72b4"
FROZEN_DIGEST = "4b5d8892dc201fb0f9cda1decca4d1350225e2a6f29f6a3b8c7ddad123b8dcfe"
PARENT_SHA256 = {
    "launch.json": "d53d5e332672b55a242c2659f72e7039522379ff80c58fddfc9b63e1b0a65391",
    "coordinator.json": "0ae88c7c52b71d8cbc231c34356c0f955f63aa7e987ee86b23f6b3ba67f646dc",
    "prepare/result.json": "bd61370b52ce99f3c329e9f5cf7168270d5034d9c9efa0594dd097bbd800b332",
    "train/result.json": "cc918a2fefef95754bc8a99ecefa6f167db2b9bbec1c1395114a57d5e1dc9193",
    "posteval/result.json": "7143fdf6ee688d26b64131649ae24b8b656eb1a3e4f1b7eac1e2a4b27b9da678",
    "prepare/wrapper.json": "faa4131bbf9013aa39754e1997007862ad15ef3f194b29c298cf459f40682bf0",
    "train/wrapper.json": "2fe56c63b52eeed4f1bc3b29cdbb36834f11f677ff54e2f110afda31405615c8",
    "posteval/wrapper.json": "b6c2e985f5970c44a7472061a1004829262fbd307175bf5b5426f601adda076a",
    "training/checkpoints/checkpoint-256/trainer_state.json":
        "f2ca4675003b42d93085994b893b2780f54eb0977969a67f3a5f017e74d0c00c",
}
LOCAL_PARENT_SHA256 = {
    "analysis.json": "5481605f5228f133945ac2d9ef8aabfc2bd1ada95ecd1424f44ddc624f2cedee",
    "cost_addendum.json": R02_ADDENDUM_SHA256,
    "orchestration.json": "35d956b36426532ad7b8cd345934b926e0e8a2d45d196256e9b567e419f0cd51",
    "stop_receipt.json": "8e4d23c396a7f7718d8b0c4d0840f739b5d86422f1308468ab6dbd1b411cdcd7",
}
POLICY = {"stages": list(STAGE_SECONDS), "additional_optimizer_steps": 512,
          "parent_optimizer_steps": 512, "cumulative_optimizer_steps": 1024,
          "additional_presentations": 4096, "cumulative_presentations": 8192,
          "additional_unique_examples": 0, "cumulative_unique_examples": 3200,
          "original_corpus_rows": 3200, "cumulative_corpus_epoch": 1.0,
          "epoch2_rows": 3200, "epoch3_rows": 896,
          "epoch2_rows_one_based_inclusive": [1, 3200],
          "epoch3_rows_one_based_inclusive": [1, 896],
          "checkpoint_steps": CHECKPOINT_STEPS, "fresh_optimizer_and_schedule": True,
          "sequential_sampling": True, "reuse_parent_uploaded_data": True,
          "automatic_retries_or_extensions": False,
          "budget_approval": "user approved ~$31 for 512 repeat steps; never allow above 31"}
SCHEMA = "catan_board_fluency_extension_r03_launch/v1"
app = modal.App("catan-board-fluency-extension-r03")
COMMON: ModalCommon = dict(
    image=original.eval_image, volumes=original.VOLUMES, secrets=[original.secret],
    startup_timeout=STARTUP, retries=LIMITS["retries"],
    max_containers=LIMITS["max_containers"],
    scaledown_window=LIMITS["scaledown_seconds"])
