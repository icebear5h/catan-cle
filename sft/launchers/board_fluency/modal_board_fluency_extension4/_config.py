from __future__ import annotations

from decimal import Decimal

import modal

from sft.launchers.board_fluency import modal_board_fluency_extension2 as ext2
from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_eval import ModalCommon
from sft.launchers.board_fluency.modal_board_fluency_extension3._types import Limits, Resource

DEFAULT_RUN_NAME = "board-fluency-extension-20260915-r04"
PARENT_RUN = "board-fluency-extension-20260915-r03"
PARENT_ROOT = f"/runs/catan-vision-sft/{PARENT_RUN}"
PARENT = PARENT_ROOT + "/training/checkpoints/checkpoint-512"
R02_RUN = ext2.DEFAULT_RUN_NAME
R02_ROOT = f"/runs/catan-vision-sft/{R02_RUN}"
R02_CHECKPOINT = R02_ROOT + "/training/checkpoints/checkpoint-256"
R06_RUN = "board-fluency-sft-20260915-r06"
R06_ROOT = f"/runs/catan-vision-sft/{R06_RUN}"
DATA_DIR = f"/data/board-fluency-sft/{R06_RUN}"
LOCAL_ROOT = original.PROJECT_ROOT / "artifacts/runs/sft"
LOCAL_PARENT = LOCAL_ROOT / PARENT_RUN
LOCAL_R06 = LOCAL_ROOT / R06_RUN
LOCAL_R02 = LOCAL_ROOT / R02_RUN
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
    "sft/launchers/board_fluency/modal_board_fluency_extension3/_types.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension4/__init__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension4/__main__.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension4/_config.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension4/_planning.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension4/_baselines.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension4/_prepare.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension4/_training.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension4/_coordination.py",
    "sft/launchers/board_fluency/modal_board_fluency_extension4/_launch.py",)
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
PARENT_CHECKPOINT_STEPS = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
BUDGET_CLAIMS = "catan-board-fluency-budget-claims"
R02_ADDENDUM_SHA256 = "69a840261b824f9f67d59f67bb04dea5e4d4ff1b429c85a0452aca15c78c59c2"
R02_TOTAL_KEY = "cumulative_recorded_window_plus_prior_allowances_usd"
PRIOR_R02_USD = Decimal("16.43294676615930")
R03_ADDENDUM_SHA256 = "0b68f7691cf0b986df0f1bf90aaab014d16087c9ccf71b1c5fb3889385dbcac0"
R03_RECORDED_USD = Decimal("9.65717822547840")
R03_SUBTOTAL_KEY = "extension_recorded_window_subtotal_usd"
R03_TOTAL_KEY = "cumulative_recorded_window_plus_prior_allowances_usd"
R03_PRIOR_KEY = "prior_allowances_usd"
PRIOR_USD = Decimal("26.09012499163770")
CLAIM_KEY = R03_ADDENDUM_SHA256
TEACHER_SHA256 = "11e533d592566ae367fab87b3351a18a80d655aaa81a408250267bc5ecdc72b4"
FROZEN_DIGEST = "4b5d8892dc201fb0f9cda1decca4d1350225e2a6f29f6a3b8c7ddad123b8dcfe"
PARENT_SHA256 = {
    "launch.json": "60d34d85154883538831defbe52668657a20b1ef0600d6d52e9ecddb9447c0e9",
    "coordinator.json": "e4ab48db340aa17517af242682ca7872d120f3d7b81a27fe111fe0c5746320f4",
    "prepare/result.json": "a76f905a61fe9741043c7e7d8ef76e233646f1ccdb01f3155298ef3f15957249",
    "train/result.json": "724d268ff9899df8aae7780b59242bbd5d501f0be8639f29a521aa27ed993260",
    "posteval/result.json": "5c0ed7379cbf4916997412d61ed0040ff1e184bf1694e8b661f61506205feba2",
    "prepare/wrapper.json": "18d8ecb030bd3622d9882537095f00e5f8dd3f1f262d47338b0326acd4174dfb",
    "train/wrapper.json": "ad1bff8955dfe0c55b4fa65f0566028ffc596e3bd504ac9d2057898a74f89333",
    "posteval/wrapper.json": "9be58522a3bdec275e67a9876d925025dc80841926efac84d140901dc81d0b11",
    "training/checkpoints/checkpoint-512/trainer_state.json":
        "8a7087f3853388b2319a1038f3134c119c5c08a3ac5675ec2c28e73fd7633e29",
}
LOCAL_PARENT_SHA256 = {
    "analysis.json": "049250ea103a14afd18f895dd7c144106b731353f35345fe0614c84b03a3fab2",
    "cost_addendum.json": R03_ADDENDUM_SHA256,
    "orchestration.json": "9b549c5d645a0604838d225f865ae0e8e82bf90b6c897e34d0a09c8f10d71ca8",
    "stop_receipt.json": "52f08c847a93b60110eeda942fbf6df12aec24483fa1fadfe18e0cfb2c3085b3",
}
POLICY = {"stages": list(STAGE_SECONDS), "additional_optimizer_steps": 512,
          "parent_optimizer_steps": 1024, "cumulative_optimizer_steps": 1536,
          "additional_presentations": 4096, "cumulative_presentations": 12288,
          "additional_unique_examples": 0, "cumulative_unique_examples": 3200,
          "original_corpus_rows": 3200, "cumulative_corpus_epoch": 1.0,
          "epoch3_rows": 2304, "epoch4_rows": 1792,
          "epoch3_rows_one_based_inclusive": [897, 3200],
          "epoch4_rows_one_based_inclusive": [1, 1792],
          "checkpoint_steps": CHECKPOINT_STEPS, "fresh_optimizer_and_schedule": True,
          "sequential_sampling": True, "reuse_parent_uploaded_data": True,
          "automatic_retries_or_extensions": False,
          "budget_approval": "user approved ~$41 after a ~$37 underestimate was corrected; never allow above 41"}
SCHEMA = "catan_board_fluency_extension_r04_launch/v1"
app = modal.App("catan-board-fluency-extension-r04")
COMMON: ModalCommon = dict(
    image=original.eval_image, volumes=original.VOLUMES, secrets=[original.secret],
    startup_timeout=STARTUP, retries=LIMITS["retries"],
    max_containers=LIMITS["max_containers"],
    scaledown_window=LIMITS["scaledown_seconds"])
