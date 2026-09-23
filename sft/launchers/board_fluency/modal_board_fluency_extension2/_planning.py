from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from sft.json_types import JsonDict, JsonLikeDict, as_dict
from sft.launchers._train_config import train_config
from sft.launchers.board_fluency import modal_board_fluency_extension as ext1
from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_eval import ModalStage
from sft.launchers.board_fluency.modal_board_fluency_sft import BASE, shared
from sft.scripts.train.train_trl_catan_vision import TrainConfig, sha256_file

from ._config import (
    COMMON,
    COORDINATOR_SECONDS,
    COST_R06_KEY,
    COST_R06_SHA256,
    DATA_DIR,
    DEFAULT_RUN_NAME,
    PARENT,
    PARENT_CHECKPOINT_STEPS,
    PARENT_ROOT,
    PARENT_RUN,
    PARENT_SHA256,
    PRIOR_R06_USD,
    PRIOR_USD,
    R01_ADDENDUM_SHA256,
    R01_RECORDED_USD,
    R01_SUBTOTAL_KEY,
    R01_TOTAL_KEY,
    R06_ROOT,
    R06_RUN,
    RESOURCES,
    SOURCE_FILES,
    STAGE_SECONDS,
    STARTUP,
    TEACHER_SHA256,
)


def resource_options(kind: str) -> ModalStage:
    resources = RESOURCES[kind]
    cores = float(resources["cpu"])
    memory = resources["memory_gib"] * 1024
    options = ModalStage(
        image=COMMON["image"], volumes=COMMON["volumes"], secrets=COMMON["secrets"],
        startup_timeout=COMMON["startup_timeout"], retries=COMMON["retries"],
        max_containers=COMMON["max_containers"], scaledown_window=COMMON["scaledown_window"],
        cpu=(cores, cores), memory=(memory, memory),
    )
    if "gpu" in resources:
        options["gpu"] = resources["gpu"]
    return options


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(original.PROJECT_ROOT / name) for name in SOURCE_FILES}


def configuration(run_name: str, parent_config: JsonDict) -> TrainConfig:
    shared.validate_cli("", "", shared.MODEL_ID, shared.MODEL_REVISION, run_name, "", PARENT)
    if run_name in (PARENT_RUN, R06_RUN, ext1.DEFAULT_RUN_NAME):
        raise ValueError("extension-r02 must have a fresh run name")
    if parent_config.get("max_steps") != 128 or parent_config.get("save_total_limit") != 4:
        raise ValueError("parent configuration is not the completed r01 128-step configuration")
    if (parent_config.get("per_device_train_batch_size") != 4
            or parent_config.get("gradient_accumulation_steps") != 2):
        raise ValueError("parent microbatch/accumulation differs")
    root = f"/runs/catan-vision-sft/{run_name}"
    config = replace(train_config(parent_config), initial_bundle=PARENT,
                     train_jsonl=root + "/prepare/train-suffix-r02.jsonl",
                     output_dir=root + "/training", max_steps=256,
                     save_steps=32, eval_steps=32, save_total_limit=8,
                     resume_from_checkpoint=None)
    config.validate()
    return config


def budget_plan(budget_usd: float = 21) -> JsonLikeDict:
    if not math.isfinite(budget_usd) or not 0 < budget_usd <= 21:
        raise ValueError("budget must be positive and at most the approved $21 ceiling (raise-the-ceiling chosen over fitting $15)")
    h200, cpu, memory = map(Decimal, ("0.001261", "0.0000131", "0.00000222"))
    rates = {kind: r["cpu"] * cpu + r["memory_gib"] * memory + (h200 if "gpu" in r else 0)
             for kind, r in RESOURCES.items()}
    gpu_seconds = sum(STAGE_SECONDS[name] + STARTUP for name in ("train", "posteval"))
    gpu = gpu_seconds * rates["gpu"]
    prepare = (STAGE_SECONDS["prepare"] + STARTUP) * rates["prepare"]
    coordinator = (COORDINATOR_SECONDS + STARTUP) * rates["coordinator"]
    reserve = Decimal("0.75")
    extension = gpu + prepare + coordinator + reserve
    upper = PRIOR_USD + extension
    if upper > Decimal(str(budget_usd)):
        raise ValueError(f"aggregate bounded compute ${upper} exceeds ${budget_usd}")
    return {"approved_usd": budget_usd, "ceiling_usd": 21,
            "user_approval": "user chose raise-the-ceiling to $21 over fitting $15; never allow above 21",
            "rates_checked": "2026-09-15",
            "h200_per_second": float(h200), "cpu_core_per_second": float(cpu),
            "memory_gib_per_second": float(memory), "gpu_stage_per_second": float(rates["gpu"]),
            "gpu_seconds_including_startups": gpu_seconds, "gpu_upper_usd": float(gpu),
            "prepare_upper_usd": float(prepare), "coordinator_upper_usd": float(coordinator),
            "termination_and_control_reserve_usd": float(reserve),
            "prior_r06_allowance_inclusive_usd": float(PRIOR_R06_USD),
            "prior_r01_recorded_usd": float(R01_RECORDED_USD),
            "prior_recorded_window_plus_allowances_usd": float(PRIOR_USD),
            "prior_r06_cost_estimate_sha256": COST_R06_SHA256, "prior_r06_cost_total_key": COST_R06_KEY,
            "prior_r01_cost_addendum_sha256": R01_ADDENDUM_SHA256,
            "prior_r01_total_key": R01_TOTAL_KEY, "prior_r01_subtotal_key": R01_SUBTOTAL_KEY,
            "extension_upper_usd": float(extension), "compute_upper_usd": float(upper),
            "compute_upper_usd_decimal": str(upper), "actual_billed_usd": None,
            "excluded": "unrelated jobs, storage, image storage/build charges and subscriptions",
            "policy": "fixed checked rates; no automatic retry, extension or replacement stage"}


def read_parent(root: Path) -> dict[str, JsonDict]:
    ext1.verify_hashes(root, PARENT_SHA256)
    parent = {name: shared.read_json(root / f"{name}/result.json")
              for name in ("prepare", "train", "posteval")}
    launch = parent["launch"] = shared.read_json(root / "launch.json")
    coordinator = parent["coordinator"] = shared.read_json(root / "coordinator.json")
    config = as_dict(launch["config"])
    if (launch["run_name"] != PARENT_RUN or launch["root"] != PARENT_ROOT
            or launch["data_dir"] != DATA_DIR or launch["base"] != BASE
            or launch["model_revision"] != shared.MODEL_REVISION
            or coordinator["status"] != "completed" or coordinator["budget"] != launch["budget"]):
        raise ValueError("parent launch/coordinator identity or completion differs")
    if (config["max_steps"] != 128 or config["save_total_limit"] != 4
            or config["save_steps"] != 32 or config["eval_steps"] != 32
            or config["initial_bundle"] != R06_ROOT + "/training/checkpoints/checkpoint-128"
            or config["output_dir"] != PARENT_ROOT + "/training"
            or config["resume_from_checkpoint"] is not None):
        raise ValueError("parent is not the exact completed r01 launch configuration")
    current = source_hashes()
    for name, expected in as_dict(launch["source_sha256"]).items():
        if current.get(name) != expected:
            raise ValueError(f"a parent launcher/model/trainer/evaluator/scorer source changed since r01: {name}")
    configuration(DEFAULT_RUN_NAME, config)
    for stage in ("prepare", "train", "posteval"):
        entry = as_dict(as_dict(coordinator["stages"])[stage])
        receipt = parent[stage]
        wrapper = shared.read_json(root / stage / "wrapper.json")
        if (entry["status"] != "completed" or entry["result"] != receipt
                or receipt["status"] != "completed" or receipt["stage"] != stage
                or receipt["launch_sha256"] != shared.digest(launch)
                or not receipt["started_at"] or not receipt["ended_at"]
                or wrapper["status"] != "completed" or wrapper["call_id"] != entry["call_id"]):
            raise ValueError(f"parent {stage} did not complete for its exact launch")
    training = as_dict(parent["train"]["result"])
    state = shared.read_json(root / "training/checkpoints/checkpoint-128/trainer_state.json")
    if (state["global_step"] != 128
            or training["committed_steps"] != PARENT_CHECKPOINT_STEPS
            or training["final_checkpoint"] != PARENT
            or as_dict(training["training"])["status"] != "completed"
            or as_dict(parent["posteval"]["result"])["checkpoint"] != PARENT):
        raise ValueError("parent must be the completed, trained r01 checkpoint-128")
    for split, rows, expected_hash in (
        ("train", 2176, as_dict(launch["suffix"])["sha256"]),
        ("eval", 120, TEACHER_SHA256),
    ):
        evidence = as_dict(as_dict(as_dict(training["training"])["dataset"])[split])
        if (evidence["rows"] != rows or evidence["source_sha256"] != expected_hash
                or evidence["input_mode"] != "text" or evidence["truncation"]):
            raise ValueError("parent training dataset evidence differs")
    return parent
