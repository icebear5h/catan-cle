from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

from sft.json_types import JsonDict, JsonLikeDict, JsonValue, as_dict, loads_json
from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_eval import ModalStage
from sft.launchers.board_fluency.modal_board_fluency_sft import BASE, shared
from sft.scripts.train.train_trl_catan_vision import TrainConfig, sha256_file

from ._config import (
    CHECKPOINT_STEPS,
    COMMON,
    COORDINATOR_SECONDS,
    COST_KEY,
    COST_SHA256,
    DATA_DIR,
    DEFAULT_RUN_NAME,
    PARENT,
    PARENT_ROOT,
    PARENT_RUN,
    PARENT_SHA256,
    PRIOR_USD,
    RESOURCES,
    SOURCE_FILES,
    STAGE_SECONDS,
    STARTUP,
)
from ._types import at


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


def normalized(value: object) -> JsonValue:
    return loads_json(json.dumps(value, allow_nan=False))


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"pinned file changed: {root / name}")


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(original.PROJECT_ROOT / name) for name in SOURCE_FILES}


def configuration(run_name: str, parent_config: JsonDict) -> TrainConfig:
    shared.validate_cli("", "", shared.MODEL_ID, shared.MODEL_REVISION, run_name, "", PARENT)
    if run_name == PARENT_RUN:
        raise ValueError("extension must have a fresh run name")
    approved = original.configuration(PARENT_RUN)
    if parent_config != asdict(approved):
        raise ValueError("parent configuration is not the exact approved r06 configuration")
    root = f"/runs/catan-vision-sft/{run_name}"
    # `parent_config` equals `asdict(approved)` field for field, so the approved
    # instance stands in for rebuilding a TrainConfig from its JSON.
    config = replace(approved, initial_bundle=PARENT,
                     train_jsonl=root + "/prepare/train-after-1024.jsonl",
                     output_dir=root + "/training", resume_from_checkpoint=None)
    config.validate()
    return config


def budget_plan(budget_usd: float = 15) -> JsonLikeDict:
    if not math.isfinite(budget_usd) or not 0 < budget_usd <= 15:
        raise ValueError("budget must be positive and at most the original aggregate $15")
    h200, cpu, memory = map(Decimal, ("0.001261", "0.0000131", "0.00000222"))
    rates = {kind: r["cpu"] * cpu + r["memory_gib"] * memory + (h200 if "gpu" in r else 0)
             for kind, r in RESOURCES.items()}
    gpu_seconds = sum(STAGE_SECONDS[name] + STARTUP for name in ("train", "posteval"))
    gpu = gpu_seconds * rates["gpu"]
    prepare = (STAGE_SECONDS["prepare"] + STARTUP) * rates["prepare"]
    coordinator = (COORDINATOR_SECONDS + STARTUP) * rates["coordinator"]
    reserve = Decimal("0.75")  # Reservation, scaledown and termination/control allowance.
    extension = gpu + prepare + coordinator + reserve
    upper = PRIOR_USD + extension
    if upper > Decimal(str(budget_usd)):
        raise ValueError(f"aggregate bounded compute ${upper} exceeds ${budget_usd}")
    return {"approved_usd": budget_usd, "rates_checked": "2026-09-15",
            "h200_per_second": float(h200), "cpu_core_per_second": float(cpu),
            "memory_gib_per_second": float(memory), "gpu_stage_per_second": float(rates["gpu"]),
            "gpu_seconds_including_startups": gpu_seconds, "gpu_upper_usd": float(gpu),
            "prepare_upper_usd": float(prepare), "coordinator_upper_usd": float(coordinator),
            "termination_and_control_reserve_usd": float(reserve),
            "prior_recorded_window_plus_allowances_usd": float(PRIOR_USD),
            "prior_cost_estimate_sha256": COST_SHA256, "prior_cost_total_key": COST_KEY,
            "extension_upper_usd": float(extension), "compute_upper_usd": float(upper),
            "compute_upper_usd_decimal": str(upper), "actual_billed_usd": None,
            "excluded": "unrelated jobs, storage, image storage/build charges and subscriptions",
            "policy": "fixed checked rates; no automatic retry, extension or replacement stage"}


def read_parent(root: Path) -> dict[str, JsonDict]:
    verify_hashes(root, PARENT_SHA256)
    parent = {name: shared.read_json(root / f"{name}/result.json")
              for name in ("prepare", "gate", "train", "posteval")}
    launch = parent["launch"] = shared.read_json(root / "launch.json")
    coordinator = parent["coordinator"] = shared.read_json(root / "coordinator.json")
    config = as_dict(launch["config"])
    if (launch["run_name"] != PARENT_RUN or launch["root"] != PARENT_ROOT
            or launch["data_dir"] != DATA_DIR or launch["base"] != BASE
            or launch["model_revision"] != shared.MODEL_REVISION
            or coordinator["status"] != "completed" or coordinator["budget"] != launch["budget"]):
        raise ValueError("parent launch/coordinator identity or completion differs")
    configuration(DEFAULT_RUN_NAME, config)
    if original.source_hashes() != launch["source_sha256"]:
        raise ValueError("an original launcher/model/trainer/evaluator/scorer source changed since r06")
    for stage in ("prepare", "gate", "train", "posteval"):
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
    if (state["global_step"] != 128 or state["epoch"] != 0.32
            or training["committed_steps"] != CHECKPOINT_STEPS or training["final_checkpoint"] != PARENT
            or at(training, "training", "status") != "completed"
            or at(training, "training", "metrics", "epoch") != 0.32
            or at(parent["posteval"], "result", "checkpoint") != PARENT):
        raise ValueError("parent must be the completed, trained r06 checkpoint-128")
    for split, rows, expected_hash in (
        ("train", 3200, at(launch, "inputs", "files", "train.jsonl", "sha256")),
        ("eval", 120, PARENT_SHA256["prepare/teacher120.jsonl"]),
    ):
        evidence = as_dict(as_dict(as_dict(training["training"])["dataset"])[split])
        if (evidence["rows"] != rows or evidence["source_sha256"] != expected_hash
                or evidence["input_mode"] != "text" or evidence["truncation"]):
            raise ValueError("parent training dataset evidence differs")
    return parent
