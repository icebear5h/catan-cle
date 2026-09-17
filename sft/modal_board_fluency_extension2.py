"""Opt-in 256-update r02 extension of r01; local dry runs never contact Modal.

MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2 \
    .venv/bin/python -B -m sft.modal_board_fluency_extension2 \
    --run-name board-fluency-extension-20260915-r02 --budget-usd 21
Add --execute to reserve and supervise one run, or --stop to stop its entire app.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

import modal
import torch

from sft import modal_board_fluency_extension as ext1
from sft import modal_board_fluency_sft as original
from sft.modal_board_fluency_sft import (
    BASE, OFFLINE, CheckpointCallback, check_deadline, check_manifest,
    deadline_alarm, eval_panel, evaluator, load_eval, progress, rows_at, shared,
    trainer, visual_file_digest,
)
from sft.scripts.train_trl_catan_vision import (
    TrainConfig, _message_pair, encode_text_pair, load_token_inventory,
    sha256_file, write_json_atomic,
)


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
SOURCE_FILES = (*original.SOURCE_FILES, "sft/modal_board_fluency_extension.py",
                "sft/modal_board_fluency_extension2.py")
STAGE_SECONDS = {"prepare": 300, "train": 3300, "posteval": 900}
STARTUP, COORDINATOR_SECONDS, ABSOLUTE_SECONDS = 300, 6000, 5400
RESOURCES = {"gpu": {"gpu": "H200", "cpu": 16, "memory_gib": 128},
             "prepare": {"cpu": 4, "memory_gib": 16},
             "coordinator": {"cpu": 1, "memory_gib": 2}}
LIMITS = {"stage_seconds": STAGE_SECONDS, "startup_seconds": STARTUP,
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
COMMON = dict(image=original.eval_image, volumes=original.VOLUMES, secrets=[original.secret],
              startup_timeout=STARTUP, retries=LIMITS["retries"],
              max_containers=LIMITS["max_containers"], scaledown_window=LIMITS["scaledown_seconds"])


def resource_options(kind: str) -> dict:
    resources = RESOURCES[kind]
    return {**COMMON, "cpu": (float(resources["cpu"]), float(resources["cpu"])),
            "memory": (resources["memory_gib"] * 1024,) * 2,
            **({"gpu": resources["gpu"]} if "gpu" in resources else {})}


def source_hashes() -> dict:
    return {name: sha256_file(original.PROJECT_ROOT / name) for name in SOURCE_FILES}


def configuration(run_name: str, parent_config: dict) -> TrainConfig:
    shared.validate_cli("", "", shared.MODEL_ID, shared.MODEL_REVISION, run_name, "", PARENT)
    if run_name in (PARENT_RUN, R06_RUN, ext1.DEFAULT_RUN_NAME):
        raise ValueError("extension-r02 must have a fresh run name")
    if parent_config.get("max_steps") != 128 or parent_config.get("save_total_limit") != 4:
        raise ValueError("parent configuration is not the completed r01 128-step configuration")
    if (parent_config.get("per_device_train_batch_size") != 4
            or parent_config.get("gradient_accumulation_steps") != 2):
        raise ValueError("parent microbatch/accumulation differs")
    root = f"/runs/catan-vision-sft/{run_name}"
    config = replace(TrainConfig(**parent_config), initial_bundle=PARENT,
                     train_jsonl=root + "/prepare/train-suffix-r02.jsonl",
                     output_dir=root + "/training", max_steps=256,
                     save_steps=32, eval_steps=32, save_total_limit=8,
                     resume_from_checkpoint=None)
    config.validate()
    return config


def budget_plan(budget_usd: float = 21) -> dict:
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


def read_parent(root: Path) -> dict:
    ext1.verify_hashes(root, PARENT_SHA256)
    parent = {name: shared.read_json(root / f"{name}/result.json")
              for name in ("prepare", "train", "posteval")}
    launch = parent["launch"] = shared.read_json(root / "launch.json")
    coordinator = parent["coordinator"] = shared.read_json(root / "coordinator.json")
    if (launch["run_name"] != PARENT_RUN or launch["root"] != PARENT_ROOT
            or launch["data_dir"] != DATA_DIR or launch["base"] != BASE
            or launch["model_revision"] != shared.MODEL_REVISION
            or coordinator["status"] != "completed" or coordinator["budget"] != launch["budget"]):
        raise ValueError("parent launch/coordinator identity or completion differs")
    if (launch["config"]["max_steps"] != 128 or launch["config"]["save_total_limit"] != 4
            or launch["config"]["save_steps"] != 32 or launch["config"]["eval_steps"] != 32
            or launch["config"]["initial_bundle"] != R06_ROOT + "/training/checkpoints/checkpoint-128"
            or launch["config"]["output_dir"] != PARENT_ROOT + "/training"
            or launch["config"]["resume_from_checkpoint"] is not None):
        raise ValueError("parent is not the exact completed r01 launch configuration")
    current = source_hashes()
    for name, expected in launch["source_sha256"].items():
        if current.get(name) != expected:
            raise ValueError(f"a parent launcher/model/trainer/evaluator/scorer source changed since r01: {name}")
    configuration(DEFAULT_RUN_NAME, launch["config"])
    for stage in ("prepare", "train", "posteval"):
        entry, receipt = coordinator["stages"][stage], parent[stage]
        wrapper = shared.read_json(root / stage / "wrapper.json")
        if (entry["status"] != "completed" or entry["result"] != receipt
                or receipt["status"] != "completed" or receipt["stage"] != stage
                or receipt["launch_sha256"] != shared.digest(launch)
                or not receipt["started_at"] or not receipt["ended_at"]
                or wrapper["status"] != "completed" or wrapper["call_id"] != entry["call_id"]):
            raise ValueError(f"parent {stage} did not complete for its exact launch")
    training = parent["train"]["result"]
    state = shared.read_json(root / "training/checkpoints/checkpoint-128/trainer_state.json")
    if (state["global_step"] != 128
            or training["committed_steps"] != PARENT_CHECKPOINT_STEPS
            or training["final_checkpoint"] != PARENT
            or training["training"]["status"] != "completed"
            or parent["posteval"]["result"]["checkpoint"] != PARENT):
        raise ValueError("parent must be the completed, trained r01 checkpoint-128")
    for split, rows, expected_hash in (
        ("train", 2176, launch["suffix"]["sha256"]),
        ("eval", 120, TEACHER_SHA256),
    ):
        evidence = training["training"]["dataset"][split]
        if (evidence["rows"] != rows or evidence["source_sha256"] != expected_hash
                or evidence["input_mode"] != "text" or evidence["truncation"]):
            raise ValueError("parent training dataset evidence differs")
    return parent


def suffix_identity(path: Path, inputs: dict) -> tuple[bytes, dict]:
    """Slice raw lines, never serialize or regenerate the original examples."""
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    if (len(lines) != 3200 or any(not line.strip() for line in lines)
            or hashlib.sha256(payload).hexdigest() != inputs["files"]["train.jsonl"]["sha256"]):
        raise ValueError("original raw training lines changed")
    rows = [json.loads(line) for line in lines]
    if [row["id"] for row in rows] != inputs["panels"]["train"]["ids"]:
        raise ValueError("original ordered training IDs changed")
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        canonical = hashlib.sha256(json.dumps(metadata["target"]["state"], sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        if metadata["state_sha256"] != canonical or metadata["row_position"] != index:
            raise ValueError("canonical state hash or original row position differs")

    def contract(items):
        return {**original.row_contract(items),
                "ordered_state_sha256": [row["metadata"]["state_sha256"] for row in items]}

    consumed_ids = {row["id"] for row in rows[:2048]}
    consumed_states = {row["metadata"]["state_sha256"] for row in rows[:2048]}
    if len(consumed_ids) != 2048 or len(inputs["panels"]["train"]["ids"]) != 3200:
        raise ValueError("already-consumed rows 1-2048 identity differs")
    fresh_rows, repeat_rows = rows[2048:3200], rows[0:896]
    if len(fresh_rows) != 1152 or len(repeat_rows) != 896:
        raise ValueError("fresh/repeat partition differs")
    fresh, repeat = contract(fresh_rows), contract(repeat_rows)
    if set(fresh["ids"]) & consumed_ids:
        raise ValueError("fresh part repeats already-consumed example IDs 1-2048")
    if set(fresh["state_sha256"]) & consumed_states:
        raise ValueError("fresh part repeats already-consumed canonical states 1-2048")
    fresh_bytes = b"".join(lines[2048:3200])
    repeat_bytes = b"".join(lines[0:896])
    suffix = fresh_bytes + repeat_bytes
    suffix_lines = suffix.splitlines(keepends=True)
    if len(suffix_lines) != 2048 or suffix_lines[:1152] != lines[2048:3200]:
        raise ValueError("fresh part bytes differ from original rows 2049-3200")
    if suffix_lines[1152:] != lines[0:896]:
        raise ValueError("repeat part is not byte-identical to original rows 1-896")
    suffix_rows = [json.loads(line) for line in suffix_lines]
    if ([row["id"] for row in suffix_rows[:1152]] != [row["id"] for row in fresh_rows]
            or [row["id"] for row in suffix_rows[1152:]] != [row["id"] for row in repeat_rows]):
        raise ValueError("suffix fresh/repeat IDs differ from original rows")
    consumed = contract(suffix_rows)
    return suffix, {"sha256": hashlib.sha256(suffix).hexdigest(), "bytes": len(suffix),
                    "rows": 2048, "fresh_rows": 1152, "repeat_rows": 896,
                    "fresh_start_row_zero_based": 2048, "repeat_start_row_zero_based": 0,
                    "source_sha256": inputs["files"]["train.jsonl"]["sha256"],
                    "fresh": fresh, "repeat": repeat, "consumed": consumed,
                    "already_consumed_rows_one_based_inclusive": [1, 2048],
                    "fresh_rows_one_based_inclusive": [2049, 3200],
                    "repeat_rows_one_based_inclusive": [1, 896],
                    "cumulative_presentations": 4096, "cumulative_unique_examples": 3200,
                    "cumulative_corpus_epoch": 1.0,
                    "unique_examples_basis": "row IDs; fresh part ID- and canonical-state-disjoint from rows 1-2048; repeat part byte- and ID-identical to rows 1-896"}


def retained_baselines(parent: dict, root: Path, data: Path, review: Path) -> dict:
    """Strictly rescore the complete r01 post panels against unchanged gold/meta/IDs."""
    inputs = parent["launch"]["inputs"]
    baselines = parent["posteval"]["result"]["panels"]
    if set(baselines) != {"review", "validation_eval"}:
        raise ValueError("both complete r01 post panels are required")
    for panel, (count, correct) in {"review": (200, 57), "validation_eval": (190, 96)}.items():
        saved = baselines[panel]
        output = root / "posteval" / panel
        check_manifest(output, saved["files"])
        records = rows_at(output / "records.jsonl")
        gold = rows_at(review if panel == "review" else data / "validation_eval.jsonl")
        by_id = {row["id"]: row for row in gold}
        if (saved["rows"] != count or saved["correct"] != correct
                or saved["output_dir"] != PARENT_ROOT + f"/posteval/{panel}"
                or len(records) != count
                or Counter(row.get("id") for row in records) != Counter(inputs["panels"][panel]["ids"])):
            raise ValueError(f"{panel}: incomplete/duplicate/wrong retained baseline IDs")
        for record in records:
            row = by_id[record["id"]]
            metadata = {**evaluator.evaluation_metadata(row, image_variant="original"), "input_mode": "text"}
            if (record["expected"] != evaluator.expected_text(row) or record["metadata"] != metadata
                    or not isinstance(record["response"], str) or record.get("candidate_score") is not None
                    or record["score"] != evaluator.score_response(record["expected"], record["response"], metadata=metadata)
                    or record["score"].get("scoring") != metadata["schema"]
                    or type(record["score"].get("correct")) is not bool):
                raise ValueError(f"{panel}: retained prediction/gold/metadata/strict score differs")
        summary = shared.read_json(output / "summary.json")
        conditions = {"adapter_dir": PARENT, "model_id": BASE, "model_revision": shared.MODEL_REVISION,
                      "eval_jsonl": DATA_DIR + f"/{panel}.jsonl", "input_mode": "text", "bits": 16,
                      "max_sequence_length": 4096, "batch_size": 16, "max_new_tokens": 512,
                      "long_max_new_tokens": 512, "candidate_scoring": False,
                      "reasoning_enabled": False, "image_variant": "original", "truncation": False}
        recomputed = evaluator.summarize(records)
        conditions.update({key: value for key, value in recomputed.items() if key != "generated_at"})
        if (any(summary.get(key) != value for key, value in conditions.items())
                or recomputed["correct"] != correct or saved["exact_accuracy"] != correct / count
                or summary["precision"]["preserve_visual_fp32"] is not True):
            raise ValueError(f"{panel}: retained inference conditions/summary disagree")
    return baselines


def inspect_original(parent: dict, root: Path, data: Path, review: Path, teacher: Path) -> tuple[bytes, dict, dict]:
    inputs = parent["launch"]["inputs"]
    if original.inspect_data(data, review) != inputs:
        raise ValueError("original data/inventory/review manifest changed")
    if sha256_file(teacher) != TEACHER_SHA256:
        raise ValueError("teacher120 hash differs")
    teacher_rows = rows_at(teacher)
    validation = {row["id"]: row for row in rows_at(data / "validation_eval.jsonl")}
    if teacher_rows != [validation[row_id] for row_id in inputs["teacher_ids"]]:
        raise ValueError("teacher120 is not the exact original fixed panel")
    suffix, identity = suffix_identity(data / "train.jsonl", inputs)
    return suffix, identity, retained_baselines(parent, root, data, review)


def teacher_path_for(remote: bool) -> Path:
    return Path(R06_ROOT) / "prepare/teacher120.jsonl" if remote else LOCAL_R06 / "prepare/teacher120.jsonl"


def build_plan(run_name: str, budget_usd: float = 21) -> dict:
    parent = read_parent(LOCAL_PARENT)
    config = configuration(run_name, parent["launch"]["config"])
    if (LOCAL_ROOT / run_name).exists():
        raise FileExistsError("local run name already used")
    ext1.verify_hashes(LOCAL_PARENT, LOCAL_PARENT_SHA256)
    if shared.read_json(LOCAL_PARENT / "analysis.json")["checks_passed"] is not True:
        raise ValueError("parent offline analysis was not successful")
    _, suffix, baselines = inspect_original(parent, LOCAL_PARENT, original.DATASET, original.REVIEW,
                                           LOCAL_R06 / "prepare/teacher120.jsonl")
    plan = {"schema": SCHEMA, "run_name": run_name,
            "created_at": shared.now(), "reservation_id": uuid.uuid4().hex,
            "root": str(Path(config.output_dir).parent), "data_dir": DATA_DIR,
            "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
            "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
            "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
            "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
            "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
            "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "suffix": suffix, "saved_baselines": baselines,
            "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": CLAIM_KEY},
            "cost_lineage": {"r06_sha256": COST_R06_SHA256, "r06_total_key": COST_R06_KEY,
                             "r06_receipt_utf8": (LOCAL_R06 / "cost_estimate.json").read_text(),
                             "r01_sha256": R01_ADDENDUM_SHA256, "r01_total_key": R01_TOTAL_KEY,
                             "r01_subtotal_key": R01_SUBTOTAL_KEY,
                             "r01_receipt_utf8": (LOCAL_PARENT / "cost_addendum.json").read_text(),
                             "carry_forward_usd_decimal": str(PRIOR_USD)},
            "budget": budget_plan(budget_usd), "limits": LIMITS, "policy": POLICY}
    return ext1.normalized(plan)


def verify_plan(plan: dict) -> None:
    """Repeat offline admission locally; the same guards use old mounts on the server."""
    if ext1.normalized(plan) != plan:
        raise ValueError("plan must round-trip through strict JSON unchanged")
    remote = os.environ.get("MODAL_IS_REMOTE") == "1"
    root = Path(PARENT_ROOT) if remote else LOCAL_PARENT
    parent = read_parent(root)
    config = configuration(plan["run_name"], parent["launch"]["config"])
    expected = {"schema": SCHEMA, "root": str(Path(config.output_dir).parent),
                "data_dir": DATA_DIR, "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
                "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
                "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
                "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
                "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
                "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "limits": LIMITS, "policy": POLICY,
                "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": CLAIM_KEY},
                "budget": budget_plan(plan["budget"]["approved_usd"])}
    if any(plan[key] != value for key, value in expected.items()):
        raise ValueError("plan configuration/identity/source/budget/resource caps differ")
    if plan["budget_claim"]["key"] == ext1.COST_SHA256:
        raise ValueError("r02 must use a NEW budget-lineage claim key, not r01's claim key")
    if (len(plan["reservation_id"]) != 32
            or uuid.UUID(hex=plan["reservation_id"]).hex != plan["reservation_id"]):
        raise ValueError("invalid immutable reservation identity")
    cost = plan["cost_lineage"]
    if (cost["r06_sha256"] != COST_R06_SHA256 or cost["r06_total_key"] != COST_R06_KEY
            or hashlib.sha256(cost["r06_receipt_utf8"].encode()).hexdigest() != COST_R06_SHA256
            or cost["r01_sha256"] != R01_ADDENDUM_SHA256 or cost["r01_total_key"] != R01_TOTAL_KEY
            or cost["r01_subtotal_key"] != R01_SUBTOTAL_KEY
            or hashlib.sha256(cost["r01_receipt_utf8"].encode()).hexdigest() != R01_ADDENDUM_SHA256
            or cost["carry_forward_usd_decimal"] != str(PRIOR_USD)):
        raise ValueError("parent cost lineage differs")
    r06 = json.loads(cost["r06_receipt_utf8"], parse_float=Decimal)
    r01 = json.loads(cost["r01_receipt_utf8"], parse_float=Decimal)
    if (r06["run_name"] != R06_RUN or r06["totals"][COST_R06_KEY] != PRIOR_R06_USD):
        raise ValueError("r06 cost carry-forward is not the pinned allowance-inclusive total")
    if (r01["run_name"] != PARENT_RUN or r01[R01_TOTAL_KEY] != PRIOR_USD
            or r01[R01_SUBTOTAL_KEY] != R01_RECORDED_USD
            or r01["prior_allowance_inclusive_usd"] != PRIOR_R06_USD):
        raise ValueError("r01 cost carry-forward is not the pinned recorded/total lineage")
    if PRIOR_R06_USD + R01_RECORDED_USD != PRIOR_USD:
        raise ValueError("carry-forward arithmetic differs")
    for key in ("h200_per_second", "cpu_core_per_second", "memory_gib_per_second", "gpu_stage_per_second"):
        if (not math.isclose(float(r06["rates"][key + "_usd"]), plan["budget"][key], rel_tol=0, abs_tol=1e-15)
                or not math.isclose(parent["launch"]["budget"][key], plan["budget"][key], rel_tol=0, abs_tol=1e-15)):
            raise ValueError("extension rates differ from the parent cost lineage")
    if not remote:
        ext1.verify_hashes(root, LOCAL_PARENT_SHA256)
    data, review = (Path(DATA_DIR), Path(DATA_DIR) / "review.jsonl") if remote else (original.DATASET, original.REVIEW)
    _, suffix, baselines = inspect_original(parent, root, data, review, teacher_path_for(remote))
    if plan["suffix"] != suffix or plan["saved_baselines"] != baselines:
        raise ValueError("suffix identity or complete retained baseline differs")
    original.require_dependencies()


def verify_initialization(config: TrainConfig, prepared: dict, report: dict) -> None:
    if Path(report["path"]).resolve(strict=True) != Path(PARENT).resolve(strict=True):
        raise ValueError("trainer did not load the trained r01 checkpoint-128")
    for key, name in {"adapter_sha256": "adapter_model.safetensors", "visual_sha256": trainer.VISUAL_STATE_FILE,
                      "parent_training_config_sha256": trainer.RUN_CONFIG_FILE}.items():
        if report[key] != prepared["parent_files"][name]["sha256"]:
            raise ValueError(f"actual initial-bundle content differs: {key}")
    expected = {"input_mode": "text", "source_input_mode": "text", "lora_rank": 16, "lora_alpha": 32,
                "optimizer_state_restored": False, "scheduler_state_restored": False, "rng_state_restored": False}
    if any(report[key] != value for key, value in expected.items()) or config.resume_from_checkpoint is not None:
        raise ValueError("extension requires the trained text bundle and fresh optimizer/schedule/RNG")


class Extension2Callback(CheckpointCallback):
    def __init__(self, config: TrainConfig, deadline: float, prepared: dict):
        super().__init__(config, deadline)
        self.prepared = prepared

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if (args.world_size != 1 or args.train_sampling_strategy != "sequential" or args.shuffle_dataset
                or args.max_steps != 256 or args.per_device_train_batch_size != 4
                or args.gradient_accumulation_steps != 2 or args.save_total_limit != 8):
            raise ValueError("actual trainer sampling/update/checkpoint retention differs")
        output = Path(self.config.output_dir)
        verify_initialization(self.config, self.prepared, shared.read_json(output / trainer.INITIAL_BUNDLE_FILE))
        dataset = shared.read_json(output / trainer.DATASET_REPORT_FILE)
        for split, path, rows, expected in (
            ("train", self.config.train_jsonl, 2048, self.prepared["suffix"]["sha256"]),
            ("eval", self.config.eval_jsonl, 120, self.prepared["teacher_sha256"]),
        ):
            actual = dataset[split]
            if (Path(actual["source"]).resolve(strict=True) != Path(path).resolve(strict=True)
                    or actual["source_sha256"] != expected or actual["rows"] != rows
                    or actual["input_mode"] != "text" or actual["truncation"]):
                raise ValueError("actual trainer dataset differs from admitted sequential data")


def prepare_work(plan: dict, deadline: float) -> dict:
    parent = read_parent(Path(PARENT_ROOT))
    prior = parent["prepare"]["result"]
    inventory = load_token_inventory(plan["config"]["token_inventory"])
    bundle, _ = shared.volume_bundle(PARENT)
    parent_files = ext1.full_checkpoint_manifest(bundle)
    tokenizer, checkpoint = shared.adapter_preflight(bundle, inventory, shared.MODEL_ID, shared.MODEL_REVISION)
    saved_config = shared.read_json(bundle / trainer.RUN_CONFIG_FILE)
    semantic = parent["train"]["result"]["training"]["reload_validation"]["semantic_tokens"]
    if (saved_config != parent["launch"]["config"] or saved_config["input_mode"] != "text"
            or checkpoint["saved_model_id"] != BASE or checkpoint["lora_rank"] != 16
            or checkpoint["lora_alpha"] != 32 or checkpoint["adapter_config"]["lora_dropout"] != 0.05
            or checkpoint["semantic_tokens"]["token_ids"] != semantic["token_ids"]
            or checkpoint["semantic_tokens"]["tokens"] != semantic["tokens"]):
        raise ValueError("trained parent rank/alpha/dropout/input/base/atlas identity differs")
    base_files = original.cached_base_files()
    if set(base_files) != set(prior["base_files"]):
        raise ValueError("cached native base file set changed")
    check_manifest(Path(BASE), prior["base_files"])
    base_audit = shared.base_preflight(Path(BASE), base_files, bundle, inventory, checkpoint)
    if base_audit != prior["base_audit"]:
        raise ValueError("cached base metadata/shards/header/architecture audit changed")
    if shared.read_json(bundle / "trainer_state.json")["global_step"] != 128:
        raise ValueError("trained parent global step differs")
    frozen = visual_file_digest(bundle / trainer.VISUAL_STATE_FILE)
    if frozen != parent["train"]["result"]["frozen_visual_file_tensor_sha256"] or frozen != FROZEN_DIGEST:
        raise ValueError("parent frozen visual tensors differ from r01 training")
    suffix, identity, baselines = inspect_original(parent, Path(PARENT_ROOT), Path(DATA_DIR),
                                                  Path(DATA_DIR) / "review.jsonl",
                                                  Path(R06_ROOT) / "prepare/teacher120.jsonl")
    if identity != plan["suffix"] or baselines != plan["saved_baselines"]:
        raise ValueError("CPU suffix/baseline admission differs")
    lengths = {"train_suffix": ext1.text_boundaries(tokenizer, [json.loads(line) for line in suffix.splitlines()],
                                                   generation=False, deadline=deadline)}
    for panel, path in {"teacher120": plan["config"]["eval_jsonl"],
                        "review": DATA_DIR + "/review.jsonl", "validation_eval": DATA_DIR + "/validation_eval.jsonl"}.items():
        lengths[panel] = ext1.text_boundaries(tokenizer, rows_at(Path(path)), generation=panel != "teacher120", deadline=deadline)
    del tokenizer
    check_deadline(deadline)
    destination = Path(plan["config"]["train_jsonl"])
    with destination.open("xb") as handle:
        handle.write(suffix)
        handle.flush()
        os.fsync(handle.fileno())
    suffix_files = shared.file_manifest(destination.parent, [destination.name])
    if suffix_files[destination.name] != {key: identity[key] for key in ("bytes", "sha256")}:
        raise ValueError("exclusive raw-line suffix write differs")
    check_manifest(bundle, parent_files)
    return {"checkpoint": checkpoint, "parent_files": parent_files,
            "parent_manifest_provenance": "full current r01 checkpoint manifest established by this CPU prepare; no historical r06 adapter hash claimed",
            "frozen_visual_file_tensor_sha256": frozen, "base_files": prior["base_files"],
            "base_audit": base_audit, "base_manifest_provenance": "r01 metadata hashes and shard sizes, rechecked tensor headers/index",
            "suffix_files": suffix_files, "suffix": identity, "baselines": baselines,
            "teacher_sha256": TEACHER_SHA256,
            "token_lengths": lengths, "prior_token_lengths": prior["token_lengths"],
            "token_boundary_provenance": "actual native trained-parent tokenizer, exact original raw rows"}


def prepared_for(plan: dict) -> dict:
    prepared = ext1.completed_stage(plan, "prepare")
    check_manifest(Path(PARENT), prepared["parent_files"])
    check_manifest(Path(BASE), prepared["base_files"])
    check_manifest(Path(plan["config"]["train_jsonl"]).parent, prepared["suffix_files"])
    if (prepared["suffix"] != plan["suffix"] or prepared["baselines"] != plan["saved_baselines"]
            or sha256_file(Path(plan["config"]["eval_jsonl"])) != prepared["teacher_sha256"]):
        raise ValueError("prepared suffix/teacher/baseline identity changed")
    return prepared


def train_work(plan: dict, deadline: float) -> dict:
    prepared = prepared_for(plan)
    config = TrainConfig(**plan["config"])
    if Path(config.output_dir).exists():
        raise FileExistsError("training output exists; retries/resumption are not authorized")
    callback = Extension2Callback(config, deadline, prepared)
    progress("extension-r02: trained r01 checkpoint, fresh optimizer, rows 2049-3200 plus repeat 1-896, 256 updates")
    result = trainer.run_training(config, extra_callbacks=[callback])
    if result["status"] != "completed" or callback.saved_steps != CHECKPOINT_STEPS:
        raise RuntimeError("extension did not commit exactly checkpoints 32/64/96/128/160/192/224/256")
    verify_initialization(config, prepared, result["initial_bundle"])
    for step in CHECKPOINT_STEPS:
        saved = Path(config.output_dir) / "checkpoints" / f"checkpoint-{step}"
        shared.volume_bundle(str(saved))
        if shared.read_json(saved / "trainer_state.json")["global_step"] != step:
            raise ValueError("retained checkpoint global step differs")
    final = Path(config.output_dir) / "checkpoints/checkpoint-256"
    frozen = visual_file_digest(final / trainer.VISUAL_STATE_FILE)
    if frozen != prepared["frozen_visual_file_tensor_sha256"]:
        raise ValueError("extension changed the parent's canonical frozen visual tensors")
    check_manifest(Path(PARENT), prepared["parent_files"])
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("source manifest changed during training")
    return {"training": result, "final_checkpoint": str(final), "committed_steps": callback.saved_steps,
            "final_checkpoint_files": ext1.full_checkpoint_manifest(final),
            "frozen_visual_file_tensor_sha256": frozen, "additional_updates": 256, "cumulative_updates": 512,
            "consumed": plan["suffix"]["consumed"], "fresh": plan["suffix"]["fresh"], "repeat": plan["suffix"]["repeat"],
            "cumulative_unique_examples": 3200, "cumulative_corpus_epoch": 1.0, "cumulative_presentations": 4096,
            "consumption_basis": "verified single-process sequential sampler, exact suffix hash, 256 completed batch8 updates"}


def posteval_work(plan: dict, deadline: float) -> dict:
    prepared = prepared_for(plan)
    training = ext1.completed_stage(plan, "train")
    checkpoint = plan["config"]["output_dir"] + "/checkpoints/checkpoint-256"
    if training["final_checkpoint"] != checkpoint:
        raise ValueError("post-evaluation checkpoint differs")
    check_manifest(Path(checkpoint), training["final_checkpoint_files"])
    model, tokenizer, evidence = load_eval(plan, checkpoint)
    panels = {}
    for panel in ("review", "validation_eval"):
        check_deadline(deadline)
        panels[panel] = eval_panel(plan, model, tokenizer, evidence, checkpoint, panel,
                                  Path(plan["root"]) / "posteval" / panel)
    check_manifest(Path(PARENT), prepared["parent_files"])
    return {"checkpoint": checkpoint, "panels": panels, "saved_baselines": prepared["baselines"]}


def worker(plan: dict, stage: str, deadline: float) -> None:
    os.environ.update(OFFLINE)
    torch.set_num_threads(RESOURCES["prepare" if stage == "prepare" else "gpu"]["cpu"])
    directory = Path(plan["root"]) / stage
    receipt = {"status": "running", "stage": stage, "launch_sha256": shared.digest(plan),
               "started_at": shared.now(), "inner_deadline_unix": deadline}
    try:
        write_json_atomic(directory / "result.json", receipt)
        original.sft_runs.commit()
        with deadline_alarm(deadline):
            if shared.read_json(Path(plan["root"]) / "launch.json") != plan:
                raise ValueError("extension reservation changed")
            verify_plan(plan)
            receipt["dependencies"] = trainer.assert_runtime_versions()
            function = {"prepare": prepare_work, "train": train_work, "posteval": posteval_work}[stage]
            receipt.update(result=function(plan, deadline), status="completed")
            if source_hashes() != plan["source_sha256"]:
                raise ValueError("source manifest changed during stage")
    except BaseException as exc:
        receipt.update(status="failed", error=shared.error_record(exc))
        raise
    finally:
        receipt["ended_at"] = shared.now()
        write_json_atomic(directory / "result.json", receipt)
        original.sft_runs.commit()


def bounded_stage(plan: dict, stage: str, deadline: float) -> dict:
    hard_deadline = min(deadline - 30, time.time() + STAGE_SECONDS[stage] - 30)
    inner_deadline = hard_deadline - 45
    check_deadline(inner_deadline)
    for volume in original.VOLUMES.values():
        volume.reload()
    directory = Path(plan["root"]) / stage
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-m", "sft.modal_board_fluency_extension2", "--worker", stage,
               "--plan", plan["root"] + "/launch.json", "--deadline", str(inner_deadline)]
    wrapper = {"status": "running", "stage": stage, "launch_sha256": shared.digest(plan),
               "started_at": shared.now(), "command": command,
               "call_id": modal.current_function_call_id(), "hard_deadline_unix": hard_deadline}
    process, error = None, None
    log_path = Path("/tmp") / f"board-fluency-extension-r02-{plan['reservation_id']}-{stage}.log"
    try:
        write_json_atomic(directory / "wrapper.json", wrapper)
        original.sft_runs.commit()
        with log_path.open("x", buffering=1) as log:
            process = subprocess.Popen(command, env={**os.environ, **OFFLINE}, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            relay_errors = []

            def forward_logs():
                try:
                    for line in process.stdout:
                        log.write(line)
                        print(line, end="", flush=True)
                except BaseException as exc:
                    relay_errors.append(shared.error_record(exc))

            relay = threading.Thread(target=forward_logs, daemon=True)
            try:
                relay.start()
                code = process.wait(timeout=max(0.001, hard_deadline - time.time()))
                if code:
                    raise RuntimeError(f"{stage} worker exited {code}; see {directory}/worker.log")
            finally:
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                if relay.ident is not None:
                    relay.join(timeout=5)
            if relay.is_alive() or relay_errors:
                raise RuntimeError(f"worker log relay incomplete: {relay_errors}")
        original.sft_runs.reload()
        result = shared.read_json(directory / "result.json")
        if (result["status"] != "completed" or result["stage"] != stage
                or result["launch_sha256"] != shared.digest(plan)):
            raise RuntimeError(f"{stage} worker did not complete this launch")
        return result
    except BaseException as exc:
        error = shared.error_record(exc)
        raise
    finally:
        try:
            original.sft_runs.reload()
            if log_path.is_file():
                shutil.copyfile(log_path, directory / "worker.log")
        except BaseException as exc:
            error = shared.error_record(exc)
            raise
        finally:
            wrapper.update(status="failed" if error else "completed", error=error, ended_at=shared.now())
            if error:
                path = directory / "result.json"
                receipt = shared.read_json(path) if path.is_file() else {
                    "stage": stage, "launch_sha256": shared.digest(plan), "started_at": wrapper["started_at"]}
                if receipt.get("status") != "completed":
                    receipt.update(status="failed", error=error, ended_at=shared.now())
                    write_json_atomic(path, receipt)
            write_json_atomic(directory / "wrapper.json", wrapper)
            original.sft_runs.commit()


@app.function(**resource_options("prepare"), timeout=STAGE_SECONDS["prepare"])
def prepare_cpu(plan: dict, deadline: float) -> dict:
    return bounded_stage(plan, "prepare", deadline)


@app.function(**resource_options("gpu"), timeout=STAGE_SECONDS["train"])
def train_h200(plan: dict, deadline: float) -> dict:
    return bounded_stage(plan, "train", deadline)


@app.function(**resource_options("gpu"), timeout=STAGE_SECONDS["posteval"])
def posteval_h200(plan: dict, deadline: float) -> dict:
    return bounded_stage(plan, "posteval", deadline)


@app.function(**resource_options("coordinator"), timeout=LIMITS["reservation_seconds"])
def reserve(plan: dict) -> None:
    original.sft_runs.reload()
    verify_plan(plan)
    claims = modal.Dict.from_name(BUDGET_CLAIMS, create_if_missing=True)
    if not claims.put(CLAIM_KEY, {"run_name": plan["run_name"],
                                 "reservation_id": plan["reservation_id"],
                                 "launch_sha256": shared.digest(plan)}, skip_if_exists=True):
        raise RuntimeError("this parent budget lineage is already reserved; reconcile spend before another run")
    root = Path(plan["root"])
    root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(root / "launch.json", plan)
    write_json_atomic(root / "budget_guard.json", plan["budget"])
    original.sft_runs.commit()


@app.function(**resource_options("coordinator"), timeout=COORDINATOR_SECONDS)
def coordinate(plan: dict, absolute_deadline: float, app_id: str) -> dict:
    try:
        with deadline_alarm(absolute_deadline - 30):
            return coordinate_work(plan, absolute_deadline, app_id)
    except BaseException:
        try:
            original.stop_app(app_id)
        except BaseException as stop_error:
            progress(f"whole-app stop failed: {shared.error_record(stop_error)}")
        raise


def coordinate_work(plan: dict, absolute_deadline: float, app_id: str) -> dict:
    original.sft_runs.reload()
    root = Path(plan["root"])
    if shared.read_json(root / "launch.json") != plan or (root / "coordinator.json").exists():
        raise ValueError("reservation differs or coordinator already used")
    if not 0 < absolute_deadline - time.time() <= ABSOLUTE_SECONDS:
        raise ValueError("invalid absolute coordinator deadline")
    result = {"status": "running", "launch_sha256": shared.digest(plan), "started_at": shared.now(),
              "stages": {}, "coordinator_call_id": modal.current_function_call_id(), "app_id": app_id,
              "absolute_deadline_unix": absolute_deadline, "budget": plan["budget"]}
    active, phase = None, None

    def persist():
        original.sft_runs.reload()
        write_json_atomic(root / "coordinator.json", result)
        original.sft_runs.commit()

    try:
        persist()
        verify_plan(plan)
        for phase, function in (("prepare", prepare_cpu), ("train", train_h200), ("posteval", posteval_h200)):
            stage_deadline = min(absolute_deadline - 30, time.time() + STAGE_SECONDS[phase] + STARTUP)
            check_deadline(stage_deadline - 75)
            entry = result["stages"][phase] = {"status": "starting", "started_at": shared.now(),
                                              "deadline_unix": stage_deadline}
            persist()
            with deadline_alarm(stage_deadline):
                active = function.spawn(plan, stage_deadline)
                entry.update(status="running", call_id=active.object_id)
                persist()
                while True:
                    check_deadline(min(stage_deadline, absolute_deadline - 30))
                    remaining = min(stage_deadline, absolute_deadline - 30) - time.time()
                    try:
                        value = active.get(timeout=min(30, remaining))
                        break
                    except TimeoutError as polling_timeout:
                        if str(polling_timeout):
                            raise
                        progress(f"coordinator: {phase} {active.object_id}, {int(remaining)}s left")
            original.sft_runs.reload()
            receipt_path = root / phase / "result.json"
            if (value["status"] != "completed" or value["launch_sha256"] != shared.digest(plan)
                    or shared.read_json(receipt_path) != value
                    or shared.read_json(root / phase / "wrapper.json")["status"] != "completed"):
                raise RuntimeError(f"{phase} did not durably complete")
            entry.update(status="completed", ended_at=shared.now(), result=value,
                         result_sha256=sha256_file(receipt_path))
            persist()
            active = None
            if phase != "posteval":
                time.sleep(LIMITS["scaledown_seconds"])
        before = result["stages"]["prepare"]["result"]["result"]["baselines"]
        after = result["stages"]["posteval"]["result"]["result"]["panels"]
        if any(before[panel]["rows"] != after[panel]["rows"] for panel in before):
            raise ValueError("comparison requires complete identical panel coverage")
        result.update(status="completed", comparison={panel: {
            "before": before[panel]["correct"], "after": after[panel]["correct"],
            "rows": before[panel]["rows"], "baseline_complete": True,
            "before_files": before[panel]["files"], "after_files": after[panel]["files"],
        } for panel in before}, additional_updates=256, cumulative_updates=512)
    except BaseException as exc:
        result.update(status="failed", error=shared.error_record(exc))
        if phase in result["stages"]:
            result["stages"][phase].update(status="failed", ended_at=shared.now(), error=result["error"])
        if active is not None:
            try:
                with deadline_alarm(time.time() + 10):
                    active.cancel(terminate_containers=True)
                result["cancelled_call_id"] = active.object_id
            except BaseException as cancellation:
                result["cancellation_error"] = shared.error_record(cancellation)
        raise
    finally:
        result["ended_at"] = shared.now()
        with deadline_alarm(time.time() + 10):
            persist()
    return result


def stop_run(run_name: str) -> dict:
    shared.validate_cli("", "", shared.MODEL_ID, shared.MODEL_REVISION, run_name, "", PARENT)
    if os.environ.get("MODAL_PROFILE") != "icebear5h":
        raise ValueError("stop requires MODAL_PROFILE=icebear5h")
    local = LOCAL_ROOT / run_name
    plan = shared.read_json(local / "launch.json")
    if plan["schema"] != SCHEMA or plan["run_name"] != run_name:
        raise ValueError("not this extension-r02's local launch receipt")
    state = shared.read_json(local / "orchestration.json")
    if state["launch_sha256"] != shared.digest(plan):
        raise ValueError("stop receipt launch identity differs")
    original.stop_app(state["app_id"])
    state.update(status="app_stop_requested", stop_requested_at=shared.now())
    write_json_atomic(local / "orchestration.json", state)
    return state


def launch(run_name: str, budget_usd: float = 21, execute: bool = False) -> dict:
    plan = build_plan(run_name, budget_usd)
    verify_plan(plan)
    if not execute:
        return {"dry_run": True, "remote_calls": False, "plan": plan,
                "deferred_to_cpu": "old mounted receipts/data/baselines; current full parent checkpoint manifest; native cached base audit"}
    if os.environ.get("MODAL_PROFILE") != "icebear5h" or original.HF_SECRET_NAME != "huggingface-secret-2":
        raise ValueError("execute requires MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2")
    local = LOCAL_ROOT / run_name
    local.mkdir(parents=True, exist_ok=False)
    write_json_atomic(local / "launch.json", plan)
    state = {"status": "reserving", "launch_sha256": shared.digest(plan)}
    app_id = None
    write_json_atomic(local / "orchestration.json", state)
    try:
        with app.run(detach=True):
            app_id = app.app_id
            state["app_id"] = app_id
            write_json_atomic(local / "orchestration.json", state)
            try:
                with deadline_alarm(time.time() + LIMITS["reservation_seconds"] + STARTUP):
                    reservation = reserve.spawn(plan)
                    state["reservation_call_id"] = reservation.object_id
                    write_json_atomic(local / "orchestration.json", state)
                    reservation.get(timeout=LIMITS["reservation_seconds"] + STARTUP)
                absolute_deadline = time.time() + ABSOLUTE_SECONDS
                with deadline_alarm(absolute_deadline):
                    call = coordinate.spawn(plan, absolute_deadline, app_id)
                    state.update(status="spawned", coordinator_call_id=call.object_id,
                                 absolute_deadline_unix=absolute_deadline,
                                 remote_receipt=plan["root"] + "/coordinator.json")
                    write_json_atomic(local / "orchestration.json", state)
                    while True:
                        check_deadline(absolute_deadline)
                        try:
                            result = call.get(timeout=min(30, absolute_deadline - time.time()))
                            break
                        except TimeoutError as polling_timeout:
                            if str(polling_timeout):
                                raise
                    if result["status"] != "completed" or result["launch_sha256"] != shared.digest(plan):
                        raise RuntimeError("coordinator did not complete the reserved extension")
                    write_json_atomic(local / "coordinator.json", result)
                    state.update(status="completed", completed_at=shared.now(), comparison=result["comparison"])
            except BaseException as exc:
                state.update(status="launch_failed", error=shared.error_record(exc))
                if isinstance(exc, KeyboardInterrupt):
                    raise RuntimeError("extension interrupted; stopping dedicated app") from exc
                raise
            finally:
                try:
                    original.stop_app(app_id)
                    state.update(whole_app_stop_requested=True, stop_requested_at=shared.now())
                except BaseException as stop_error:
                    state["stop_error"] = shared.error_record(stop_error)
                write_json_atomic(local / "orchestration.json", state)
    except BaseException as exc:
        state.update(status="launch_failed", error=shared.error_record(exc))
        if app_id is not None and not state.get("whole_app_stop_requested"):
            try:
                original.stop_app(app_id)
                state["whole_app_stop_requested"] = True
            except BaseException as stop_error:
                state["stop_error"] = shared.error_record(stop_error)
        write_json_atomic(local / "orchestration.json", state)
        raise
    return {**state, "local_receipt": str(local / "orchestration.json"),
            "stop": f"MODAL_PROFILE=icebear5h {sys.executable} -B -m sft.modal_board_fluency_extension2 --run-name {run_name} --stop"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--budget-usd", type=float, default=21)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--stop", action="store_true")
    parser.add_argument("--worker", choices=tuple(STAGE_SECONDS), help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if (os.environ.get("MODAL_IS_REMOTE") != "1" or args.plan is None or args.deadline is None
                or args.execute or args.stop):
            parser.error("internal worker requires a remote container, plan and deadline")
        worker(shared.read_json(args.plan), args.worker, args.deadline)
    else:
        result = stop_run(args.run_name) if args.stop else launch(args.run_name, args.budget_usd, args.execute)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
