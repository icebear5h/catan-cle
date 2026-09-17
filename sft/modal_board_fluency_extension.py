"""Opt-in 128-update extension of r06; local dry runs never contact Modal.

MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2 \
    .venv/bin/python -B -m sft.modal_board_fluency_extension \
    --run-name board-fluency-extension-20260915-r01 --budget-usd 15
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


DEFAULT_RUN_NAME = "board-fluency-extension-20260915-r01"
PARENT_RUN = "board-fluency-sft-20260915-r06"
PARENT_ROOT = f"/runs/catan-vision-sft/{PARENT_RUN}"
PARENT = PARENT_ROOT + "/training/checkpoints/checkpoint-128"
DATA_DIR = f"/data/board-fluency-sft/{PARENT_RUN}"
LOCAL_ROOT = original.PROJECT_ROOT / "artifacts/runs/sft"
LOCAL_PARENT = LOCAL_ROOT / PARENT_RUN
SOURCE_FILES = (*original.SOURCE_FILES, "sft/modal_board_fluency_extension.py")
STAGE_SECONDS = {"prepare": 300, "train": 1800, "posteval": 1050}
STARTUP, COORDINATOR_SECONDS, ABSOLUTE_SECONDS = 300, 4500, 4350
RESOURCES = {"gpu": {"gpu": "H200", "cpu": 16, "memory_gib": 128},
             "prepare": {"cpu": 4, "memory_gib": 16},
             "coordinator": {"cpu": 1, "memory_gib": 2}}
LIMITS = {"stage_seconds": STAGE_SECONDS, "startup_seconds": STARTUP,
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
COMMON = dict(image=original.eval_image, volumes=original.VOLUMES, secrets=[original.secret],
              startup_timeout=STARTUP, retries=LIMITS["retries"],
              max_containers=LIMITS["max_containers"], scaledown_window=LIMITS["scaledown_seconds"])


def resource_options(kind: str) -> dict:
    resources = RESOURCES[kind]
    return {**COMMON, "cpu": (float(resources["cpu"]), float(resources["cpu"])),
            "memory": (resources["memory_gib"] * 1024,) * 2,
            **({"gpu": resources["gpu"]} if "gpu" in resources else {})}


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def verify_hashes(root: Path, hashes: dict) -> None:
    for name, expected in hashes.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"pinned file changed: {root / name}")


def source_hashes() -> dict:
    return {name: sha256_file(original.PROJECT_ROOT / name) for name in SOURCE_FILES}


def configuration(run_name: str, parent_config: dict) -> TrainConfig:
    shared.validate_cli("", "", shared.MODEL_ID, shared.MODEL_REVISION, run_name, "", PARENT)
    if run_name == PARENT_RUN:
        raise ValueError("extension must have a fresh run name")
    if parent_config != asdict(original.configuration(PARENT_RUN)):
        raise ValueError("parent configuration is not the exact approved r06 configuration")
    root = f"/runs/catan-vision-sft/{run_name}"
    config = replace(TrainConfig(**parent_config), initial_bundle=PARENT,
                     train_jsonl=root + "/prepare/train-after-1024.jsonl",
                     output_dir=root + "/training", resume_from_checkpoint=None)
    config.validate()
    return config


def budget_plan(budget_usd: float = 15) -> dict:
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


def read_parent(root: Path) -> dict:
    verify_hashes(root, PARENT_SHA256)
    parent = {name: shared.read_json(root / f"{name}/result.json")
              for name in ("prepare", "gate", "train", "posteval")}
    launch = parent["launch"] = shared.read_json(root / "launch.json")
    coordinator = parent["coordinator"] = shared.read_json(root / "coordinator.json")
    if (launch["run_name"] != PARENT_RUN or launch["root"] != PARENT_ROOT
            or launch["data_dir"] != DATA_DIR or launch["base"] != BASE
            or launch["model_revision"] != shared.MODEL_REVISION
            or coordinator["status"] != "completed" or coordinator["budget"] != launch["budget"]):
        raise ValueError("parent launch/coordinator identity or completion differs")
    configuration(DEFAULT_RUN_NAME, launch["config"])
    if original.source_hashes() != launch["source_sha256"]:
        raise ValueError("an original launcher/model/trainer/evaluator/scorer source changed since r06")
    for stage in ("prepare", "gate", "train", "posteval"):
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
    if (state["global_step"] != 128 or state["epoch"] != 0.32
            or training["committed_steps"] != CHECKPOINT_STEPS or training["final_checkpoint"] != PARENT
            or training["training"]["status"] != "completed"
            or training["training"]["metrics"]["epoch"] != 0.32
            or parent["posteval"]["result"]["checkpoint"] != PARENT):
        raise ValueError("parent must be the completed, trained r06 checkpoint-128")
    for split, rows, expected_hash in (
        ("train", 3200, launch["inputs"]["files"]["train.jsonl"]["sha256"]),
        ("eval", 120, PARENT_SHA256["prepare/teacher120.jsonl"]),
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

    prefix, consumed = contract(rows[:1024]), contract(rows[1024:2048])
    suffix = b"".join(lines[1024:])
    shared_states = set(prefix["state_sha256"]) & {row["metadata"]["state_sha256"] for row in rows[1024:]}
    if (len(set(inputs["panels"]["train"]["ids"])) != 3200
            or set(prefix["ids"]) & {row["id"] for row in rows[1024:]} or shared_states):
        raise ValueError("suffix repeats parent-prefix example IDs or canonical states")
    return suffix, {"sha256": hashlib.sha256(suffix).hexdigest(), "bytes": len(suffix),
                    "rows": 2176, "original_start_row_zero_based": 1024,
                    "source_sha256": inputs["files"]["train.jsonl"]["sha256"],
                    "parent_prefix": prefix, "consumed": consumed,
                    "parent_prefix_id_overlap": [],
                    "shared_parent_prefix_state_sha256": sorted(shared_states),
                    "unique_examples_basis": "row IDs; entire suffix also canonical-state-disjoint from parent prefix"}


def retained_baselines(parent: dict, root: Path, data: Path, review: Path) -> dict:
    """Strictly rescore the complete r06 post panels against unchanged gold/meta/IDs."""
    inputs = parent["launch"]["inputs"]
    baselines = parent["posteval"]["result"]["panels"]
    if set(baselines) != {"review", "validation_eval"}:
        raise ValueError("both complete r06 post panels are required")
    for panel, (count, correct) in {"review": (200, 58), "validation_eval": (190, 87)}.items():
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


def inspect_original(parent: dict, root: Path, data: Path, review: Path) -> tuple[bytes, dict, dict]:
    inputs = parent["launch"]["inputs"]
    if original.inspect_data(data, review) != inputs:
        raise ValueError("original data/inventory/review manifest changed")
    if root == Path(PARENT_ROOT):
        if sha256_file(data / "launch.json") != PARENT_SHA256["launch.json"]:
            raise ValueError("old uploaded data launch receipt changed")
    teacher = rows_at(root / "prepare/teacher120.jsonl")
    validation = {row["id"]: row for row in rows_at(data / "validation_eval.jsonl")}
    if teacher != [validation[row_id] for row_id in inputs["teacher_ids"]]:
        raise ValueError("teacher120 is not the exact original fixed panel")
    suffix, identity = suffix_identity(data / "train.jsonl", inputs)
    return suffix, identity, retained_baselines(parent, root, data, review)


def build_plan(run_name: str, budget_usd: float = 15) -> dict:
    parent = read_parent(LOCAL_PARENT)
    config = configuration(run_name, parent["launch"]["config"])
    if (LOCAL_ROOT / run_name).exists():
        raise FileExistsError("local run name already used")
    verify_hashes(LOCAL_PARENT, LOCAL_PARENT_SHA256)
    if shared.read_json(LOCAL_PARENT / "analysis.json")["checks_passed"] is not True:
        raise ValueError("parent offline analysis was not successful")
    _, suffix, baselines = inspect_original(parent, LOCAL_PARENT, original.DATASET, original.REVIEW)
    plan = {"schema": "catan_board_fluency_extension_launch/v1", "run_name": run_name,
            "created_at": shared.now(), "reservation_id": uuid.uuid4().hex,
            "root": str(Path(config.output_dir).parent), "data_dir": DATA_DIR,
            "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
            "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
            "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
            "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
            "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
            "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "suffix": suffix, "saved_baselines": baselines,
             "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": COST_SHA256},
             "cost_lineage": {"sha256": COST_SHA256, "total_key": COST_KEY,
                             "receipt_utf8": (LOCAL_PARENT / "cost_estimate.json").read_text()},
            "budget": budget_plan(budget_usd), "limits": LIMITS, "policy": POLICY}
    return normalized(plan)


def verify_plan(plan: dict) -> None:
    """Repeat offline admission locally; the same guards use old mounts on the server."""
    if normalized(plan) != plan:
        raise ValueError("plan must round-trip through strict JSON unchanged")
    remote = os.environ.get("MODAL_IS_REMOTE") == "1"
    root = Path(PARENT_ROOT) if remote else LOCAL_PARENT
    parent = read_parent(root)
    config = configuration(plan["run_name"], parent["launch"]["config"])
    expected = {"schema": "catan_board_fluency_extension_launch/v1", "root": str(Path(config.output_dir).parent),
                "data_dir": DATA_DIR, "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
                "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
                "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
                "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
                "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
                 "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "limits": LIMITS, "policy": POLICY,
                 "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": COST_SHA256},
                "budget": budget_plan(plan["budget"]["approved_usd"])}
    if any(plan[key] != value for key, value in expected.items()):
        raise ValueError("plan configuration/identity/source/budget/resource caps differ")
    if (len(plan["reservation_id"]) != 32
            or uuid.UUID(hex=plan["reservation_id"]).hex != plan["reservation_id"]):
        raise ValueError("invalid immutable reservation identity")
    cost = plan["cost_lineage"]
    if (cost["sha256"] != COST_SHA256 or cost["total_key"] != COST_KEY
            or hashlib.sha256(cost["receipt_utf8"].encode()).hexdigest() != COST_SHA256):
        raise ValueError("parent cost lineage differs")
    receipt = json.loads(cost["receipt_utf8"], parse_float=Decimal)
    if (receipt["run_name"] != PARENT_RUN or receipt["totals"][COST_KEY] != PRIOR_USD
            or receipt["provenance"]["launch_sha256"] != shared.digest(parent["launch"])):
        raise ValueError("parent cost carry-forward is not the pinned allowance-inclusive total")
    for key in ("h200_per_second", "cpu_core_per_second", "memory_gib_per_second", "gpu_stage_per_second"):
        if (not math.isclose(float(receipt["rates"][key + "_usd"]), plan["budget"][key], rel_tol=0, abs_tol=1e-15)
                or not math.isclose(parent["launch"]["budget"][key], plan["budget"][key], rel_tol=0, abs_tol=1e-15)):
            raise ValueError("extension rates differ from the parent cost lineage")
    if not remote:
        verify_hashes(root, LOCAL_PARENT_SHA256)
    data, review = (Path(DATA_DIR), Path(DATA_DIR) / "review.jsonl") if remote else (original.DATASET, original.REVIEW)
    _, suffix, baselines = inspect_original(parent, root, data, review)
    if plan["suffix"] != suffix or plan["saved_baselines"] != baselines:
        raise ValueError("suffix identity or complete retained baseline differs")
    original.require_dependencies()


def full_checkpoint_manifest(path: Path) -> dict:
    shared.volume_bundle(str(path))
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    return shared.file_manifest(path, sorted(str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()))


def text_boundaries(tokenizer, rows: list[dict], *, generation: bool, deadline: float) -> dict:
    maximum, max_prompt, max_answer = 0, 0, 0
    for row in rows:
        check_deadline(deadline)
        prompt, answer = _message_pair(row, line_number=0, input_mode="text")
        pair = encode_text_pair(tokenizer, prompt, answer, max_sequence_length=4096)
        prefix = pair["labels"].count(-100)
        completion = len(pair["input_ids"]) - prefix
        if generation and (prefix + 512 > 4096 or completion > 512):
            raise ValueError("native text boundary exceeds greedy512 allowance")
        maximum = max(maximum, len(pair["input_ids"]))
        max_prompt, max_answer = max(max_prompt, prefix), max(max_answer, completion)
    return {"rows": len(rows), "max_sequence": maximum, "max_prompt": max_prompt, "max_completion": max_answer}


def prepare_work(plan: dict, deadline: float) -> dict:
    parent = read_parent(Path(PARENT_ROOT))
    prior = parent["prepare"]["result"]
    inventory = load_token_inventory(plan["config"]["token_inventory"])
    bundle, _ = shared.volume_bundle(PARENT)
    parent_files = full_checkpoint_manifest(bundle)
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
    if frozen != parent["train"]["result"]["frozen_visual_file_tensor_sha256"]:
        raise ValueError("parent frozen visual tensors differ from r06 training")
    suffix, identity, baselines = inspect_original(parent, Path(PARENT_ROOT), Path(DATA_DIR), Path(DATA_DIR) / "review.jsonl")
    if identity != plan["suffix"] or baselines != plan["saved_baselines"]:
        raise ValueError("CPU suffix/baseline admission differs")
    lengths = {"train_suffix": text_boundaries(tokenizer, [json.loads(line) for line in suffix.splitlines()],
                                              generation=False, deadline=deadline)}
    for panel, path in {"teacher120": plan["config"]["eval_jsonl"],
                        "review": DATA_DIR + "/review.jsonl", "validation_eval": DATA_DIR + "/validation_eval.jsonl"}.items():
        lengths[panel] = text_boundaries(tokenizer, rows_at(Path(path)), generation=panel != "teacher120", deadline=deadline)
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
            "parent_manifest_provenance": "full current checkpoint manifest established by this CPU prepare; no historical r06 adapter hash claimed",
            "frozen_visual_file_tensor_sha256": frozen, "base_files": prior["base_files"],
            "base_audit": base_audit, "base_manifest_provenance": "r06 metadata hashes and shard sizes, rechecked tensor headers/index",
            "suffix_files": suffix_files, "suffix": identity, "baselines": baselines,
            "teacher_sha256": PARENT_SHA256["prepare/teacher120.jsonl"],
            "token_lengths": lengths, "prior_token_lengths": prior["token_lengths"],
            "token_boundary_provenance": "actual native trained-parent tokenizer, exact original raw rows"}


def completed_stage(plan: dict, stage: str) -> dict:
    root = Path(plan["root"])
    receipt = shared.read_json(root / stage / "result.json")
    entry = shared.read_json(root / "coordinator.json")["stages"][stage]
    if (receipt["status"] != "completed" or receipt["stage"] != stage
            or receipt["launch_sha256"] != shared.digest(plan) or entry["status"] != "completed"
            or entry["result"] != receipt or entry["result_sha256"] != sha256_file(root / stage / "result.json")):
        raise ValueError(f"{stage} completion/receipt hash differs")
    return receipt["result"]


def prepared_for(plan: dict) -> dict:
    prepared = completed_stage(plan, "prepare")
    check_manifest(Path(PARENT), prepared["parent_files"])
    check_manifest(Path(BASE), prepared["base_files"])
    check_manifest(Path(plan["config"]["train_jsonl"]).parent, prepared["suffix_files"])
    if (prepared["suffix"] != plan["suffix"] or prepared["baselines"] != plan["saved_baselines"]
            or sha256_file(Path(plan["config"]["eval_jsonl"])) != prepared["teacher_sha256"]):
        raise ValueError("prepared suffix/teacher/baseline identity changed")
    return prepared


def verify_initialization(config: TrainConfig, prepared: dict, report: dict) -> None:
    # Resolve both sides on the mounted volume; Modal may record /__modal/volumes/vo-... .
    if Path(report["path"]).resolve(strict=True) != Path(PARENT).resolve(strict=True):
        raise ValueError("trainer did not load the trained r06 checkpoint-128")
    for key, name in {"adapter_sha256": "adapter_model.safetensors", "visual_sha256": trainer.VISUAL_STATE_FILE,
                      "parent_training_config_sha256": trainer.RUN_CONFIG_FILE}.items():
        if report[key] != prepared["parent_files"][name]["sha256"]:
            raise ValueError(f"actual initial-bundle content differs: {key}")
    expected = {"input_mode": "text", "source_input_mode": "text", "lora_rank": 16, "lora_alpha": 32,
                "optimizer_state_restored": False, "scheduler_state_restored": False, "rng_state_restored": False}
    if any(report[key] != value for key, value in expected.items()) or config.resume_from_checkpoint is not None:
        raise ValueError("extension requires the trained text bundle and fresh optimizer/schedule/RNG")


class ExtensionCallback(CheckpointCallback):
    def __init__(self, config: TrainConfig, deadline: float, prepared: dict):
        super().__init__(config, deadline)
        self.prepared = prepared

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if (args.world_size != 1 or args.train_sampling_strategy != "sequential" or args.shuffle_dataset
                or args.max_steps != 128 or args.per_device_train_batch_size != 4
                or args.gradient_accumulation_steps != 2 or args.save_total_limit != 4):
            raise ValueError("actual trainer sampling/update/checkpoint retention differs")
        output = Path(self.config.output_dir)
        verify_initialization(self.config, self.prepared, shared.read_json(output / trainer.INITIAL_BUNDLE_FILE))
        dataset = shared.read_json(output / trainer.DATASET_REPORT_FILE)
        for split, path, rows, expected in (
            ("train", self.config.train_jsonl, 2176, self.prepared["suffix"]["sha256"]),
            ("eval", self.config.eval_jsonl, 120, self.prepared["teacher_sha256"]),
        ):
            actual = dataset[split]
            if (Path(actual["source"]).resolve(strict=True) != Path(path).resolve(strict=True)
                    or actual["source_sha256"] != expected or actual["rows"] != rows
                    or actual["input_mode"] != "text" or actual["truncation"]):
                raise ValueError("actual trainer dataset differs from admitted sequential data")


def train_work(plan: dict, deadline: float) -> dict:
    prepared = prepared_for(plan)
    config = TrainConfig(**plan["config"])
    if Path(config.output_dir).exists():
        raise FileExistsError("training output exists; retries/resumption are not authorized")
    callback = ExtensionCallback(config, deadline, prepared)
    progress("extension: trained r06 checkpoint, fresh optimizer, rows 1025–2048, 128 updates")
    result = trainer.run_training(config, extra_callbacks=[callback])
    if result["status"] != "completed" or callback.saved_steps != CHECKPOINT_STEPS:
        raise RuntimeError("extension did not commit exactly checkpoints 32/64/96/128")
    verify_initialization(config, prepared, result["initial_bundle"])
    for step in CHECKPOINT_STEPS:
        saved = Path(config.output_dir) / "checkpoints" / f"checkpoint-{step}"
        shared.volume_bundle(str(saved))
        if shared.read_json(saved / "trainer_state.json")["global_step"] != step:
            raise ValueError("retained checkpoint global step differs")
    final = Path(config.output_dir) / "checkpoints/checkpoint-128"
    frozen = visual_file_digest(final / trainer.VISUAL_STATE_FILE)
    if frozen != prepared["frozen_visual_file_tensor_sha256"]:
        raise ValueError("extension changed the parent's canonical frozen visual tensors")
    check_manifest(Path(PARENT), prepared["parent_files"])
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("source manifest changed during training")
    return {"training": result, "final_checkpoint": str(final), "committed_steps": callback.saved_steps,
            "final_checkpoint_files": full_checkpoint_manifest(final),
            "frozen_visual_file_tensor_sha256": frozen, "additional_updates": 128, "cumulative_updates": 256,
            "consumed": plan["suffix"]["consumed"], "cumulative_unique_examples": 2048,
            "cumulative_corpus_epoch": 0.64, "consumption_basis": "verified single-process sequential sampler, exact suffix hash, 128 completed batch8 updates"}


def posteval_work(plan: dict, deadline: float) -> dict:
    prepared = prepared_for(plan)
    training = completed_stage(plan, "train")
    checkpoint = plan["config"]["output_dir"] + "/checkpoints/checkpoint-128"
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
    command = [sys.executable, "-B", "-m", "sft.modal_board_fluency_extension", "--worker", stage,
               "--plan", plan["root"] + "/launch.json", "--deadline", str(inner_deadline)]
    wrapper = {"status": "running", "stage": stage, "launch_sha256": shared.digest(plan),
               "started_at": shared.now(), "command": command,
               "call_id": modal.current_function_call_id(), "hard_deadline_unix": hard_deadline}
    process, error = None, None
    log_path = Path("/tmp") / f"board-fluency-extension-{plan['reservation_id']}-{stage}.log"
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
    if not claims.put(COST_SHA256, {"run_name": plan["run_name"],
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
        # Disarm the work alarm before bounded cancellation, including failures
        # preceding the first coordinator receipt. Stop the dedicated app only.
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
                persist()  # Child call identity is durable before any wait.
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
        } for panel in before}, additional_updates=128, cumulative_updates=256)
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
    if plan["schema"] != "catan_board_fluency_extension_launch/v1" or plan["run_name"] != run_name:
        raise ValueError("not this extension's local launch receipt")
    state = shared.read_json(local / "orchestration.json")
    if state["launch_sha256"] != shared.digest(plan):
        raise ValueError("stop receipt launch identity differs")
    original.stop_app(state["app_id"])
    state.update(status="app_stop_requested", stop_requested_at=shared.now())
    write_json_atomic(local / "orchestration.json", state)
    return state


def launch(run_name: str, budget_usd: float = 15, execute: bool = False) -> dict:
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
            # Handle interruption inside app.run: Modal otherwise suppresses
            # KeyboardInterrupt for detached apps and leaves paid work running.
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
                    # The local supervisor bounds coordinator import/startup
                    # loops before its own remote alarm can be installed.
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
            "stop": f"MODAL_PROFILE=icebear5h {sys.executable} -B -m sft.modal_board_fluency_extension --run-name {run_name} --stop"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--budget-usd", type=float, default=15)
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
