"""Opt-in 256-additional-update extension of the completed spatial continuation.

Dry run: python -B -m sft.modal_spatial_extension --run-name spatial-continuation-20260912-r01
Add --execute to reserve a new run and detach its bounded CPU coordinator.
The unchanged, already-uploaded parent datasets and all six saved post panels
are pinned by local receipts and checked on CPU before the only training call.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import modal
import torch
from transformers import AddedToken, AutoTokenizer

from sft.modal_spatial_continuation import (
    CPU_OPTIONS, DEFAULT_INPUTS, GPU_OPTIONS, LOCAL_RUN_ROOT, NEW_PANEL_TASKS,
    OLD_PANELS, PANEL_BUDGETS, PREFLIGHT_TIMEOUT, RUN_ROOT, SNAPSHOT,
    STARTUP_TIMEOUT, WAIT_GRACE, checkpoint_audit, check_identity,
    compare_panels, completion_audit, continuation_image, dataset_identity,
    digest, evaluation_conditions, evaluator, matched_baseline, now,
    read_json, reload_volumes, sft_runs, source_hashes, validate_mixture,
    validate_run_name, panel_args,
)
from sft.scripts.train_trl_catan_vision import (
    TrainConfig, inspect_jsonl_contract, iter_jsonl, load_token_inventory,
    normalize_training_config, run_training, sha256_file, write_json_atomic,
)

PARENT_RUN = "spatial-continuation-20260909-r01"
PARENT_RESULT = LOCAL_RUN_ROOT / PARENT_RUN / "result.json"
PARENT_CHECKPOINT = str(RUN_ROOT / PARENT_RUN / "checkpoints/checkpoint-128")
DEFAULT_RUN_NAME = "spatial-continuation-20260912-r01"
TRAIN_TIMEOUT = 7200
POST_TIMEOUT = 3600
COORDINATOR_TIMEOUT = 14400
CHECKPOINT_STEPS = tuple(range(32, 257, 32))
CONFIG_CHANGES = dict(max_steps=256, save_steps=32, eval_steps=32, save_total_limit=8,
                      resume_from_checkpoint=None)
app = modal.App("catan-spatial-extension")
TRAIN_GPU_OPTIONS = {**GPU_OPTIONS, "image": continuation_image, "timeout": TRAIN_TIMEOUT}
POST_GPU_OPTIONS = {**GPU_OPTIONS, "image": continuation_image, "timeout": POST_TIMEOUT}


def validate_config(payload: dict, run_name: str, parent_config: dict) -> TrainConfig:
    validate_run_name(run_name)
    if run_name == PARENT_RUN:
        raise ValueError("extension must not overwrite its parent")
    parent = TrainConfig(**parent_config)
    parent.validate()
    if (parent.max_steps != 128 or parent.seed != 44 or parent.profile != "vision_tokens_lora"
            or parent.per_device_train_batch_size != 4 or parent.gradient_accumulation_steps != 2
            or parent.per_device_eval_batch_size != 2 or not parent.eval_jsonl
            or parent.token_init != "keep" or parent.require_curriculum
            or parent.input_mode != "vision" or parent.max_sequence_length is not None
            or parent.publish_to_hub or parent.resume_from_checkpoint or parent.olora):
        raise ValueError("parent configuration is not the approved completed mixed run")
    expected = asdict(replace(parent, **CONFIG_CHANGES, initial_bundle=PARENT_CHECKPOINT,
                              output_dir=str(RUN_ROOT / run_name)))
    if normalize_training_config(payload) != normalize_training_config(expected):
        raise ValueError("configuration must inherit ALL parent settings except approved extension fields")
    config = TrainConfig(**payload)
    config.validate()
    return config


def build_plan(inputs_path: Path, run_name: str) -> dict:
    parent = read_json(PARENT_RESULT)
    post_path = PARENT_RESULT.parent / "post/result.json"
    post = read_json(post_path)
    launch_path = PARENT_RESULT.with_name("launch.json")
    parent_launch = read_json(launch_path)
    if (parent["status"] != "completed" or parent["phase"] != "completed"
            or post["status"] != "completed"
            or any(parent["stages"][key]["status"] != "completed" for key in ("preflight", "training", "post"))
            or parent["stages"]["post"]["result"] != post
            or normalize_training_config(parent_launch["config"]) != normalize_training_config(parent["config"])
            or digest(parent_launch["config"]) != parent_launch["config_sha256"]):
        raise ValueError("parent launch/result/post receipts disagree or are incomplete")
    if (post["checkpoint"] != PARENT_CHECKPOINT
            or post["checkpoint_audit"]["checkpoint"] != PARENT_CHECKPOINT
            or set(post["panels"]) != set(PANEL_BUDGETS)):
        raise ValueError("parent must supply the checkpoint-128 audit and ALL SIX post panels")
    config = asdict(replace(TrainConfig(**parent["config"]), **CONFIG_CHANGES,
                            initial_bundle=PARENT_CHECKPOINT, output_dir=str(RUN_ROOT / run_name)))
    validate_config(config, run_name, parent["config"])
    if (LOCAL_RUN_ROOT / run_name).exists():
        raise FileExistsError(LOCAL_RUN_ROOT / run_name)
    inputs = read_json(inputs_path)
    if inputs != parent_launch["dataset_inputs"]:
        raise ValueError("dataset_inputs must be the unchanged parent manifest")
    local_hashes = {}
    for path in (inputs_path, Path(inputs["metadata"]), Path(inputs["token_inventory"])):
        value = sha256_file(path)
        if value != parent_launch["input_files_sha256"][str(path)]:
            raise ValueError(f"parent input receipt changed: {path}")
        local_hashes[str(path)] = value
    inventory = load_token_inventory(inputs["token_inventory"])
    if inventory["atlas_tokens"] != post["checkpoint_audit"]["semantic_tokens"]["tokens"]:
        raise ValueError("parent atlas inventory differs")
    train_identity = dataset_identity(inputs["train_jsonl"], inputs["image_root"])
    if train_identity != parent_launch["train_identity"]:
        raise ValueError("frozen ordered training dataset changed")
    rows = [row for _, row in iter_jsonl(Path(inputs["train_jsonl"]))]
    mixture = validate_mixture(rows)
    if mixture != parent_launch["mixture"]:
        raise ValueError("parent optimizer-step ordering changed")
    inspect_jsonl_contract(inputs["train_jsonl"], inputs["image_root"], require_curriculum=False)
    preflight = parent["stages"]["preflight"]["result"]
    check_identity(preflight["train_identity"], train_identity)
    panels, baselines = {}, {}
    remote_hashes = {str(RUN_ROOT / "pipelines" / PARENT_RUN / name): sha256_file(path)
                     for name, path in (("result.json", PARENT_RESULT), ("post/result.json", post_path))}
    for label, (count, batch, budget) in PANEL_BUDGETS.items():
        local = OLD_PANELS[label] if label in OLD_PANELS else inputs["new_panels"][NEW_PANEL_TASKS[label]]
        identity = dataset_identity(local["eval_jsonl"], local["image_root"])
        local_hashes[local["eval_jsonl"]] = identity["sha256"]
        saved = post["panels"][label]
        if (identity != parent_launch["panels"][label]["identity"] or identity["rows"] != count
                or saved["conditions"] != evaluation_conditions(label)
                or saved["scorer_sha256"] != digest(post["source_sha256"])
                or saved["summary"]["eval_source_sha256"] != identity["sha256"]):
            raise ValueError(f"frozen panel identity/conditions differ: {label}")
        check_identity(saved["identity"], identity)
        output = RUN_ROOT / "pipelines" / PARENT_RUN / "post" / label
        if saved["records_path"] != str(output / "records.jsonl"):
            raise ValueError(f"parent records path differs: {label}")
        summary_path = post_path.parent / f"{label}-summary.json"
        records_path = post_path.parent / f"{label}-records.jsonl"
        if (read_json(summary_path) != saved["summary"]
                or sha256_file(records_path) != saved["records_sha256"]):
            raise ValueError(f"local saved parent responses or summary differ: {label}")
        for path in (summary_path, records_path):
            local_hashes[str(path)] = sha256_file(path)
        panels[label] = {"eval_jsonl": saved["summary"]["eval_jsonl"],
                         "image_root": saved["summary"]["image_root"], "identity": saved["identity"],
                         "batch_size": batch, "max_new_tokens": budget, "local_identity": identity}
        baselines[label] = {**saved, "output_dir": str(output),
                            "summary_sha256": local_hashes[str(summary_path)]}
    check_identity(preflight["teacher_eval_identity"], panels["fullboard"]["identity"])
    for path in (PARENT_RESULT, post_path, launch_path, Path(inputs["train_jsonl"])):
        local_hashes[str(path)] = sha256_file(path)
    return {"schema": "catan_spatial_extension_launch/v1", "run_name": run_name,
            "config": config, "config_sha256": digest(config), "parent_config": parent["config"],
            "parent_config_sha256": digest(parent["config"]), "parent_checkpoint": PARENT_CHECKPOINT,
            "parent_audit": post["checkpoint_audit"], "dataset_inputs": inputs,
            "train_identity": preflight["train_identity"], "teacher_eval_identity": preflight["teacher_eval_identity"],
            "token_inventory_sha256": local_hashes[inputs["token_inventory"]],
            "mixture": mixture, "panels": panels, "saved_baselines": baselines,
            "input_files_sha256": local_hashes, "remote_receipts_sha256": remote_hashes,
            "source_sha256": source_hashes(),
            "policy": {"stages": ["cpu_preflight", "train256", "post_all_six"],
                       "additional_optimizer_steps": 256, "cumulative_mixed_updates": 384,
                       "dataset_passes": 2, "family_steps": {k: 2 * v for k, v in mixture["family_steps"].items()},
                       "checkpoint_steps": list(CHECKPOINT_STEPS), "fresh_optimizer_and_schedule": True,
                       "preflight_timeout_seconds": PREFLIGHT_TIMEOUT, "train_timeout_seconds": TRAIN_TIMEOUT,
                       "post_timeout_seconds": POST_TIMEOUT, "startup_timeout_seconds": STARTUP_TIMEOUT,
                       "coordinator_timeout_seconds": COORDINATOR_TIMEOUT, "retries": 0,
                       "blank_controls": False, "reuse_parent_uploaded_data": True,
                       "saved_baseline_scoring": "all six original post panels, CPU rescore before training"}}


def verify_runtime(plan: dict) -> None:
    validate_config(plan["config"], plan["run_name"], plan["parent_config"])
    if (plan["parent_checkpoint"] != PARENT_CHECKPOINT
            or digest(plan["config"]) != plan["config_sha256"]
            or digest(plan["parent_config"]) != plan["parent_config_sha256"]
            or source_hashes() != plan["source_sha256"]):
        raise ValueError("source/config/checkpoint changed since launch planning")
    if set(plan["panels"]) != set(PANEL_BUDGETS) or set(plan["saved_baselines"]) != set(PANEL_BUDGETS):
        raise ValueError("all six frozen panels and saved baselines are required")
    for label, panel in plan["panels"].items():
        if (panel["identity"]["rows"], panel["batch_size"], panel["max_new_tokens"]) != PANEL_BUDGETS[label]:
            raise ValueError("unapproved panel budget")


def verify_files(hashes: dict) -> None:
    for path, expected in hashes.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"pinned file changed: {path}")


def audit_inputs(plan: dict) -> dict:
    config = plan["config"]
    identities = {"train_identity": dataset_identity(config["train_jsonl"], config["image_root"]),
                  "teacher_eval_identity": dataset_identity(config["eval_jsonl"], config["eval_image_root"])}
    for key, actual in identities.items():
        if actual != plan[key]:
            raise ValueError(f"uploaded input bytes/pixels changed: {key}")
    if sha256_file(Path(config["token_inventory"])) != plan["token_inventory_sha256"]:
        raise ValueError("uploaded token inventory changed")
    return identities


def audit_parent(plan: dict) -> dict:
    return checkpoint_audit(Path(plan["parent_checkpoint"]), plan["parent_config"], parent=True,
                            expected_step=128, expected_audit=plan["parent_audit"])


@app.function(**CPU_OPTIONS, cpu=(4.0, 4.0), memory=(8192, 8192), timeout=PREFLIGHT_TIMEOUT)
def extension_preflight(plan: dict) -> dict:
    reload_volumes()
    verify_runtime(plan)
    if Path(plan["config"]["output_dir"]).exists():
        raise FileExistsError(plan["config"]["output_dir"])
    if not Path(SNAPSHOT).is_dir():
        raise FileNotFoundError("pinned base snapshot is not cached")
    verify_files(plan["remote_receipts_sha256"])
    checkpoint = audit_parent(plan)
    identities = audit_inputs(plan)
    inventory = load_token_inventory(plan["config"]["token_inventory"])
    tokenizer = AutoTokenizer.from_pretrained(PARENT_CHECKPOINT, local_files_only=True)
    base = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    base.add_tokens([AddedToken(t, normalized=False, special=False) for t in inventory["atlas_tokens"]])
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in inventory["atlas_tokens"]]
    expected = checkpoint["semantic_tokens"]
    if (expected["tokens"] != inventory["atlas_tokens"] or ids != [[i] for i in expected["token_ids"]]
            or [base.encode(t, add_special_tokens=False) for t in inventory["atlas_tokens"]] != ids):
        raise ValueError("parent/base tokenizer atlas IDs differ")
    indices = read_json(Path(PARENT_CHECKPOINT) / "adapter_config.json")["trainable_token_indices"]
    if len(indices) != 2 or any(value != expected["token_ids"] for value in indices.values()):
        raise ValueError("both input/output atlas row IDs must be retained")
    rows = [row for _, row in iter_jsonl(Path(plan["config"]["train_jsonl"]))]
    if validate_mixture(rows) != plan["mixture"]:
        raise ValueError("training step ownership changed")
    report = {"status": "completed", "checkpoint": PARENT_CHECKPOINT, "checkpoint_audit": checkpoint,
              "source_sha256": plan["source_sha256"], **identities, "mixture": plan["mixture"],
              "train_tokens": completion_audit(rows, tokenizer), "panel_tokens": {}, "panels": {}, "baselines": {}}
    for label, panel in plan["panels"].items():
        identity = dataset_identity(panel["eval_jsonl"], panel["image_root"])
        if identity != panel["identity"]:
            raise ValueError(f"uploaded panel changed: {label}")
        rows = [row for _, row in iter_jsonl(Path(panel["eval_jsonl"]))]
        report["panel_tokens"][label] = completion_audit(rows, tokenizer, budget=panel["max_new_tokens"])
        report["panels"][label] = identity
        saved = plan["saved_baselines"][label]
        baseline = matched_baseline(label, saved, panel, checkpoint,
                                    expected_checkpoint=PARENT_CHECKPOINT,
                                    expected_correct=saved["summary"]["correct"], strict_summary=True)
        report["baselines"][label] = {**baseline, "scorer_sha256": digest(plan["source_sha256"]),
                                     "conditions": evaluation_conditions(label)}
    sft_runs.commit()
    return report


def training_history(output: Path, *, completed: bool) -> dict:
    """Retain periodic teacher-forced metrics from the trainer's real saved states."""
    checkpoints = {}
    history = {}
    for step in CHECKPOINT_STEPS:
        path = output / "checkpoints" / f"checkpoint-{step}"
        state_path = path / "trainer_state.json"
        if not state_path.exists() and not completed:
            continue
        state = read_json(state_path)
        if state["global_step"] != step:
            raise ValueError("saved checkpoint global_step differs")
        for row in state["log_history"]:
            if "eval_loss" in row:
                if row["step"] not in CHECKPOINT_STEPS or row["step"] > step or not math.isfinite(row["eval_loss"]):
                    raise ValueError("invalid periodic teacher-forced history")
                if row["step"] in history and history[row["step"]] != row:
                    raise ValueError("teacher-forced history differs across checkpoints")
                history[row["step"]] = row
        if completed:
            for name in ("adapter_config.json", "adapter_model.safetensors", "visual_model.safetensors",
                         "trainable_parameters.json", "training_config.json", "tokenizer_config.json", "tokenizer.json"):
                if not (path / name).is_file():
                    raise FileNotFoundError(path / name)
        checkpoints[str(step)] = {"checkpoint": str(path), "trainer_state_sha256": sha256_file(state_path)}
    if completed and set(history) != set(CHECKPOINT_STEPS):
        raise ValueError("all eight periodic teacher-forced evaluations are required")
    return {"checkpoints": checkpoints, "teacher_forced_history": [history[k] for k in sorted(history)]}


@app.function(**TRAIN_GPU_OPTIONS)
def train_bounded(plan: dict, audit: dict) -> dict:
    reload_volumes()
    verify_runtime(plan)
    if audit["status"] != "completed" or set(audit["baselines"]) != set(PANEL_BUDGETS):
        raise ValueError("all six CPU baselines must complete before training")
    if audit_parent(plan) != audit["checkpoint_audit"]:
        raise ValueError("parent changed after CPU preflight")
    for key, identity in audit_inputs(plan).items():
        if identity != audit[key]:
            raise ValueError("inputs changed after CPU preflight")
    output = Path(plan["config"]["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoints/checkpoint-256"
    result = {"status": "running", "config": plan["config"], "started_at": now(),
              "checkpoint": str(checkpoint), "checkpoint_audit": None, "panels": {},
              "source_sha256": plan["source_sha256"], "timeout_seconds": TRAIN_TIMEOUT}
    destination = output / "result.json"
    write_json_atomic(destination, result)
    sft_runs.commit()
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        torch.set_num_threads(16)
        result["training"] = run_training(TrainConfig(**plan["config"]))
        result.update(training_history(output, completed=True))
        result["checkpoint_audit"] = checkpoint_audit(checkpoint, plan["config"], parent=False, expected_step=256)
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            result.update(training_history(output, completed=False))
        except Exception as history_error:
            result["history_error"] = repr(history_error)
        raise
    finally:
        result["ended_at"] = now()
        write_json_atomic(destination, result)
        sft_runs.commit()
    return result


@app.function(**POST_GPU_OPTIONS)
def evaluate_bounded(plan: dict, audit: dict, training: dict) -> dict:
    reload_volumes()
    verify_runtime(plan)
    checkpoint = str(Path(plan["config"]["output_dir"]) / "checkpoints/checkpoint-256")
    if training["status"] != "completed" or training["checkpoint"] != checkpoint:
        raise ValueError("post evaluation requires completed checkpoint-256 training")
    checkpoint_report = checkpoint_audit(Path(checkpoint), plan["config"], parent=False,
                                         expected_step=256, expected_audit=training["checkpoint_audit"])
    output = RUN_ROOT / "pipelines" / plan["run_name"] / "post"
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "checkpoint": checkpoint, "checkpoint_audit": checkpoint_report,
              "started_at": now(), "panels": {}, "source_sha256": plan["source_sha256"]}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        torch.set_num_threads(16)
        model, processor, evidence = evaluator.load_model(
            model_id=SNAPSHOT, adapter_dir=checkpoint, bits=16, disable_flash_attn2=True,
            token_inventory=plan["config"]["token_inventory"], preserve_visual_fp32=True)
        for label, panel in plan["panels"].items():
            identity = dataset_identity(panel["eval_jsonl"], panel["image_root"])
            if identity != audit["panels"][label] or identity != panel["identity"]:
                raise ValueError("uploaded panel bytes/pixels changed after CPU preflight")
            summary = evaluator.run_eval_job(model=model, processor=processor, adapter_evidence=evidence,
                args=panel_args(plan, checkpoint, label), eval_jsonl=panel["eval_jsonl"],
                image_variant="original", output_dir=output / label)
            if summary["rows"] != PANEL_BUDGETS[label][0] or summary["attempted"] != summary["rows"]:
                raise ValueError("evaluation returned incomplete panel")
            records = output / label / "records.jsonl"
            result["panels"][label] = {"summary": summary, "identity": identity,
                "records_path": str(records), "records_sha256": sha256_file(records),
                "scorer_sha256": digest(plan["source_sha256"]), "conditions": evaluation_conditions(label)}
            write_json_atomic(output / "result.json", result)
            sft_runs.commit()
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["ended_at"] = now()
        write_json_atomic(output / "result.json", result)
        sft_runs.commit()
    return result


@app.function(**CPU_OPTIONS, cpu=(0.25, 0.25), memory=(2048, 2048), timeout=300)
def reserve(plan: dict) -> dict:
    sft_runs.reload()
    verify_runtime(plan)
    if Path(plan["config"]["output_dir"]).exists():
        raise FileExistsError(plan["config"]["output_dir"])
    directory = RUN_ROOT / "pipelines" / plan["run_name"]
    directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(directory / "launch.json", plan)
    sft_runs.commit()
    return {"status": "reserved", "reservation_id": plan["reservation_id"]}


@app.function(**CPU_OPTIONS, cpu=(1.0, 1.0), memory=(4096, 4096), timeout=COORDINATOR_TIMEOUT)
def coordinate(plan: dict) -> dict:
    sft_runs.reload()
    directory = RUN_ROOT / "pipelines" / plan["run_name"]
    if read_json(directory / "launch.json") != plan:
        raise ValueError("remote launch reservation differs")
    destination = directory / "result.json"
    if destination.exists() or Path(plan["config"]["output_dir"]).exists():
        raise FileExistsError(destination)
    result = {"status": "running", "phase": "preflight", "started_at": now(), "stages": {},
              "config": plan["config"], "source_sha256": plan["source_sha256"],
              "coordinator_call_id": modal.current_function_call_id()}
    write_json_atomic(directory / "launch.json", {**plan, "coordinator_call_id": result["coordinator_call_id"]})
    sft_runs.commit()
    active = None

    def persist():
        sft_runs.reload()
        write_json_atomic(destination, result)
        sft_runs.commit()

    def stage(name, function, args, timeout):
        nonlocal active
        result["phase"] = name
        entry = result["stages"][name] = {"status": "starting", "started_at": now()}
        persist()
        active = function.spawn(*args)
        wait = timeout + STARTUP_TIMEOUT + WAIT_GRACE
        entry.update(call_id=active.object_id, status="running", wait_timeout_seconds=wait)
        persist()
        value = active.get(timeout=wait)
        if value.get("status") != "completed":
            raise RuntimeError(f"{name} did not complete: {value.get('status')}")
        entry.update(status="completed", ended_at=now(), result=value)
        persist()
        active = None
        return value

    try:
        persist()
        verify_runtime(plan)
        audit = stage("preflight", extension_preflight, (plan,), PREFLIGHT_TIMEOUT)
        training = stage("training", train_bounded, (plan, audit), TRAIN_TIMEOUT)
        post = stage("post", evaluate_bounded, (plan, audit, training), POST_TIMEOUT)
        result["comparison"] = compare_panels(audit["baselines"], post["panels"])
        result.update(status="completed", phase="completed")
    except BaseException as exc:
        if active is not None:
            try:
                active.cancel(terminate_containers=True)
                result["cancelled_call_id"] = active.object_id
            except BaseException as cancellation:
                result["cancellation_error"] = repr(cancellation)
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if result["phase"] in result["stages"]:
            result["stages"][result["phase"]].update(status="failed", ended_at=now(), error=result["error"])
        raise
    finally:
        result["ended_at"] = now()
        persist()
    return result


def launch(*, inputs: str = str(DEFAULT_INPUTS), run_name: str = DEFAULT_RUN_NAME, execute: bool = False) -> dict:
    plan = build_plan(Path(inputs).resolve(), run_name)
    verify_runtime(plan)
    if not execute:
        return {"dry_run": True, "remote_calls": False, "plan": plan,
                "deferred_cpu_checks": ["remote receipt/input/checkpoint hashes and FP32/atlas scope",
                                        "tokenizer IDs and completion lengths", "all six saved original response rescores"]}
    verify_files(plan["input_files_sha256"])
    directory = LOCAL_RUN_ROOT / run_name
    directory.mkdir(parents=True, exist_ok=False)
    plan.update(reservation_id=uuid.uuid4().hex, created_at=now(), status="reserved")
    receipt = directory / "launch.json"
    write_json_atomic(receipt, plan)
    try:
        with app.run(detach=True):
            verify_runtime(plan)
            verify_files(plan["input_files_sha256"])
            reserve.remote(plan)
            call = coordinate.spawn(plan)
            plan.update(status="spawned", coordinator_call_id=call.object_id, spawned_at=now())
            write_json_atomic(receipt, plan)
    except BaseException as exc:
        # A detached coordinator owns cancellation even if writing locally fails.
        write_json_atomic(receipt, {**plan, "status": "launch_failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return {"receipt": str(receipt), "coordinator_call_id": call.object_id,
            "remote_result": str(RUN_ROOT / "pipelines" / run_name / "result.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=str(DEFAULT_INPUTS), help="unchanged parent dataset_inputs.json")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--execute", action="store_true", help="OPT IN to the detached bounded Modal pipeline")
    args = parser.parse_args()
    print(json.dumps(launch(inputs=args.inputs, run_name=args.run_name, execute=args.execute), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
