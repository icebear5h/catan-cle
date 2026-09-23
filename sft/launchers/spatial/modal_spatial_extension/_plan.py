"""Config validation, plan building and input/parent audits."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from sft.json_types import JsonDict, JsonValue, as_dict, as_int, opt_str
from sft.launchers._json import at, at_dict, at_str
from sft.launchers._train_config import train_config
from sft.launchers.spatial import modal_spatial_extension as extension
from sft.launchers.spatial.modal_spatial_continuation import (
    NEW_PANEL_TASKS,
    PANEL_BUDGETS,
    PREFLIGHT_TIMEOUT,
    STARTUP_TIMEOUT,
    check_identity,
    checkpoint_audit,
    dataset_identity,
    digest,
    evaluation_conditions,
    read_json,
    validate_mixture,
    validate_run_name,
)
from sft.scripts.train.train_trl_catan_vision import (
    TrainConfig,
    inspect_jsonl_contract,
    iter_jsonl,
    load_token_inventory,
    normalize_training_config,
    sha256_file,
)

from ._base import (
    CHECKPOINT_STEPS,
    CONFIG_CHANGES,
    COORDINATOR_TIMEOUT,
    PARENT_RUN,
    POST_TIMEOUT,
    TRAIN_TIMEOUT,
)


def validate_config(payload: JsonDict, run_name: str, parent_config: JsonDict) -> TrainConfig:
    validate_run_name(run_name)
    if run_name == PARENT_RUN:
        raise ValueError("extension must not overwrite its parent")
    parent = train_config(parent_config)
    parent.validate()
    if (parent.max_steps != 128 or parent.seed != 44 or parent.profile != "vision_tokens_lora"
            or parent.per_device_train_batch_size != 4 or parent.gradient_accumulation_steps != 2
            or parent.per_device_eval_batch_size != 2 or not parent.eval_jsonl
            or parent.token_init != "keep" or parent.require_curriculum
            or parent.input_mode != "vision" or parent.max_sequence_length is not None
            or parent.publish_to_hub or parent.resume_from_checkpoint or parent.olora):
        raise ValueError("parent configuration is not the approved completed mixed run")
    expected: JsonDict = asdict(replace(parent, **CONFIG_CHANGES, initial_bundle=extension.PARENT_CHECKPOINT,
                              output_dir=str(extension.RUN_ROOT / run_name)))
    if normalize_training_config(payload) != normalize_training_config(expected):
        raise ValueError("configuration must inherit ALL parent settings except approved extension fields")
    config = train_config(payload)
    config.validate()
    return config


def build_plan(inputs_path: Path, run_name: str) -> JsonDict:
    parent = read_json(extension.PARENT_RESULT)
    post_path = extension.PARENT_RESULT.parent / "post/result.json"
    post = read_json(post_path)
    launch_path = extension.PARENT_RESULT.with_name("launch.json")
    parent_launch = read_json(launch_path)
    if (parent["status"] != "completed" or parent["phase"] != "completed"
            or post["status"] != "completed"
            or any(at(parent, "stages", key, "status") != "completed" for key in ("preflight", "training", "post"))
            or at(parent, "stages", "post", "result") != post
            or normalize_training_config(at_dict(parent_launch, "config")) != normalize_training_config(at_dict(parent, "config"))
            or digest(parent_launch["config"]) != parent_launch["config_sha256"]):
        raise ValueError("parent launch/result/post receipts disagree or are incomplete")
    if (post["checkpoint"] != extension.PARENT_CHECKPOINT
            or at(post, "checkpoint_audit", "checkpoint") != extension.PARENT_CHECKPOINT
            or set(at_dict(post, "panels")) != set(PANEL_BUDGETS)):
        raise ValueError("parent must supply the checkpoint-128 audit and ALL SIX post panels")
    parent_config = at_dict(parent, "config")
    config: JsonDict = asdict(replace(train_config(parent_config), **CONFIG_CHANGES,
                            initial_bundle=extension.PARENT_CHECKPOINT, output_dir=str(extension.RUN_ROOT / run_name)))
    validate_config(config, run_name, parent_config)
    if (extension.LOCAL_RUN_ROOT / run_name).exists():
        raise FileExistsError(extension.LOCAL_RUN_ROOT / run_name)
    inputs = read_json(inputs_path)
    if inputs != parent_launch["dataset_inputs"]:
        raise ValueError("dataset_inputs must be the unchanged parent manifest")
    local_hashes: JsonDict = {}
    for path in (inputs_path, Path(at_str(inputs, "metadata")), Path(at_str(inputs, "token_inventory"))):
        value = sha256_file(path)
        if value != at(parent_launch, "input_files_sha256", str(path)):
            raise ValueError(f"parent input receipt changed: {path}")
        local_hashes[str(path)] = value
    inventory = load_token_inventory(at_str(inputs, "token_inventory"))
    if inventory["atlas_tokens"] != at(post, "checkpoint_audit", "semantic_tokens", "tokens"):
        raise ValueError("parent atlas inventory differs")
    train_jsonl, image_root = at_str(inputs, "train_jsonl"), at_str(inputs, "image_root")
    train_identity = dataset_identity(train_jsonl, image_root)
    if train_identity != parent_launch["train_identity"]:
        raise ValueError("frozen ordered training dataset changed")
    rows = [row for _, row in iter_jsonl(Path(train_jsonl))]
    mixture = validate_mixture(rows)
    if mixture != parent_launch["mixture"]:
        raise ValueError("parent optimizer-step ordering changed")
    inspect_jsonl_contract(train_jsonl, image_root, require_curriculum=False)
    preflight = at_dict(parent, "stages", "preflight", "result")
    check_identity(preflight["train_identity"], train_identity)
    panels: JsonDict = {}
    baselines: JsonDict = {}
    remote_hashes: JsonDict = {str(extension.RUN_ROOT / "pipelines" / PARENT_RUN / name): sha256_file(path)
                     for name, path in (("result.json", extension.PARENT_RESULT), ("post/result.json", post_path))}
    for label, (count, batch, budget) in PANEL_BUDGETS.items():
        local: JsonDict = dict(extension.OLD_PANELS[label]) if label in extension.OLD_PANELS else at_dict(inputs, "new_panels", NEW_PANEL_TASKS[label])
        local_eval_jsonl = at_str(local, "eval_jsonl")
        identity = dataset_identity(local_eval_jsonl, opt_str(local["image_root"]))
        local_hashes[local_eval_jsonl] = identity["sha256"]
        saved = at_dict(post, "panels", label)
        if (identity != at(parent_launch, "panels", label, "identity") or identity["rows"] != count
                or saved["conditions"] != evaluation_conditions(label)
                or saved["scorer_sha256"] != digest(post["source_sha256"])
                or at(saved, "summary", "eval_source_sha256") != identity["sha256"]):
            raise ValueError(f"frozen panel identity/conditions differ: {label}")
        check_identity(saved["identity"], identity)
        output = extension.RUN_ROOT / "pipelines" / PARENT_RUN / "post" / label
        if saved["records_path"] != str(output / "records.jsonl"):
            raise ValueError(f"parent records path differs: {label}")
        summary_path = post_path.parent / f"{label}-summary.json"
        records_path = post_path.parent / f"{label}-records.jsonl"
        if (read_json(summary_path) != saved["summary"]
                or sha256_file(records_path) != saved["records_sha256"]):
            raise ValueError(f"local saved parent responses or summary differ: {label}")
        for path in (summary_path, records_path):
            local_hashes[str(path)] = sha256_file(path)
        panels[label] = {"eval_jsonl": at(saved, "summary", "eval_jsonl"),
                         "image_root": at(saved, "summary", "image_root"), "identity": saved["identity"],
                         "batch_size": batch, "max_new_tokens": budget, "local_identity": identity}
        baselines[label] = {**saved, "output_dir": str(output),
                            "summary_sha256": local_hashes[str(summary_path)]}
    check_identity(preflight["teacher_eval_identity"], at(panels, "fullboard", "identity"))
    for path in (extension.PARENT_RESULT, post_path, launch_path, Path(train_jsonl)):
        local_hashes[str(path)] = sha256_file(path)
    family_steps: JsonDict = {k: 2 * as_int(v) for k, v in at_dict(mixture, "family_steps").items()}
    checkpoint_steps: list[JsonValue] = list(CHECKPOINT_STEPS)
    source_sha256: JsonDict = dict(extension.source_hashes())
    return {"schema": "catan_spatial_extension_launch/v1", "run_name": run_name,
            "config": config, "config_sha256": digest(config), "parent_config": parent_config,
            "parent_config_sha256": digest(parent_config), "parent_checkpoint": extension.PARENT_CHECKPOINT,
            "parent_audit": post["checkpoint_audit"], "dataset_inputs": inputs,
            "train_identity": preflight["train_identity"], "teacher_eval_identity": preflight["teacher_eval_identity"],
            "token_inventory_sha256": local_hashes[at_str(inputs, "token_inventory")],
            "mixture": mixture, "panels": panels, "saved_baselines": baselines,
            "input_files_sha256": local_hashes, "remote_receipts_sha256": remote_hashes,
            "source_sha256": source_sha256,
            "policy": {"stages": ["cpu_preflight", "train256", "post_all_six"],
                       "additional_optimizer_steps": 256, "cumulative_mixed_updates": 384,
                       "dataset_passes": 2, "family_steps": family_steps,
                       "checkpoint_steps": checkpoint_steps, "fresh_optimizer_and_schedule": True,
                       "preflight_timeout_seconds": PREFLIGHT_TIMEOUT, "train_timeout_seconds": TRAIN_TIMEOUT,
                       "post_timeout_seconds": POST_TIMEOUT, "startup_timeout_seconds": STARTUP_TIMEOUT,
                       "coordinator_timeout_seconds": COORDINATOR_TIMEOUT, "retries": 0,
                       "blank_controls": False, "reuse_parent_uploaded_data": True,
                       "saved_baseline_scoring": "all six original post panels, CPU rescore before training"}}


def verify_runtime(plan: JsonDict) -> None:
    validate_config(at_dict(plan, "config"), at_str(plan, "run_name"), at_dict(plan, "parent_config"))
    if (plan["parent_checkpoint"] != extension.PARENT_CHECKPOINT
            or digest(plan["config"]) != plan["config_sha256"]
            or digest(plan["parent_config"]) != plan["parent_config_sha256"]
            or extension.source_hashes() != plan["source_sha256"]):
        raise ValueError("source/config/checkpoint changed since launch planning")
    if set(at_dict(plan, "panels")) != set(PANEL_BUDGETS) or set(at_dict(plan, "saved_baselines")) != set(PANEL_BUDGETS):
        raise ValueError("all six frozen panels and saved baselines are required")
    for label, panel in at_dict(plan, "panels").items():
        if (at(panel, "identity", "rows"), at(panel, "batch_size"), at(panel, "max_new_tokens")) != PANEL_BUDGETS[label]:
            raise ValueError("unapproved panel budget")


def verify_files(hashes: JsonValue) -> None:
    for path, expected in as_dict(hashes).items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"pinned file changed: {path}")


def audit_inputs(plan: JsonDict) -> JsonDict:
    config = at_dict(plan, "config")
    identities: JsonDict = {
        "train_identity": dataset_identity(at_str(config, "train_jsonl"), opt_str(config["image_root"])),
        "teacher_eval_identity": dataset_identity(at_str(config, "eval_jsonl"), opt_str(config["eval_image_root"]))}
    for key, actual in identities.items():
        if actual != plan[key]:
            raise ValueError(f"uploaded input bytes/pixels changed: {key}")
    if sha256_file(Path(at_str(config, "token_inventory"))) != plan["token_inventory_sha256"]:
        raise ValueError("uploaded token inventory changed")
    return identities


def audit_parent(plan: JsonDict) -> JsonDict:
    return checkpoint_audit(Path(at_str(plan, "parent_checkpoint")), at_dict(plan, "parent_config"), parent=True,
                            expected_step=128, expected_audit=at_dict(plan, "parent_audit"))
