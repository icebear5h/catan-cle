"""Validate the approved config and build the immutable launch plan."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from sft.json_types import JsonDict, JsonValue, as_dict, loads_json
from sft.launchers._json import at, at_dict, at_str
from sft.launchers._train_config import train_config
from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.paths import PROJECT_ROOT, resolve_dataset_asset, resolve_dataset_image
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import (
    TrainConfig,
    inspect_jsonl_contract,
    iter_jsonl,
    load_token_inventory,
    normalize_training_config,
    sha256_file,
)

from ._base import (
    COORDINATOR_TIMEOUT,
    FAMILY_STEPS,
    FIXED_CONFIG,
    GPU_TIMEOUT,
    NEW_LABELS,
    NEW_PANEL_TASKS,
    PANEL_BUDGETS,
    STARTUP_TIMEOUT,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: str | Path) -> JsonDict:
    return as_dict(loads_json(Path(path).read_text()))


def source_hashes() -> dict[str, str]:
    return {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
            for package in ("sft", "cle", "evals", "data_pipeline")
            for p in sorted((PROJECT_ROOT / package).rglob("*.py"))}


def validate_run_name(run_name: str) -> None:
    if not run_name or len(run_name) > 100 or not run_name.replace("-", "").isalnum() or not run_name.isascii():
        raise ValueError("run name must contain only ASCII letters, digits and hyphens")
    if str(launcher.RUN_ROOT / run_name) == str(Path(launcher.PARENT_CHECKPOINT).parents[1]):
        raise ValueError("continuation must not overwrite the parent run")


def validate_config(payload: JsonDict, run_name: str) -> TrainConfig:
    validate_run_name(run_name)
    config = train_config(payload)
    config.validate()
    expected: dict[str, object] = {**FIXED_CONFIG, "input_mode": "vision", "max_sequence_length": None,
                "initial_bundle": launcher.PARENT_CHECKPOINT,
                "output_dir": str(launcher.RUN_ROOT / run_name)}
    normalized = normalize_training_config(payload)
    if any(normalized.get(k) != v for k, v in expected.items()):
        raise ValueError("configuration differs from the approved 128-step continuation")
    if not config.eval_jsonl or config.per_device_eval_batch_size != 2:
        raise ValueError("matched teacher-forced fullboard64 evaluation is required")
    return config


def validate_mixture(rows: list[JsonDict]) -> JsonDict:
    if len(rows) != 1024:
        raise ValueError("training must contain exactly 1024 rows / 128 steps")
    ids = [r.get("id") or r.get("row_id") for r in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("training row IDs must be present and unique")
    steps: list[str] = []
    tasks: Counter[str] = Counter()
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        family = batch[0].get("training_family")
        if not isinstance(family, str) or family not in FAMILY_STEPS or any(r.get("training_family") != family for r in batch):
            raise ValueError(f"optimizer step {start // 8 + 1} is not family-homogeneous")
        for row in batch:
            task = row.get("task_type")
            if not isinstance(task, str) or not task or "curriculum_stage" in row:
                raise ValueError("each row needs top-level task_type and no curriculum_stage")
            metadata_task = as_dict(row.get("metadata") or {}).get("task_type", task)
            if metadata_task != task:
                raise ValueError("top-level and metadata task_type disagree")
            if family == "directions":
                valid = task in {f"{entity}_direction_{answer}" for entity in ("node", "tile")
                                 for answer in ("yes", "no", "token")}
            elif family == "adjacency_connectivity":
                valid = task in {f"{entity}_adjacent_{answer}" for entity in ("node", "tile")
                                 for answer in ("yes", "no")} | {"node_connected_yes", "node_connected_no"}
            else:
                valid = task == family
            if not valid:
                raise ValueError(f"task {task!r} does not belong to training family {family!r}")
            tasks[task] += 1
        steps.append(family)
    if Counter(steps) != FAMILY_STEPS:
        raise ValueError("optimizer step quotas must be 16 per spatial family and 32 readout")
    family_steps: JsonDict = dict(Counter(steps))
    step_families: list[JsonValue] = list(steps)
    task_rows: JsonDict = dict(tasks)
    family_step_share: JsonDict = {k: v / 128 for k, v in FAMILY_STEPS.items()}
    return {"rows": len(rows), "steps": len(steps), "family_steps": family_steps,
            "step_families": step_families, "task_rows": task_rows,
            "family_step_share": family_step_share}


def dataset_identity(path: str, image_root: str | None) -> JsonDict:
    """Hash ordered, unchanged row payloads AND pixels, independent of upload renaming."""
    rows = [r for _, r in iter_jsonl(Path(path))]
    images: dict[Path, str] = {}
    normalized: list[JsonDict] = []
    for row in rows:
        reference = evaluator.image_reference(row)
        image = (resolve_dataset_image(Path(image_root), reference) if image_root
                 else resolve_dataset_asset(Path(path), reference))
        if image not in images:
            images[image] = sha256_file(image)
        item = {k: v for k, v in row.items() if k not in ("image", "images", "metadata")}
        metadata: JsonDict = {k: v for k, v in as_dict(row.get("metadata") or {}).items()
                    if k not in ("eval_set_id", "eval_source_sha256")}
        item.update(metadata=metadata, image_sha256=images[image])
        normalized.append(item)
    ids = [r.get("id") or r.get("row_id") for r in rows]
    if not rows or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"empty dataset or duplicate/missing IDs: {path}")
    return {"rows": len(rows), "sha256": sha256_file(Path(path)),
            "content_sha256": digest(normalized), "unique_images": len(images)}


def check_identity(actual: JsonValue, expected: JsonValue) -> None:
    for key in ("rows", "content_sha256", "unique_images"):
        if at(actual, key) != at(expected, key):
            raise ValueError(f"input identity changed: {key}")


def build_plan(inputs_path: Path, run_name: str) -> JsonDict:
    validate_run_name(run_name)
    if (launcher.LOCAL_RUN_ROOT / run_name).exists():
        raise FileExistsError(launcher.LOCAL_RUN_ROOT / run_name)
    inputs = read_json(inputs_path)
    if inputs.get("schema") != "catan_spatial_continuation_inputs/v1":
        raise ValueError("unsupported dataset_inputs schema")
    if set(as_dict(inputs.get("new_panels", {}))) != set(NEW_PANEL_TASKS.values()):
        raise ValueError("exactly node_tiles, shortest_node_path, local_node_tiles and dice_production panels are required")
    for key in ("train_jsonl", "image_root", "token_inventory", "metadata"):
        if not Path(at_str(inputs, key)).is_absolute():
            raise ValueError(f"manifest {key} must be absolute")
    # Metadata is provenance, not a second, speculative schema for step ownership.
    metadata = loads_json(Path(at_str(inputs, "metadata")).read_text())
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    load_token_inventory(at_str(inputs, "token_inventory"))
    parent = read_json(launcher.PARENT_RESULT)
    if parent["status"] != "completed" or at(parent, "evaluation", "status") != "completed":
        raise ValueError("parent run is incomplete")
    if at(parent, "evaluation", "checkpoint") != launcher.PARENT_CHECKPOINT:
        raise ValueError("parent evaluation is not the approved latest checkpoint")
    parent_config = at_dict(parent, "config")
    config: JsonDict = asdict(replace(train_config(parent_config), **FIXED_CONFIG,
                           train_jsonl=at_str(inputs, "train_jsonl"), image_root=at_str(inputs, "image_root"),
                           token_inventory=at_str(inputs, "token_inventory"),
                           eval_jsonl=launcher.OLD_PANELS["fullboard"]["eval_jsonl"],
                           eval_image_root=launcher.OLD_PANELS["fullboard"]["image_root"],
                           initial_bundle=at_str(parent, "evaluation", "checkpoint"),
                           output_dir=str(launcher.RUN_ROOT / run_name)))
    validate_config(config, run_name)
    train_rows = [r for _, r in iter_jsonl(Path(at_str(inputs, "train_jsonl")))]
    mixture = validate_mixture(train_rows)
    inspect_jsonl_contract(at_str(inputs, "train_jsonl"), at_str(inputs, "image_root"), require_curriculum=False)
    panels: JsonDict = {}
    for label, (count, batch, budget) in PANEL_BUDGETS.items():
        panel: JsonDict = dict(launcher.OLD_PANELS[label] if label in launcher.OLD_PANELS else at_dict(inputs, "new_panels", NEW_PANEL_TASKS[label]))
        eval_jsonl, image_root = at_str(panel, "eval_jsonl"), at_str(panel, "image_root")
        if not all(Path(p).is_absolute() for p in (eval_jsonl, image_root)):
            raise ValueError("panel paths must be absolute")
        if label in NEW_LABELS and (panel.get("max_new_tokens"), panel.get("batch_size")) != (budget, batch):
            raise ValueError(f"unapproved panel budget: {label}")
        identity = dataset_identity(eval_jsonl, image_root)
        if identity["rows"] != count or (label in launcher.OLD_PANELS and identity["sha256"] != panel["sha256"]):
            raise ValueError(f"fixed baseline hash or panel row count differs: {label}")
        if label in NEW_LABELS:
            tasks = {evaluator.evaluation_metadata(r, image_variant="original").get("task_type")
                     for _, r in iter_jsonl(Path(eval_jsonl))}
            if tasks != {NEW_PANEL_TASKS[label]}:
                raise ValueError(f"panel contains the wrong task: {label}")
        panels[label] = {"eval_jsonl": eval_jsonl, "image_root": image_root,
                         "identity": identity, "batch_size": batch, "max_new_tokens": budget}
    parent_launch = read_json(launcher.PARENT_RESULT.with_name("launch.json"))
    baselines: JsonDict = {
        "spatial": {"summary": read_json(launcher.SPATIAL_RECEIPT), "summary_sha256": sha256_file(launcher.SPATIAL_RECEIPT),
                    "output_dir": "/runs/qwen-series-eval/full-board-new-layouts-ck128-spatial-answer-only-v1-20260908-r01"},
        "fullboard": {"summary": read_json(launcher.BOARD_RECEIPT), "summary_sha256": sha256_file(launcher.BOARD_RECEIPT),
                      "output_dir": str(Path(launcher.PARENT_CHECKPOINT).parents[1] / "generated-eval-128-full64/heldout")},
    }
    source_sha256: JsonDict = dict(launcher.source_hashes())
    input_files_sha256: JsonDict = {
        str(p): sha256_file(p) for p in
        (inputs_path, Path(at_str(inputs, "metadata")), Path(at_str(inputs, "token_inventory")), launcher.PARENT_RESULT)}
    return {"schema": "catan_spatial_continuation_launch/v1", "run_name": run_name,
            "config": config, "config_sha256": digest(config), "parent_config": parent_config,
            "dataset_inputs": inputs, "panels": panels,
            "train_identity": dataset_identity(at_str(inputs, "train_jsonl"), at_str(inputs, "image_root")),
            "mixture": mixture, "saved_baselines": baselines,
            # Historical receipt key: preserve the original path recorded by the parent.
            "legacy_pilot_sha256": at(parent_launch, "source_sha256", "sft/modal_full_board_pilot.py"),
            "source_sha256": source_sha256,
            "metadata": {"path": inputs["metadata"], "sha256": sha256_file(Path(at_str(inputs, "metadata"))),
                         "schema": metadata.get("schema")},
            "input_files_sha256": input_files_sha256,
            "policy": {"stages": ["cpu_preflight", "new_parent_baselines", "train128", "post_all_six"],
                       "gpu_timeout_per_stage_seconds": GPU_TIMEOUT,
                       "startup_timeout_seconds": STARTUP_TIMEOUT,
                       "coordinator_timeout_seconds": COORDINATOR_TIMEOUT,
                       "fresh_optimizer_and_schedule": True, "blank_controls": False,
                       "teacher_forced_eval": "fullboard64 every 32 steps; matched parent config, costs included in training timeout",
                       "saved_baseline_scoring": "rescore retained original-image responses with this launch's scorer; no GPU rerun"}}
