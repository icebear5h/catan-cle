"""One opt-in, bounded spatial continuation. Importing/dry-running never contacts Modal.

Run ``python -m sft.modal_spatial_continuation --help`` locally. Only --execute
reserves remote paths, uploads inputs, and spawns the detached CPU coordinator.
Saved original-image generations are independently rescored before training;
there are no new blank controls, probes, sweeps, or conditional training stages.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import uuid
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import modal
import torch
from safetensors import safe_open
from transformers import AddedToken, AutoTokenizer

from sft.modal_full_board_pilot import (
    GPU_OPTIONS, SNAPSHOT, VOLUMES, app, hf_cache, sft_data, sft_runs,
    train as pilot_train, training_image, upload_training_bundle,
)
from sft.modal_qwen_series_eval import upload_eval_jsonl
from sft.paths import PROJECT_ROOT, resolve_dataset_asset, resolve_dataset_image
from sft.scripts import eval_qwen_vl_adapter as evaluator
from sft.scripts.train_trl_catan_vision import (
    TrainConfig, inspect_jsonl_contract, iter_jsonl, load_token_inventory,
    normalize_training_config, sha256_file, write_json_atomic,
)

RUN_ROOT = Path("/runs/catan-vision-sft")
LOCAL_RUN_ROOT = PROJECT_ROOT / "artifacts/runs/sft"
DEFAULT_INPUTS = PROJECT_ROOT / "artifacts/generated/board_recognition/spatial_continuation_v1/dataset_inputs.json"
PARENT_RESULT = LOCAL_RUN_ROOT / "full-board-new-layouts-20260907/result.json"
PARENT_CHECKPOINT = str(RUN_ROOT / "full-board-new-layouts-20260907/checkpoints/checkpoint-128")
VISUAL_SHA256 = "3535adfa86dbc4675f612f98995222f2f15619d151b5aff2f7537eab147b7183"
SPATIAL_RECEIPT = LOCAL_RUN_ROOT / "full-board-new-layouts-ck128-spatial-answer-only-v1-20260908-r01/summary.json"
BOARD_RECEIPT = LOCAL_RUN_ROOT / "full-board-new-layouts-ck128-spatial-choice-order-v1-20260908-r01/prior_full_board_summary.json"
OLD_PANELS = {
    "spatial": {
        "eval_jsonl": str(PROJECT_ROOT / "artifacts/generated/board_recognition/replay_v1/evals/spatial_choice_order_answer_only_validation_v1.jsonl"),
        "image_root": str(PROJECT_ROOT / "artifacts/generated/board_recognition/replay_v1/images"),
        "sha256": "53c4330d955d369833ca6024307b7b725a4cc46117fe6c1efc6170e7e7486786",
    },
    "fullboard": {
        "eval_jsonl": str(LOCAL_RUN_ROOT / "full-board-epoch2-20260907/validation64.jsonl"),
        "image_root": str(PROJECT_ROOT / "artifacts/generated/board_recognition/full_board_diverse_v1/full_board_readout_v1/images"),
        "sha256": "902d2833b9fe705888e834822488593c14c1c08e9e4c9939464d1501655cbc8e",
    },
}
# label: (rows, batch size, short AND long generation allowance)
PANEL_BUDGETS = {"spatial": (120, 48, 16), "fullboard": (64, 4, 1280),
                 "node_tiles": (54, 16, 128), "paths": (64, 16, 128),
                 "local": (64, 16, 128), "production": (64, 16, 128)}
FAMILY_STEPS = dict.fromkeys(("directions", "adjacency_connectivity", "node_tiles", "shortest_node_path",
                            "local_node_tiles", "dice_production"), 16) | {"full_board_readout": 32}
NEW_PANEL_TASKS = {"node_tiles": "node_tiles", "paths": "shortest_node_path",
                   "local": "local_node_tiles", "production": "dice_production"}
NEW_LABELS = tuple(NEW_PANEL_TASKS)
GPU_TIMEOUT = 3600
STARTUP_TIMEOUT = 300
PREFLIGHT_TIMEOUT = 1200
WAIT_GRACE = 60
COORDINATOR_TIMEOUT = 14400
# The trainer image already mounts sft/cle/evals, but not the graph/data package
# used by the task-aware spatial scorer. Keep its pinned dependency stack.
continuation_image = training_image.add_local_python_source("data_pipeline")
CONTINUATION_GPU_OPTIONS = {**GPU_OPTIONS, "image": continuation_image, "timeout": GPU_TIMEOUT}
CPU_OPTIONS = dict(image=continuation_image, volumes=VOLUMES, retries=0,
                   startup_timeout=STARTUP_TIMEOUT, max_containers=1, scaledown_window=2)
FIXED_CONFIG = dict(max_steps=128, per_device_train_batch_size=4, gradient_accumulation_steps=2,
                    save_steps=32, eval_steps=32, save_total_limit=4, require_curriculum=False,
                    token_init="keep", resume_from_checkpoint=None, publish_to_hub=False,
                    profile="vision_tokens_lora", frozen_bundle=None, visual_delta_factors=None,
                    lora_rank=8, lora_alpha=16, lora_dropout=0.05, learning_rate=5e-4,
                    language_lora_learning_rate=1e-4, vision_learning_rate=5e-6,
                    merger_learning_rate=5e-5, warmup_ratio=0.1, patch_loss_weight=0.0,
                    spatial_target_mode="correct", model_id=SNAPSHOT)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def source_hashes() -> dict:
    return {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
            for package in ("sft", "cle", "evals", "data_pipeline")
            for p in sorted((PROJECT_ROOT / package).rglob("*.py"))}


def validate_run_name(run_name: str) -> None:
    if not run_name or len(run_name) > 100 or not run_name.replace("-", "").isalnum() or not run_name.isascii():
        raise ValueError("run name must contain only ASCII letters, digits and hyphens")
    if str(RUN_ROOT / run_name) == str(Path(PARENT_CHECKPOINT).parents[1]):
        raise ValueError("continuation must not overwrite the parent run")


def validate_config(payload: dict, run_name: str) -> TrainConfig:
    validate_run_name(run_name)
    config = TrainConfig(**payload)
    config.validate()
    expected = {**FIXED_CONFIG, "input_mode": "vision", "max_sequence_length": None,
                "initial_bundle": PARENT_CHECKPOINT,
                "output_dir": str(RUN_ROOT / run_name)}
    normalized = normalize_training_config(payload)
    if any(normalized.get(k) != v for k, v in expected.items()):
        raise ValueError("configuration differs from the approved 128-step continuation")
    if not config.eval_jsonl or config.per_device_eval_batch_size != 2:
        raise ValueError("matched teacher-forced fullboard64 evaluation is required")
    return config


def validate_mixture(rows: list[dict]) -> dict:
    if len(rows) != 1024:
        raise ValueError("training must contain exactly 1024 rows / 128 steps")
    ids = [r.get("id") or r.get("row_id") for r in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("training row IDs must be present and unique")
    steps = []
    tasks = Counter()
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        family = batch[0].get("training_family")
        if family not in FAMILY_STEPS or any(r.get("training_family") != family for r in batch):
            raise ValueError(f"optimizer step {start // 8 + 1} is not family-homogeneous")
        for row in batch:
            task = row.get("task_type")
            if not isinstance(task, str) or not task or "curriculum_stage" in row:
                raise ValueError("each row needs top-level task_type and no curriculum_stage")
            metadata_task = (row.get("metadata") or {}).get("task_type", task)
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
    return {"rows": len(rows), "steps": len(steps), "family_steps": dict(Counter(steps)),
            "step_families": steps, "task_rows": dict(tasks),
            "family_step_share": {k: v / 128 for k, v in FAMILY_STEPS.items()}}


def dataset_identity(path: str, image_root: str | None) -> dict:
    """Hash ordered, unchanged row payloads AND pixels, independent of upload renaming."""
    rows = [r for _, r in iter_jsonl(Path(path))]
    images = {}
    normalized = []
    for row in rows:
        reference = evaluator.image_reference(row)
        image = (resolve_dataset_image(Path(image_root), reference) if image_root
                 else resolve_dataset_asset(Path(path), reference))
        if image not in images:
            images[image] = sha256_file(image)
        item = {k: v for k, v in row.items() if k not in ("image", "images", "metadata")}
        metadata = {k: v for k, v in (row.get("metadata") or {}).items()
                    if k not in ("eval_set_id", "eval_source_sha256")}
        item.update(metadata=metadata, image_sha256=images[image])
        normalized.append(item)
    ids = [r.get("id") or r.get("row_id") for r in rows]
    if not rows or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"empty dataset or duplicate/missing IDs: {path}")
    return {"rows": len(rows), "sha256": sha256_file(Path(path)),
            "content_sha256": digest(normalized), "unique_images": len(images)}


def check_identity(actual: dict, expected: dict) -> None:
    for key in ("rows", "content_sha256", "unique_images"):
        if actual[key] != expected[key]:
            raise ValueError(f"input identity changed: {key}")


def build_plan(inputs_path: Path, run_name: str) -> dict:
    validate_run_name(run_name)
    if (LOCAL_RUN_ROOT / run_name).exists():
        raise FileExistsError(LOCAL_RUN_ROOT / run_name)
    inputs = read_json(inputs_path)
    if inputs.get("schema") != "catan_spatial_continuation_inputs/v1":
        raise ValueError("unsupported dataset_inputs schema")
    if set(inputs.get("new_panels", {})) != set(NEW_PANEL_TASKS.values()):
        raise ValueError("exactly node_tiles, shortest_node_path, local_node_tiles and dice_production panels are required")
    for key in ("train_jsonl", "image_root", "token_inventory", "metadata"):
        if not Path(inputs[key]).is_absolute():
            raise ValueError(f"manifest {key} must be absolute")
    # Metadata is provenance, not a second, speculative schema for step ownership.
    metadata = read_json(inputs["metadata"])
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    load_token_inventory(inputs["token_inventory"])
    parent = read_json(PARENT_RESULT)
    if parent["status"] != "completed" or parent["evaluation"]["status"] != "completed":
        raise ValueError("parent run is incomplete")
    if parent["evaluation"]["checkpoint"] != PARENT_CHECKPOINT:
        raise ValueError("parent evaluation is not the approved latest checkpoint")
    config = asdict(replace(TrainConfig(**parent["config"]), **FIXED_CONFIG,
                           train_jsonl=inputs["train_jsonl"], image_root=inputs["image_root"],
                           token_inventory=inputs["token_inventory"],
                           eval_jsonl=OLD_PANELS["fullboard"]["eval_jsonl"],
                           eval_image_root=OLD_PANELS["fullboard"]["image_root"],
                           initial_bundle=parent["evaluation"]["checkpoint"],
                           output_dir=str(RUN_ROOT / run_name)))
    validate_config(config, run_name)
    train_rows = [r for _, r in iter_jsonl(Path(inputs["train_jsonl"]))]
    mixture = validate_mixture(train_rows)
    inspect_jsonl_contract(inputs["train_jsonl"], inputs["image_root"], require_curriculum=False)
    panels = {}
    for label, (count, batch, budget) in PANEL_BUDGETS.items():
        panel = dict(OLD_PANELS[label] if label in OLD_PANELS else inputs["new_panels"][NEW_PANEL_TASKS[label]])
        if not all(Path(panel[k]).is_absolute() for k in ("eval_jsonl", "image_root")):
            raise ValueError("panel paths must be absolute")
        if label in NEW_LABELS and (panel.get("max_new_tokens"), panel.get("batch_size")) != (budget, batch):
            raise ValueError(f"unapproved panel budget: {label}")
        identity = dataset_identity(panel["eval_jsonl"], panel["image_root"])
        if identity["rows"] != count or (label in OLD_PANELS and identity["sha256"] != panel["sha256"]):
            raise ValueError(f"fixed baseline hash or panel row count differs: {label}")
        if label in NEW_LABELS:
            tasks = {evaluator.evaluation_metadata(r, image_variant="original").get("task_type")
                     for _, r in iter_jsonl(Path(panel["eval_jsonl"]))}
            if tasks != {NEW_PANEL_TASKS[label]}:
                raise ValueError(f"panel contains the wrong task: {label}")
        panels[label] = {"eval_jsonl": panel["eval_jsonl"], "image_root": panel["image_root"],
                         "identity": identity, "batch_size": batch, "max_new_tokens": budget}
    parent_launch = read_json(PARENT_RESULT.with_name("launch.json"))
    baselines = {
        "spatial": {"summary": read_json(SPATIAL_RECEIPT), "summary_sha256": sha256_file(SPATIAL_RECEIPT),
                    "output_dir": "/runs/qwen-series-eval/full-board-new-layouts-ck128-spatial-answer-only-v1-20260908-r01"},
        "fullboard": {"summary": read_json(BOARD_RECEIPT), "summary_sha256": sha256_file(BOARD_RECEIPT),
                      "output_dir": str(Path(PARENT_CHECKPOINT).parents[1] / "generated-eval-128-full64/heldout")},
    }
    return {"schema": "catan_spatial_continuation_launch/v1", "run_name": run_name,
            "config": config, "config_sha256": digest(config), "parent_config": parent["config"],
            "dataset_inputs": inputs, "panels": panels,
            "train_identity": dataset_identity(inputs["train_jsonl"], inputs["image_root"]),
            "mixture": mixture, "saved_baselines": baselines,
            "legacy_pilot_sha256": parent_launch["source_sha256"]["sft/modal_full_board_pilot.py"],
            "source_sha256": source_hashes(),
            "metadata": {"path": inputs["metadata"], "sha256": sha256_file(Path(inputs["metadata"])),
                         "schema": metadata.get("schema")},
            "input_files_sha256": {str(p): sha256_file(p) for p in
                                   (inputs_path, Path(inputs["metadata"]), Path(inputs["token_inventory"]), PARENT_RESULT)},
            "policy": {"stages": ["cpu_preflight", "new_parent_baselines", "train128", "post_all_six"],
                       "gpu_timeout_per_stage_seconds": GPU_TIMEOUT,
                       "startup_timeout_seconds": STARTUP_TIMEOUT,
                       "coordinator_timeout_seconds": COORDINATOR_TIMEOUT,
                       "fresh_optimizer_and_schedule": True, "blank_controls": False,
                       "teacher_forced_eval": "fullboard64 every 32 steps; matched parent config, costs included in training timeout",
                       "saved_baseline_scoring": "rescore retained original-image responses with this launch's scorer; no GPU rerun"}}


def checkpoint_audit(checkpoint: Path, config: dict, *, parent: bool,
                     expected_step: int = 128, expected_audit: dict | None = None) -> dict:
    required = ("adapter_config.json", "adapter_model.safetensors", "visual_model.safetensors",
                "trainable_parameters.json", "training_config.json", "tokenizer_config.json",
                "tokenizer.json", "trainer_state.json")
    hashes = {name: sha256_file(checkpoint / name) for name in required}
    if expected_audit is not None and (str(checkpoint) != expected_audit["checkpoint"]
                                      or hashes != expected_audit["files_sha256"]):
        raise ValueError("checkpoint identity or file hashes differ from the pinned audit")
    if parent and expected_audit is None and hashes["visual_model.safetensors"] != VISUAL_SHA256:
        raise ValueError("parent visual checkpoint SHA256 differs")
    saved = read_json(checkpoint / "training_config.json")
    if (normalize_training_config(saved) != normalize_training_config(config)
            or read_json(checkpoint / "trainer_state.json")["global_step"] != expected_step):
        raise ValueError("checkpoint configuration or global_step differs")
    adapter = read_json(checkpoint / "adapter_config.json")
    if (adapter["r"], adapter["lora_alpha"], adapter["lora_dropout"]) != (8, 16, 0.05):
        raise ValueError("checkpoint LoRA profile differs")
    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
        shapes = {k: tensors.get_slice(k).get_shape() for k in tensors.keys()}
    atlas = [shape for key, shape in shapes.items() if "trainable_tokens_delta" in key]
    lora_a = [shape for key, shape in shapes.items() if ".lora_A." in key]
    lora_b = [shape for key, shape in shapes.items() if ".lora_B." in key]
    if (len(atlas) != 2 or any(len(s) != 2 or s[0] != 154 for s in atlas)
            or not lora_a or len(lora_a) != len(lora_b)
            or any(len(s) != 2 or s[0] != 8 for s in lora_a)
            or any(len(s) != 2 or s[1] != 8 for s in lora_b)):
        raise ValueError("adapter tensor headers do not match rank8 LoRA and 154 input/output rows")
    scope = read_json(checkpoint / "trainable_parameters.json")
    if scope["errors"] or scope["profile"] != "vision_tokens_lora":
        raise ValueError("invalid checkpoint trainable scope")
    for name in ("vision", "merger", "atlas_input_rows", "atlas_output_rows", "language_lora"):
        if scope["groups"][name]["parameters"] <= 0:
            raise ValueError(f"missing trainable group {name}")
    if any(scope["groups"][name]["parameters"] for name in ("forbidden", "vision_lora")):
        raise ValueError("unapproved trainable parameters")
    for name in ("vision", "merger"):
        if set(scope["dtypes"][name]) != {"torch.float32"}:
            raise ValueError("visual/merger master weights must remain FP32")
    for name in ("atlas_input_rows", "atlas_output_rows"):
        shapes = [p["shape"] for p in scope["parameters"] if p["category"] == name]
        if len(shapes) != 1 or shapes[0][0] != 154:
            raise ValueError("exactly 154 trainable input and output atlas rows required")
    with safe_open(checkpoint / "visual_model.safetensors", framework="pt", device="cpu") as tensors:
        dtypes = Counter(tensors.get_slice(k).get_dtype() for k in tensors.keys())
    if dtypes != {"F32": 333}:
        raise ValueError("checkpoint must contain all 333 FP32 visual tensors")
    report = {"checkpoint": str(checkpoint), "files_sha256": hashes,
              "semantic_tokens": scope["semantic_tokens"], "visual_dtypes": dict(dtypes)}
    if expected_audit is not None and report != expected_audit:
        raise ValueError("checkpoint semantic/precision audit differs from the pinned audit")
    return report


def completion_audit(rows: list[dict], tokenizer, *, budget: int | None = None) -> dict:
    counts, exposure, maxima = Counter(), Counter(), {}
    for row in rows:
        family = row.get("training_family") or row.get("task_type") or (row.get("metadata") or {}).get("task_type", "unknown")
        length = len(tokenizer.encode(evaluator.expected_text(row), add_special_tokens=False))
        allowance = budget if budget is not None else (1280 if family == "full_board_readout" else 16 if family in ("directions", "adjacency_connectivity") else 128)
        if not 0 < length < allowance:
            raise ValueError(f"completion token length {length} reaches/exceeds {allowance}: {family}")
        counts[family] += 1
        exposure[family] += length
        maxima[family] = max(maxima.get(family, 0), length)
    total = sum(exposure.values())
    return {"max_tokens": maxima, "row_counts": dict(counts), "completion_tokens": dict(exposure),
            "completion_token_share": {k: v / total for k, v in exposure.items()},
            "note": "completion-token exposure is not optimizer-step or gradient share"}


def matched_baseline(label: str, saved: dict, panel: dict, checkpoint: dict, *,
                     expected_checkpoint: str | None = None,
                     expected_correct: int | None = None, strict_summary: bool = False) -> dict:
    """Rescore retained generations; explicit identities support later continuations.

    Strict mode pins record bytes, uploaded input bytes, metadata, every score and
    every derived summary field (only the rescoring timestamp is excluded).
    Historical callers retain their original checkpoint/count defaults.
    """
    expected_checkpoint = expected_checkpoint or PARENT_CHECKPOINT
    summary = saved["summary"]
    output = Path(saved["output_dir"])
    if (read_json(output / "summary.json") != summary
            or sha256_file(output / "summary.json") != saved["summary_sha256"]):
        raise ValueError("saved baseline summary differs from the remote original")
    evidence = summary["adapter_evidence"]
    if (summary["adapter_dir"] != expected_checkpoint or evidence["adapter_dir"] != expected_checkpoint
            or not evidence["adapter_loaded"] or summary["model_id"] != SNAPSHOT
            or evidence["visual_state"]["sha256"] != checkpoint["files_sha256"]["visual_model.safetensors"]
            or evidence["semantic_tokens"]["token_ids"] != checkpoint["semantic_tokens"]["token_ids"]):
        raise ValueError("saved baseline checkpoint identity differs")
    if (summary["image_variant"] != "original" or summary["reasoning_enabled"] is not False
            or summary["candidate_scoring"] is not False or summary["bits"] != 16
            or summary["batch_size"] != panel["batch_size"]):
        raise ValueError("saved baseline generation conditions differ")
    identity = dataset_identity(summary["eval_jsonl"], summary["image_root"])
    check_identity(identity, panel["identity"])
    if strict_summary and (identity != saved["identity"]
                           or summary["precision"]["preserve_visual_fp32"] is not True
                           or evidence["visual_state"].get("loaded") is not True
                           or summary["max_new_tokens"] != panel["max_new_tokens"]
                           or summary["long_max_new_tokens"] != panel["max_new_tokens"]
                           or saved["conditions"] != evaluation_conditions(label)
                           or sha256_file(output / "records.jsonl") != saved["records_sha256"]):
        raise ValueError("saved baseline records, input bytes or conditions differ")
    rows = [r for _, r in iter_jsonl(Path(summary["eval_jsonl"]))]
    if any(summary["long_max_new_tokens" if evaluator.is_long_answer(r) else "max_new_tokens"] != panel["max_new_tokens"] for r in rows):
        raise ValueError("saved baseline effective generation budget differs")
    if label == "spatial" and (summary["eval_source_sha256"] != OLD_PANELS[label]["sha256"]
                               or summary["precision"]["preserve_visual_fp32"] is not True):
        raise ValueError("spatial baseline source/precision differs")
    records = [r for _, r in iter_jsonl(output / "records.jsonl")]
    by_id = {r.get("id") or r.get("row_id"): r for r in rows}
    if Counter(r["id"] for r in records) != Counter(by_id.keys()):
        raise ValueError("saved baseline record IDs differ")
    for record in records:
        row = by_id[record["id"]]
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        if strict_summary and (record["metadata"] != metadata or record.get("candidate_score") is not None):
            raise ValueError("saved baseline record metadata or candidate scoring differs")
        if record["expected"] != evaluator.expected_text(row) or not isinstance(record["response"], str):
            raise ValueError("saved baseline targets or generations are incomplete")
        score = evaluator.score_response(record["expected"], record["response"],
                                         metadata=metadata)
        if strict_summary and score != record["score"]:
            raise ValueError("current scorer changes a saved baseline score")
        if score["correct"] != record["score"]["correct"]:
            raise ValueError("current scorer changes a saved baseline outcome")
        record["score"] = score
    rescored = evaluator.summarize(records)
    if expected_correct is None:
        expected_correct = 57 if label == "spatial" else 64
    if rescored["correct"] != expected_correct or any(rescored[k] != summary[k] for k in ("rows", "attempted", "correct")):
        raise ValueError("saved baseline aggregate differs")
    if label == "fullboard" and rescored["full_board"] != summary["full_board"]:
        raise ValueError("saved fullboard occupied/exact scoring differs")
    if strict_summary and any(value != summary.get(key) for key, value in rescored.items()
                              if key != "generated_at"):
        raise ValueError("saved baseline entire summary differs after rescoring")
    return {"summary": {**summary, **rescored}, "identity": panel["identity"],
            "records_path": str(output / "records.jsonl"), "records_sha256": sha256_file(output / "records.jsonl"),
            "reused": True, "original_only": True}


def reload_volumes() -> None:
    for volume in (hf_cache, sft_data, sft_runs):
        volume.reload()


def verify_runtime(plan: dict) -> None:
    validate_config(plan["config"], plan["run_name"])
    if digest(plan["config"]) != plan["config_sha256"]:
        raise ValueError("configuration hash changed since launch planning")
    if set(plan["panels"]) != set(PANEL_BUDGETS):
        raise ValueError("exactly the six approved evaluation panels are required")
    if set(plan["saved_baselines"]) != {"spatial", "fullboard"}:
        raise ValueError("both saved parent baselines must be checked before training")
    for label, panel in plan["panels"].items():
        count, batch, budget = PANEL_BUDGETS[label]
        if (panel["identity"]["rows"], panel["batch_size"], panel["max_new_tokens"]) != (count, batch, budget):
            raise ValueError("unapproved evaluation panel size or budget")
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("working-tree source hashes changed since launch planning")
    if "metadata" not in inspect.signature(evaluator.score_response).parameters:
        raise ValueError("task-aware score_response(..., metadata=...) is required before launch")


@app.function(**CPU_OPTIONS, cpu=(4.0, 4.0), memory=(8192, 8192), timeout=PREFLIGHT_TIMEOUT)
def continuation_preflight(plan: dict) -> dict:
    reload_volumes()
    verify_runtime(plan)
    config = plan["config"]
    if Path(config["output_dir"]).exists():
        raise FileExistsError(config["output_dir"])
    if not Path(SNAPSHOT).is_dir():
        raise FileNotFoundError("pinned base snapshot is not cached")
    checkpoint = checkpoint_audit(Path(PARENT_CHECKPOINT), plan["parent_config"], parent=True)
    inventory = load_token_inventory(config["token_inventory"])
    tokenizer = AutoTokenizer.from_pretrained(PARENT_CHECKPOINT, local_files_only=True)
    base = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    base.add_tokens([AddedToken(t, normalized=False, special=False) for t in inventory["atlas_tokens"]])
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in inventory["atlas_tokens"]]
    expected = checkpoint["semantic_tokens"]
    if (expected["tokens"] != inventory["atlas_tokens"] or ids != [[i] for i in expected["token_ids"]]
            or [base.encode(t, add_special_tokens=False) for t in inventory["atlas_tokens"]] != ids):
        raise ValueError("parent/base tokenizer atlas IDs differ")
    adapter = read_json(Path(PARENT_CHECKPOINT) / "adapter_config.json")
    if len(adapter.get("trainable_token_indices", {})) != 2 or any(v != expected["token_ids"] for v in adapter["trainable_token_indices"].values()):
        raise ValueError("adapter must retain both input and output atlas row IDs")
    train_rows = [r for _, r in iter_jsonl(Path(config["train_jsonl"]))]
    mixture = validate_mixture(train_rows)
    if mixture != plan["mixture"]:
        raise ValueError("training step ownership changed")
    train_identity = dataset_identity(config["train_jsonl"], config["image_root"])
    check_identity(train_identity, plan["train_identity"])
    report = {"status": "completed", "checkpoint": checkpoint, "mixture": mixture,
              "train_identity": train_identity, "train_tokens": completion_audit(train_rows, tokenizer),
              "panel_tokens": {}, "panels": {}, "baselines": {}}
    for label, panel in plan["panels"].items():
        identity = dataset_identity(panel["eval_jsonl"], panel["image_root"])
        check_identity(identity, panel["identity"])
        rows = [r for _, r in iter_jsonl(Path(panel["eval_jsonl"]))]
        report["panel_tokens"][label] = completion_audit(rows, tokenizer, budget=panel["max_new_tokens"])
        report["panels"][label] = identity
    report["teacher_eval_identity"] = dataset_identity(config["eval_jsonl"], config["eval_image_root"])
    check_identity(report["teacher_eval_identity"], plan["panels"]["fullboard"]["identity"])
    if sha256_file(PROJECT_ROOT / "sft/modal_full_board_pilot.py") != plan["legacy_pilot_sha256"]:
        raise ValueError("legacy fullboard FP32 restore/autocast recipe changed; review baseline before launch")
    for label, saved in plan["saved_baselines"].items():
        report["baselines"][label] = matched_baseline(label, saved, plan["panels"][label], checkpoint)
        report["baselines"][label]["scorer_sha256"] = digest(plan["source_sha256"])
        report["baselines"][label]["conditions"] = evaluation_conditions(label)
    sft_runs.commit()
    return report


def panel_args(plan: dict, checkpoint: str, label: str) -> argparse.Namespace:
    panel = plan["panels"][label]
    _, batch, budget = PANEL_BUDGETS[label]
    if (panel["batch_size"], panel["max_new_tokens"]) != (batch, budget):
        raise ValueError("unapproved evaluation budget")
    return argparse.Namespace(image_root=panel["image_root"], limit=None, batch_size=batch,
                              long_batch_size=batch, max_new_tokens=budget, long_max_new_tokens=budget,
                              candidate_scoring=False, preserve_visual_fp32=True, occlusion_margin=0.03,
                              model_id=SNAPSHOT, adapter_dir=checkpoint, bits=16,
                              token_inventory=plan["config"]["token_inventory"],
                              disable_flash_attn2=True, enable_thinking=False, do_sample=False)


def evaluation_conditions(label: str) -> dict:
    _, batch, budget = PANEL_BUDGETS[label]
    return dict(image_variant="original", enable_thinking=False, do_sample=False,
                bits=16, candidate_scoring=False, preserve_visual_fp32=True,
                batch_size=batch, long_batch_size=batch,
                max_new_tokens=budget, long_max_new_tokens=budget)


@app.function(**CONTINUATION_GPU_OPTIONS)
def train_bounded(plan: dict, audit: dict) -> dict:
    reload_volumes()
    verify_runtime(plan)
    config = plan["config"]
    check_identity(dataset_identity(config["train_jsonl"], config["image_root"]), audit["train_identity"])
    if sha256_file(Path(config["train_jsonl"])) != audit["train_identity"]["sha256"]:
        raise ValueError("uploaded train JSONL bytes changed after CPU preflight")
    if dataset_identity(config["eval_jsonl"], config["eval_image_root"]) != audit["teacher_eval_identity"]:
        raise ValueError("teacher-forced evaluation changed after CPU preflight")
    if checkpoint_audit(Path(PARENT_CHECKPOINT), plan["parent_config"], parent=True) != audit["checkpoint"]:
        raise ValueError("parent checkpoint changed after CPU preflight")
    # Run the existing worker IN this bounded container, not as an unobserved child.
    try:
        return pilot_train.local(config)
    finally:
        sft_runs.commit()


@app.function(**CONTINUATION_GPU_OPTIONS)
def evaluate_bounded(plan: dict, stage: str, audit: dict) -> dict:
    if stage not in ("pre", "post"):
        raise ValueError("only one pre and one post evaluation are permitted")
    reload_volumes()
    verify_runtime(plan)
    checkpoint = PARENT_CHECKPOINT if stage == "pre" else str(Path(plan["config"]["output_dir"]) / "checkpoints/checkpoint-128")
    checkpoint_report = checkpoint_audit(Path(checkpoint), plan["parent_config"] if stage == "pre" else plan["config"], parent=stage == "pre")
    if stage == "pre" and checkpoint_report != audit["checkpoint"]:
        raise ValueError("parent changed after preflight")
    output = RUN_ROOT / "pipelines" / plan["run_name"] / stage
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "checkpoint": checkpoint, "checkpoint_audit": checkpoint_report,
              "started_at": now(), "panels": {}, "source_sha256": plan["source_sha256"]}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        torch.set_num_threads(16)
        model, processor, evidence = evaluator.load_model(
            model_id=SNAPSHOT, adapter_dir=checkpoint, bits=16, disable_flash_attn2=True,
            token_inventory=plan["config"]["token_inventory"], preserve_visual_fp32=True)
        for label in (NEW_LABELS if stage == "pre" else PANEL_BUDGETS):
            panel = plan["panels"][label]
            identity = dataset_identity(panel["eval_jsonl"], panel["image_root"])
            check_identity(identity, audit["panels"][label])
            if identity["sha256"] != audit["panels"][label]["sha256"]:
                raise ValueError("uploaded eval bytes changed after preflight")
            summary = evaluator.run_eval_job(model=model, processor=processor, adapter_evidence=evidence,
                args=panel_args(plan, checkpoint, label), eval_jsonl=panel["eval_jsonl"],
                image_variant="original", output_dir=output / label)
            if summary["rows"] != PANEL_BUDGETS[label][0] or summary["attempted"] != summary["rows"]:
                raise ValueError("evaluation returned incomplete panel")
            records_path = output / label / "records.jsonl"
            result["panels"][label] = {"summary": summary, "identity": identity,
                                      "records_path": str(records_path), "records_sha256": sha256_file(records_path),
                                      "scorer_sha256": digest(plan["source_sha256"]),
                                      "conditions": evaluation_conditions(label)}
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


def compare_panels(before: dict, after: dict) -> dict:
    if set(before) != set(PANEL_BUDGETS) or set(after) != set(before):
        raise ValueError("comparison requires all six matched panels")
    comparison = {}
    for label in PANEL_BUDGETS:
        check_identity(after[label]["identity"], before[label]["identity"])
        if (not before[label].get("scorer_sha256")
                or before[label]["scorer_sha256"] != after[label].get("scorer_sha256")
                or before[label].get("conditions") != evaluation_conditions(label)
                or after[label].get("conditions") != evaluation_conditions(label)):
            raise ValueError("pre/post scorer version or generation conditions differ")
        a, b = before[label]["summary"], after[label]["summary"]
        if a["rows"] != b["rows"] or b["rows"] != PANEL_BUDGETS[label][0]:
            raise ValueError("pre/post panel counts differ")
        comparison[label] = {"rows": b["rows"], "correct_before": a["correct"], "correct_after": b["correct"],
                             "accuracy_before": a["exact_accuracy"], "accuracy_after": b["exact_accuracy"]}
    a, b = before["fullboard"]["summary"]["full_board"], after["fullboard"]["summary"]["full_board"]
    comparison["fullboard"].update({f"{metric}_{when}": result[metric]
        for when, result in (("before", a), ("after", b))
        for metric in ("board_exact", "occupied_layout_macro_accuracy", "by_layout")})
    return comparison


@app.function(**CPU_OPTIONS, cpu=(0.25, 0.25), memory=(2048, 2048), timeout=300)
def reserve(plan: dict) -> dict:
    sft_runs.reload()
    validate_config(plan["config"], plan["run_name"])
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
    launch = read_json(directory / "launch.json")
    if launch["reservation_id"] != plan["reservation_id"]:
        raise ValueError("remote launch reservation differs")
    destination = directory / "result.json"
    if destination.exists() or Path(plan["config"]["output_dir"]).exists():
        raise FileExistsError(destination)
    result = {"status": "running", "phase": "preflight", "started_at": now(), "stages": {},
              "config": plan["config"], "coordinator_call_id": modal.current_function_call_id()}
    # Persist the full transported plan and coordinator ID before ANY GPU spawn.
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
        entry.update(call_id=active.object_id, status="running", wait_timeout_seconds=timeout)
        persist()
        value = active.get(timeout=timeout)
        if value.get("status") != "completed":
            raise RuntimeError(f"{name} did not complete: {value.get('status')}")
        entry.update(status="completed", ended_at=now(), result=value)
        persist()
        active = None
        return value

    persist()
    try:
        verify_runtime(plan)
        audit = stage("preflight", continuation_preflight, (plan,), PREFLIGHT_TIMEOUT + STARTUP_TIMEOUT + WAIT_GRACE)
        pre = stage("pre", evaluate_bounded, (plan, "pre", audit), GPU_TIMEOUT + STARTUP_TIMEOUT + WAIT_GRACE)
        stage("training", train_bounded, (plan, audit), GPU_TIMEOUT + STARTUP_TIMEOUT + WAIT_GRACE)
        post = stage("post", evaluate_bounded, (plan, "post", audit), GPU_TIMEOUT + STARTUP_TIMEOUT + WAIT_GRACE)
        result["comparison"] = compare_panels({**audit["baselines"], **pre["panels"]}, post["panels"])
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


def launch(*, inputs: str = str(DEFAULT_INPUTS), run_name: str = "spatial-continuation-20260908-r01", execute: bool = False) -> dict:
    plan = build_plan(Path(inputs).resolve(), run_name)
    if not execute:
        return {"dry_run": True, "remote_calls": False, "plan": plan,
                "deferred_cpu_checks": ["checkpoint files/hash/profile/token IDs", "actual completion-token lengths",
                                        "remote baseline pixels/records and current-scorer equivalence"]}
    verify_runtime(plan)
    directory = LOCAL_RUN_ROOT / run_name
    directory.mkdir(parents=True, exist_ok=False)
    plan.update(reservation_id=uuid.uuid4().hex, created_at=now(), status="reserved")
    receipt = directory / "launch.json"
    write_json_atomic(receipt, plan)
    call = None
    try:
        with app.run(detach=True):
            reserve.remote(plan)
            config = plan["config"]
            paths = upload_training_bundle(Path(config["train_jsonl"]), Path(config["image_root"]), Path(config["token_inventory"]),
                remote_dir=f"catan-vision-sft/datasets/{run_name}/training", require_curriculum=False,
                eval_jsonl=Path(config["eval_jsonl"]), eval_image_root=Path(config["eval_image_root"]))
            for label, panel in plan["panels"].items():
                remote, _ = upload_eval_jsonl(Path(panel["eval_jsonl"]), f"catan-vision-sft/datasets/{run_name}/panels/{label}",
                    eval_set_id=f"{run_name}-{label}", image_root=Path(panel["image_root"]))
                panel.update(eval_jsonl=remote, image_root=None)
            plan["config"] = asdict(replace(TrainConfig(**config), train_jsonl=paths[0], eval_jsonl=paths[1],
                image_root=paths[2], eval_image_root=paths[2], token_inventory=paths[3]))
            plan.update(status="ready", uploaded_at=now(), config_sha256=digest(plan["config"]))
            write_json_atomic(receipt, plan)
            call = coordinate.spawn(plan)
            plan.update(status="spawned", coordinator_call_id=call.object_id, spawned_at=now())
            write_json_atomic(receipt, plan)
    except BaseException as exc:
        # Once detached, do not kill its CPU watchdog because a local receipt
        # write failed. Its remote launch receipt already owns the bounded children.
        plan.update(status="launch_failed", error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(receipt, plan)
        raise
    return {"receipt": str(receipt), "coordinator_call_id": call.object_id,
            "remote_result": str(RUN_ROOT / "pipelines" / run_name / "result.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=str(DEFAULT_INPUTS), help="absolute-path dataset_inputs.json contract")
    parser.add_argument("--run-name", default="spatial-continuation-20260908-r01")
    parser.add_argument("--execute", action="store_true", help="OPT IN to uploads and the detached bounded Modal pipeline")
    args = parser.parse_args()
    print(json.dumps(launch(inputs=args.inputs, run_name=args.run_name, execute=args.execute), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
