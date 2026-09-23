"""Bounded remote training and its teacher-forced history."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from sft.json_types import JsonDict, JsonValue, as_dict, as_float, as_int, as_list
from sft.launchers._json import at_dict, at_str
from sft.launchers._train_config import train_config
from sft.launchers.spatial import modal_spatial_extension as extension
from sft.launchers.spatial.modal_spatial_continuation import (
    PANEL_BUDGETS,
    checkpoint_audit,
    now,
    read_json,
    reload_volumes,
)
from sft.scripts.train.train_trl_catan_vision import (
    sha256_file,
    write_json_atomic,
)

from ._base import CHECKPOINT_STEPS, TRAIN_GPU_OPTIONS, TRAIN_TIMEOUT, app
from ._plan import audit_inputs, verify_runtime


def training_history(output: Path, *, completed: bool) -> JsonDict:
    """Retain periodic teacher-forced metrics from the trainer's real saved states."""
    checkpoints: JsonDict = {}
    history: dict[int, JsonDict] = {}
    for step in CHECKPOINT_STEPS:
        path = output / "checkpoints" / f"checkpoint-{step}"
        state_path = path / "trainer_state.json"
        if not state_path.exists() and not completed:
            continue
        state = read_json(state_path)
        if state["global_step"] != step:
            raise ValueError("saved checkpoint global_step differs")
        for row in map(as_dict, as_list(state["log_history"])):
            if "eval_loss" in row:
                row_step = as_int(row["step"])
                if row_step not in CHECKPOINT_STEPS or row_step > step or not math.isfinite(as_float(row["eval_loss"])):
                    raise ValueError("invalid periodic teacher-forced history")
                if row_step in history and history[row_step] != row:
                    raise ValueError("teacher-forced history differs across checkpoints")
                history[row_step] = row
        if completed:
            for name in ("adapter_config.json", "adapter_model.safetensors", "visual_model.safetensors",
                         "trainable_parameters.json", "training_config.json", "tokenizer_config.json", "tokenizer.json"):
                if not (path / name).is_file():
                    raise FileNotFoundError(path / name)
        checkpoints[str(step)] = {"checkpoint": str(path), "trainer_state_sha256": sha256_file(state_path)}
    if completed and set(history) != set(CHECKPOINT_STEPS):
        raise ValueError("all eight periodic teacher-forced evaluations are required")
    teacher_forced: list[JsonValue] = [history[k] for k in sorted(history)]
    return {"checkpoints": checkpoints, "teacher_forced_history": teacher_forced}


@app.function(**TRAIN_GPU_OPTIONS)
def train_bounded(plan: JsonDict, audit: JsonDict) -> JsonDict:
    reload_volumes()
    verify_runtime(plan)
    if audit["status"] != "completed" or set(at_dict(audit, "baselines")) != set(PANEL_BUDGETS):
        raise ValueError("all six CPU baselines must complete before training")
    if extension.audit_parent(plan) != audit["checkpoint_audit"]:
        raise ValueError("parent changed after CPU preflight")
    for key, identity in audit_inputs(plan).items():
        if identity != audit[key]:
            raise ValueError("inputs changed after CPU preflight")
    config = at_dict(plan, "config")
    output = Path(at_str(config, "output_dir"))
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoints/checkpoint-256"
    result: JsonDict = {"status": "running", "config": config, "started_at": now(),
                        "checkpoint": str(checkpoint), "checkpoint_audit": None, "panels": {},
                        "source_sha256": plan["source_sha256"], "timeout_seconds": TRAIN_TIMEOUT}
    destination = output / "result.json"
    write_json_atomic(destination, result)
    extension.sft_runs.commit()
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        torch.set_num_threads(16)
        result["training"] = extension.run_training(train_config(config))
        result.update(extension.training_history(output, completed=True))
        result["checkpoint_audit"] = checkpoint_audit(checkpoint, config, parent=False, expected_step=256)
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            result.update(extension.training_history(output, completed=False))
        except Exception as history_error:
            result["history_error"] = repr(history_error)
        raise
    finally:
        result["ended_at"] = now()
        write_json_atomic(destination, result)
        extension.sft_runs.commit()
    return result
