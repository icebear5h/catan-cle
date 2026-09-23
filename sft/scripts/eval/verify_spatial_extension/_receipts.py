"""receipts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import modal

from sft.json_types import JsonDict, JsonLike, as_dict, as_list, json_path, loads_json
from sft.launchers.spatial.modal_spatial_continuation import (
    PANEL_BUDGETS,
    digest,
    read_json,
)
from sft.launchers.spatial.modal_spatial_extension import (
    CHECKPOINT_STEPS,
    PARENT_CHECKPOINT,
    PARENT_RUN,
    validate_config,
)

from ._base import STATE_PATH


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def local_path(run: Path, relative: str) -> Path:
    """Never follow a receipt/output symlink into a historical run."""
    path = run / relative
    require(path.is_relative_to(run) and ".." not in Path(relative).parts,
            f"path escapes run: {relative}")
    for part in (path, *path.parents):
        if part == run:
            break
        require(not part.is_symlink(), f"symlink receipt path: {part}")
    return path


def atomic_write(run: Path, relative: str, data: bytes) -> None:
    path = local_path(run, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, local_path(run, relative))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def json_bytes(value: JsonLike) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_launch(run_dir: Path) -> tuple[Path, JsonDict]:
    run = run_dir.expanduser().resolve(strict=True)
    launch = read_json(local_path(run, "launch.json"))
    require(launch["schema"] == "catan_spatial_extension_launch/v1"
            and launch["run_name"] == run.name and run.name != PARENT_RUN,
            "not an extension launch directory")
    validate_config(as_dict(launch["config"]), run.name, as_dict(launch["parent_config"]))
    require(launch["config_sha256"] == digest(launch["config"])
            and launch["parent_config_sha256"] == digest(launch["parent_config"])
            and launch["parent_checkpoint"] == PARENT_CHECKPOINT,
            "launch config/parent identity differs")
    require(set(as_dict(launch["panels"])) == set(as_dict(launch["saved_baselines"]))
            == set(PANEL_BUDGETS), "launch requires all six panels")
    require(json_path(launch, "policy", "checkpoint_steps") == list(CHECKPOINT_STEPS),
            "launch checkpoint schedule differs")
    for label, (count, batch, budget) in PANEL_BUDGETS.items():
        panel = as_dict(json_path(launch, "panels", label))
        require((json_path(panel, "identity", "rows"), panel["batch_size"], panel["max_new_tokens"])
                == (count, batch, budget), f"{label}: launch budget differs")
    return run, launch


def expected_checkpoint(launch: JsonDict) -> str:
    config = as_dict(launch["config"])
    return f"{config['output_dir']}/checkpoints/checkpoint-{config['max_steps']}"


def validate_result(result: JsonDict, launch: JsonDict) -> None:
    require(result["config"] == launch["config"]
            and result["source_sha256"] == launch["source_sha256"],
            "coordinator config/source differs from launch")
    require(result["status"] in ("running", "completed", "failed")
            and result["phase"] in ("preflight", "training", "post", "completed")
            and isinstance(result["coordinator_call_id"], str) and result["coordinator_call_id"] != ""
            and set(as_dict(result["stages"])) <= {"preflight", "training", "post"},
            "invalid coordinator status/stages")
    require(all(isinstance(entry, dict)
                and entry.get("status") in ("starting", "running", "completed", "failed")
                for entry in as_dict(result["stages"]).values()), "invalid stage receipt")
    if launch.get("coordinator_call_id"):
        require(result["coordinator_call_id"] == launch["coordinator_call_id"],
                "coordinator call ID differs from launch")


def status_report(result: JsonDict) -> JsonDict:
    report: JsonDict = {key: result[key] for key in ("status", "phase", "coordinator_call_id")}
    stages = as_dict(result["stages"])
    report["stages"] = {
        name: {key: value for key, value in as_dict(entry).items()
               if key in ("status", "call_id", "error")}
        for name, entry in stages.items()
    }
    worker = as_dict(as_dict(stages.get("training", {})).get("result", {}))
    training = as_dict(worker.get("training", {}))
    metrics: JsonDict = {
        group: {key: value for key, value in as_dict(training[group]).items()
                if key in ("epoch", "train_loss", "train_runtime", "eval_loss",
                           "answer_token_accuracy", "answer_row_exact", "eval_mean_token_accuracy")}
        for group in ("metrics", "eval_metrics") if group in training}
    if "teacher_forced_history" in worker:
        metrics["teacher_forced_history"] = [
            {key: value for key, value in as_dict(row).items() if key in ("step", "eval_loss")}
            for row in as_list(worker["teacher_forced_history"])]
    if metrics:
        report["training"] = metrics
    if "error" in result:
        report["error"] = result["error"]
    return report


def completed_post(result: JsonDict, post: JsonDict, launch: JsonDict) -> None:
    require(result["status"] == result["phase"] == "completed"
            and all(json_path(result, "stages", key, "status") == "completed"
                    for key in ("preflight", "training", "post")),
            "pipeline has not completed all three stages")
    require(post == json_path(result, "stages", "post", "result"),
            "post receipt differs from coordinator")
    require(post["status"] == "completed" and set(as_dict(post["panels"])) == set(PANEL_BUDGETS),
            "post requires all six completed panels")
    require(post["checkpoint"] == expected_checkpoint(launch)
            and post["source_sha256"] == launch["source_sha256"],
            "post checkpoint/source differs from launch")


def full_history(history: object) -> bool:
    return (isinstance(history, list) and all(isinstance(row, dict) for row in history)
            and [as_dict(row).get("step") for row in history] == list(CHECKPOINT_STEPS))


def download_receipts(run: Path, launch: JsonDict, *, status_only: bool) -> JsonDict:
    """Only direct serial read_file calls; no listing, remote functions or GPU work."""
    volume = modal.Volume.from_name("catan-sft-runs", create_if_missing=False)
    remote = f"catan-vision-sft/pipelines/{launch['run_name']}"
    first = True

    def read(path: str) -> bytes:
        nonlocal first
        if not first:
            time.sleep(1)
        first = False
        return b"".join(volume.read_file(path))

    data = read(f"{remote}/result.json")
    result = as_dict(loads_json(data))
    validate_result(result, launch)
    atomic_write(run, "result.json", data)
    if status_only or result["status"] != "completed":
        return result
    data = read(f"{remote}/post/result.json")
    post = as_dict(loads_json(data))
    completed_post(result, post, launch)
    atomic_write(run, "post/result.json", data)
    for label in PANEL_BUDGETS:
        saved = as_dict(json_path(post, "panels", label))
        for name in ("summary.json", "records.jsonl"):
            data = read(f"{remote}/post/{label}/{name}")
            if name == "summary.json":
                require(json.loads(data) == saved["summary"], f"{label}: summary receipt differs")
            else:
                require(hashlib.sha256(data).hexdigest() == saved["records_sha256"],
                        f"{label}: records hash differs")
            atomic_write(run, f"post/{label}-{name}", data)
    worker = as_dict(json_path(result, "stages", "training", "result"))
    if not full_history(worker.get("teacher_forced_history", [])):
        data = read(f"catan-vision-sft/{launch['run_name']}/{STATE_PATH}")
        require(hashlib.sha256(data).hexdigest()
                == json_path(post, "checkpoint_audit", "files_sha256", "trainer_state.json"),
                "trainer_state hash differs")
        json.loads(data)
        atomic_write(run, STATE_PATH, data)
    return result
