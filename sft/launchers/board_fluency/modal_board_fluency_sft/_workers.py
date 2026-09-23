from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import modal
import torch

from sft.json_types import JsonDict, JsonLike, JsonLikeDict, JsonValue, as_dict, as_str
from sft.launchers._train_config import train_config
from sft.launchers.board_fluency import modal_board_fluency_eval as shared
from sft.launchers.board_fluency.modal_board_fluency_eval import reload_volumes
from sft.launchers.modal_catan_vision_sft import sft_runs
from sft.scripts.train import train_trl_catan_vision as trainer
from sft.scripts.train.train_trl_catan_vision import write_json_atomic

from ._callbacks import CheckpointCallback
from ._config import COMMON, CONTROL_CPU, GPU, OFFLINE, PARENT, STAGE_SECONDS, app
from ._data import progress
from ._gate import gate_work, prepared_for
from ._planning import (
    check_deadline,
    check_manifest,
    deadline_alarm,
    prepare_work,
    verify_plan,
    verify_uploaded,
)
from ._probes import eval_panel, load_eval, visual_file_digest


def train_work(plan: JsonDict, deadline: float) -> JsonDict:
    prepared = prepared_for(plan)
    gate = shared.read_json(Path(as_str(plan["root"])) / "gate/result.json")
    if gate["status"] != "completed" or gate["launch_sha256"] != shared.digest(plan):
        raise ValueError("real GPU gate must complete before main training")
    config = train_config(as_dict(plan["config"]))
    if Path(config.output_dir).exists():
        raise FileExistsError("training output exists; resumption/retry is not authorized")
    progress("training: fresh optimizer from immutable expanded-r16, 128 steps / effective batch8")
    callback = CheckpointCallback(config, deadline)
    result = trainer.run_training(config, extra_callbacks=[callback])
    if result["status"] != "completed" or callback.saved_steps != [32, 64, 96, 128]:
        raise RuntimeError("main training did not commit exactly checkpoints 32/64/96/128")
    final = Path(config.output_dir) / "checkpoints/checkpoint-128"
    frozen_visual = visual_file_digest(Path(PARENT) / trainer.VISUAL_STATE_FILE)
    if visual_file_digest(final / trainer.VISUAL_STATE_FILE) != frozen_visual:
        raise ValueError("main training changed frozen visual state")
    if config.initial_bundle is None:
        raise ValueError("main training requires the expanded initial bundle")
    check_manifest(Path(config.initial_bundle), as_dict(prepared["expanded_files"]))
    return {"training": result, "final_checkpoint": str(final),
            "committed_steps": list[JsonValue](callback.saved_steps),
            "frozen_visual_file_tensor_sha256": frozen_visual}


def posteval_work(plan: JsonDict, deadline: float) -> JsonDict:
    prepared_for(plan)
    root = Path(as_str(plan["root"]))
    training = shared.read_json(root / "train/result.json")
    if training["status"] != "completed" or training["launch_sha256"] != shared.digest(plan):
        raise ValueError("main training must complete before post-evaluation")
    checkpoint = str(Path(as_str(as_dict(plan["config"])["output_dir"])) / "checkpoints/checkpoint-128")
    if as_dict(training["result"])["final_checkpoint"] != checkpoint:
        raise ValueError("post-evaluation checkpoint differs")
    model, tokenizer, evidence = load_eval(plan, checkpoint)
    panels: JsonDict = {}
    for panel in ("review", "validation_eval"):
        check_deadline(deadline)
        panels[panel] = eval_panel(plan, model, tokenizer, evidence, checkpoint, panel,
                                   root / "posteval" / panel)
    return {"checkpoint": checkpoint, "panels": panels}


def worker(plan: JsonDict, stage: str, deadline: float) -> None:
    os.environ.update(OFFLINE)
    torch.set_num_threads(4 if stage == "prepare" else 16)
    directory = Path(as_str(plan["root"])) / stage
    receipt: JsonLikeDict = {"status": "running", "stage": stage, "launch_sha256": shared.digest(plan),
                             "started_at": shared.now(), "inner_deadline_unix": deadline}
    write_json_atomic(directory / "result.json", receipt)
    sft_runs.commit()
    try:
        with deadline_alarm(deadline):
            verify_plan(plan)
            receipt["dependencies"] = trainer.assert_runtime_versions()
            if stage != "prepare":
                verify_uploaded(plan)
            stages: dict[str, Callable[[JsonDict, float], Mapping[str, JsonLike]]] = {
                "prepare": prepare_work, "gate": gate_work, "train": train_work, "posteval": posteval_work}
            function = stages[stage]
            receipt.update(result=function(plan, deadline), status="completed")
    except BaseException as exc:
        receipt.update(status="failed", error=shared.error_record(exc))
        progress(f"{stage} failed: {receipt['error']}")
        raise
    finally:
        receipt["ended_at"] = shared.now()
        write_json_atomic(directory / "result.json", receipt)
        sft_runs.commit()


def bounded_stage(plan: JsonDict, stage: str, deadline: float) -> JsonDict:
    """Hard subprocess limit backs up the cooperative alarm/callback deadlines."""
    started = time.time()
    hard_deadline = min(deadline - 30, started + STAGE_SECONDS[stage] - 30)
    inner_deadline = hard_deadline - 45
    check_deadline(inner_deadline)
    reload_volumes()
    directory = Path(as_str(plan["root"])) / stage
    # Delivery/startup failures cannot silently reuse a paid stage's work.
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "sft.launchers.board_fluency.modal_board_fluency_sft", "--worker", stage,
               "--plan", as_str(plan["root"]) + "/launch.json", "--deadline", str(inner_deadline)]
    write_json_atomic(directory / "wrapper.json", {"status": "running", "command": command,
                      "call_id": modal.current_function_call_id(), "hard_deadline_unix": hard_deadline})
    sft_runs.commit()
    process = None
    relay = None
    error = None
    # An open file on /runs can prevent Volume.commit's reload. Keep the live
    # tee on the container disk and copy it after the worker closes its handles.
    log_path = Path("/tmp") / f"board-fluency-{plan['reservation_id']}-{stage}.log"
    try:
        progress(f"{stage}: starting bounded worker, {int(inner_deadline - time.time())}s inner allowance")
        with log_path.open("x", buffering=1) as log:
            process = subprocess.Popen(command, env={**os.environ, **OFFLINE}, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

            output = process.stdout
            if output is None:
                raise RuntimeError(f"{stage} worker has no stdout pipe")

            def forward_logs() -> None:
                for line in output:
                    log.write(line)
                    print(line, end="", flush=True)

            relay = threading.Thread(target=forward_logs, daemon=True)
            relay.start()
            try:
                code = process.wait(timeout=max(0.001, hard_deadline - time.time()))
                if code:
                    raise RuntimeError(f"{stage} worker exited {code}; see {directory}/worker.log")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                relay.join(timeout=5)
        result = shared.read_json(directory / "result.json")
        if result["status"] != "completed":
            raise RuntimeError(f"{stage} failed: {result.get('error')}")
        return result
    except BaseException as exc:
        error = shared.error_record(exc)
        raise
    finally:
        if log_path.is_file():
            shutil.copyfile(log_path, directory / "worker.log")
        write_json_atomic(directory / "wrapper.json", {"status": "failed" if error else "completed",
                          "error": error, "call_id": modal.current_function_call_id(),
                          "hard_deadline_unix": hard_deadline, "ended_at": shared.now()})
        sft_runs.commit()


@app.function(**COMMON, cpu=(4.0, 4.0), memory=(16 * 1024, 16 * 1024), timeout=1200)
def prepare_cpu(plan: JsonDict, deadline: float) -> JsonDict:
    return bounded_stage(plan, "prepare", deadline)


@app.function(**GPU, timeout=STAGE_SECONDS["gate"])
def gate_h200(plan: JsonDict, deadline: float) -> JsonDict:
    return bounded_stage(plan, "gate", deadline)


@app.function(**GPU, timeout=STAGE_SECONDS["train"])
def train_h200(plan: JsonDict, deadline: float) -> JsonDict:
    return bounded_stage(plan, "train", deadline)


@app.function(**GPU, timeout=1800)
def posteval_h200(plan: JsonDict, deadline: float) -> JsonDict:
    return bounded_stage(plan, "posteval", deadline)


@app.function(**COMMON, cpu=(CONTROL_CPU, CONTROL_CPU), memory=(2048, 2048), timeout=300)
def reserve(plan: JsonDict) -> None:
    sft_runs.reload()
    root = Path(as_str(plan["root"]))
    root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(root / "launch.json", plan)
    write_json_atomic(root / "budget_guard.json", as_dict(plan["budget"]))
    sft_runs.commit()
