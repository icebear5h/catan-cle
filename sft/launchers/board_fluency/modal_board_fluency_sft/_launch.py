"""Opt-in, single-run board-fluency continuation; dry runs never contact Modal.

MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2     .venv/bin/python -m sft.launchers.board_fluency.modal_board_fluency_sft --run-name NAME --budget-usd 15
Add --execute only after reviewing the local admission/configuration receipt.
The detached CPU coordinator owns preparation, gate, training and post-evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

from sft.json_types import JsonDict, JsonValue, as_bool, as_dict, as_list, as_str
from sft.launchers.board_fluency import modal_board_fluency_eval as shared
from sft.launchers.modal_catan_vision_sft import HF_SECRET_NAME, sft_data, sft_runs
from sft.paths import PROJECT_ROOT
from sft.scripts.train.train_trl_catan_vision import write_json_atomic

from ._config import (
    ABSOLUTE_SECONDS,
    COMMON,
    CONTROL_CPU,
    COORDINATOR_SECONDS,
    DATA_FILES,
    DATASET,
    REVIEW,
    REVIEW_SHA256,
    STAGE_SECONDS,
    STARTUP,
    app,
)
from ._data import configuration, progress, rows_at
from ._planning import build_plan, check_deadline, deadline_alarm, verify_plan
from ._workers import gate_h200, posteval_h200, prepare_cpu, reserve, train_h200, worker


def _at(value: JsonValue, *keys: str) -> JsonDict:
    """Follow nested JSON object keys, requiring an object at every step."""
    node = as_dict(value)
    for key in keys:
        node = as_dict(node[key])
    return node


@app.function(**COMMON, cpu=(CONTROL_CPU, CONTROL_CPU), memory=(2048, 2048), timeout=COORDINATOR_SECONDS)
def coordinate(plan: JsonDict, absolute_deadline: float) -> JsonDict:
    with deadline_alarm(absolute_deadline - 30):
        return coordinate_work(plan, absolute_deadline)


def coordinate_work(plan: JsonDict, absolute_deadline: float) -> JsonDict:
    sft_runs.reload()
    root = Path(as_str(plan["root"]))
    if shared.read_json(root / "launch.json") != plan or (root / "coordinator.json").exists():
        raise ValueError("reservation differs or coordinator already used")
    if not 0 < absolute_deadline - time.time() <= ABSOLUTE_SECONDS:
        raise ValueError("invalid absolute coordinator deadline")
    stages: JsonDict = {}
    result: JsonDict = {"status": "running", "started_at": shared.now(), "stages": stages,
                        "coordinator_call_id": modal.current_function_call_id(),
                        "absolute_deadline_unix": absolute_deadline, "budget": plan["budget"]}
    active: modal.FunctionCall[JsonDict] | None = None
    phase: str | None = None

    def persist() -> None:
        # A sibling worker commits checkpoints; reload before publishing control
        # receipts so this low-CPU process never obscures a completed save.
        sft_runs.reload()
        write_json_atomic(root / "coordinator.json", result)
        sft_runs.commit()

    persist()
    try:
        verify_plan(plan)
        for phase, function in (("prepare", prepare_cpu), ("gate", gate_h200),
                                ("train", train_h200), ("posteval", posteval_h200)):
            stage_deadline = min(absolute_deadline - 30, time.time() + STAGE_SECONDS[phase] + STARTUP)
            check_deadline(stage_deadline)
            entry: JsonDict = {"status": "starting", "started_at": shared.now(),
                               "deadline_unix": stage_deadline}
            stages[phase] = entry
            persist()
            with deadline_alarm(stage_deadline):
                active = function.spawn(plan, stage_deadline)
                entry.update(status="running", call_id=active.object_id)
                persist()  # The child ID is durable before waiting on ANY stage.
                progress(f"coordinator: {phase} {active.object_id}, deadline={stage_deadline}")
                while True:
                    remaining = min(stage_deadline, absolute_deadline - 30) - time.time()
                    if remaining <= 0:
                        raise TimeoutError(f"{phase} absolute wait deadline; cancelling child including startup loops")
                    try:
                        value = active.get(timeout=min(30, remaining))
                        break
                    except TimeoutError as polling_timeout:
                        # FunctionCall.get's polling timeout is the built-in
                        # exception, distinct from a remote function timeout.
                        if str(polling_timeout):
                            raise
                        progress(f"coordinator: {phase} still running; {int(remaining)}s left")
            if value["status"] != "completed":
                raise RuntimeError(f"{phase} failed: {value.get('error')}")
            entry.update(status="completed", ended_at=shared.now(), result=value)
            persist()
            active = None
            # Bound the preceding container's scaledown before the next GPU.
            if phase in ("gate", "train"):
                time.sleep(2)
        pre = _at(stages, "gate", "result", "result", "pre_validation190")
        post = _at(stages, "posteval", "result", "result", "panels")
        validation_eval = as_dict(post["validation_eval"])
        retained = {as_str(item) for item in as_list(pre["ids"])}
        matched = [row for row in rows_at(Path(as_str(validation_eval["output_dir"])) / "records.jsonl")
                   if row["id"] in retained]
        if len(matched) != pre["rows"]:
            raise ValueError("post-evaluation does not cover the retained baseline IDs")
        result.update(status="completed", comparison={
            "validation190": {"after": validation_eval["correct"], "rows": 190,
                              "baseline_complete": False},
            "validation_matched": {"before": pre["correct"], "after": sum(as_bool(as_dict(row["score"])["correct"]) for row in matched),
                                   "rows": pre["rows"], "baseline_partial": True},
            "unchanged_review200": {"saved_before": 24, "after": as_dict(post["review"])["correct"],
                                    "rows": 200}})
    except BaseException as exc:
        result.update(status="failed", error=shared.error_record(exc))
        if phase is not None and phase in stages:
            as_dict(stages[phase]).update(status="failed", error=result["error"])
        if active is not None:
            try:
                active.cancel(terminate_containers=True)
                result["cancelled_call_id"] = active.object_id
            except BaseException as cancellation:
                result["cancellation_error"] = shared.error_record(cancellation)
        raise
    finally:
        result["ended_at"] = shared.now()
        persist()
    return result


def stop_app(app_id: object) -> None:
    """Stop the whole dedicated app, covering coordinator/child spawn races."""
    if not isinstance(app_id, str) or not app_id.startswith("ap-"):
        raise ValueError("a recorded Modal app ID is required to stop this run")
    subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id],
                   check=True, timeout=60)


def stop_run(run_name: str) -> JsonDict:
    configuration(run_name)  # Validate the exact run-name/path boundary.
    if os.environ.get("MODAL_PROFILE") != "icebear5h":
        raise ValueError("stop requires MODAL_PROFILE=icebear5h")
    path = PROJECT_ROOT / "artifacts/runs/sft" / run_name / "orchestration.json"
    state = shared.read_json(path)
    stop_app(state["app_id"])
    state.update(status="app_stop_requested", stop_requested_at=shared.now())
    write_json_atomic(path, state)
    return state


def launch(run_name: str, budget_usd: float = 15, execute: bool = False) -> JsonDict:
    plan = build_plan(run_name, budget_usd)
    if not execute:
        return {"dry_run": True, "remote_calls": False, "plan": plan,
                "deferred_to_cpu": "cached parent/base, token boundaries, immutable rank expansion",
                "deferred_to_real_gate": "numerical equivalence, actual gradients/updates, reload, validation190"}
    if os.environ.get("MODAL_PROFILE") != "icebear5h" or HF_SECRET_NAME != "huggingface-secret-2":
        raise ValueError("execute requires MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2")
    local = PROJECT_ROOT / "artifacts/runs/sft" / run_name
    local.mkdir(parents=True, exist_ok=False)
    write_json_atomic(local / "launch.json", plan)
    state: JsonDict = {"status": "uploading", "launch_sha256": shared.digest(plan)}
    app_id: str | None = None
    write_json_atomic(local / "orchestration.json", state)
    try:
        with app.run(detach=True):
            app_id = app.app_id
            state["app_id"] = app_id
            write_json_atomic(local / "orchestration.json", state)
            reservation = reserve.spawn(plan)
            try:
                state["reservation_call_id"] = reservation.object_id
                write_json_atomic(local / "orchestration.json", state)
                reservation.get(timeout=600)
            except BaseException:
                reservation.cancel(terminate_containers=True)
                raise
            remote = as_str(plan["data_dir"]).removeprefix("/data")
            input_files = _at(plan, "inputs", "files")
            with sft_data.batch_upload(force=False) as batch:
                for name in DATA_FILES:
                    payload = (DATASET / name).read_bytes()
                    if hashlib.sha256(payload).hexdigest() != as_dict(input_files[name])["sha256"]:
                        raise ValueError("local inputs changed after admission")
                    batch.put_file(io.BytesIO(payload), remote + "/" + name)
                review_bytes = REVIEW.read_bytes()
                if hashlib.sha256(review_bytes).hexdigest() != REVIEW_SHA256:
                    raise ValueError("review changed after admission")
                batch.put_file(io.BytesIO(review_bytes), remote + "/review.jsonl")
                batch.put_file(local / "launch.json", remote + "/launch.json")
            absolute_deadline = time.time() + ABSOLUTE_SECONDS
            call = coordinate.spawn(plan, absolute_deadline)
            state.update(status="spawned", coordinator_call_id=call.object_id,
                         absolute_deadline_unix=absolute_deadline,
                         remote_receipt=as_str(plan["root"]) + "/coordinator.json")
            write_json_atomic(local / "orchestration.json", state)
    except BaseException as exc:
        state.update(status="launch_failed", error=shared.error_record(exc))
        if app_id is not None:
            try:
                stop_app(app_id)
                state["whole_app_stop_requested"] = True
            except BaseException as stop_error:
                state["stop_error"] = shared.error_record(stop_error)
        write_json_atomic(local / "orchestration.json", state)
        raise
    return {**state, "local_receipt": str(local / "orchestration.json"),
             "stop": f"MODAL_PROFILE=icebear5h {sys.executable} -m modal app stop {app_id}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--budget-usd", type=float, default=15)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop", action="store_true", help="Stop the recorded run's entire Modal app")
    parser.add_argument("--worker", choices=tuple(STAGE_SECONDS), help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        # A subprocess has no Modal ContainerIOManager singleton; is_local()
        # therefore returns True even inside the mounted remote container.
        if os.environ.get("MODAL_IS_REMOTE") != "1" or args.plan is None or args.deadline is None:
            parser.error("internal worker requires a remote container, plan and deadline")
        worker(shared.read_json(args.plan), args.worker, args.deadline)
    else:
        if not args.run_name:
            parser.error("--run-name is required")
        if args.stop and args.execute:
            parser.error("--stop and --execute are mutually exclusive")
        result = stop_run(args.run_name) if args.stop else launch(args.run_name, args.budget_usd, args.execute)
        print(json.dumps(result, indent=2, sort_keys=True))
