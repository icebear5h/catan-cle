from __future__ import annotations

import os
import time
from pathlib import Path

import modal

from sft.json_types import JsonDict, as_dict, as_str
from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_sft import (
    check_deadline,
    deadline_alarm,
    progress,
    shared,
)
from sft.scripts.train.train_trl_catan_vision import sha256_file, write_json_atomic

from ._baselines import verify_plan
from ._config import (
    ABSOLUTE_SECONDS,
    BUDGET_CLAIMS,
    CLAIM_KEY,
    COORDINATOR_SECONDS,
    LIMITS,
    LOCAL_ROOT,
    PARENT,
    SCHEMA,
    STAGE_SECONDS,
    STARTUP,
    app,
)
from ._planning import resource_options
from ._training import bounded_stage
from ._types import dig, recorded_app_id


@app.function(**resource_options("prepare"), timeout=STAGE_SECONDS["prepare"])
def prepare_cpu(plan: JsonDict, deadline: float) -> JsonDict:
    return bounded_stage(plan, "prepare", deadline)


@app.function(**resource_options("gpu"), timeout=STAGE_SECONDS["train"])
def train_h200(plan: JsonDict, deadline: float) -> JsonDict:
    return bounded_stage(plan, "train", deadline)


@app.function(**resource_options("gpu"), timeout=STAGE_SECONDS["posteval"])
def posteval_h200(plan: JsonDict, deadline: float) -> JsonDict:
    return bounded_stage(plan, "posteval", deadline)


@app.function(**resource_options("coordinator"), timeout=LIMITS["reservation_seconds"])
def reserve(plan: JsonDict) -> None:
    original.sft_runs.reload()
    verify_plan(plan)
    claims = modal.Dict.from_name(BUDGET_CLAIMS, create_if_missing=True)
    if not claims.put(CLAIM_KEY, {"run_name": plan["run_name"],
                                 "reservation_id": plan["reservation_id"],
                                 "launch_sha256": shared.digest(plan)}, skip_if_exists=True):
        raise RuntimeError("this parent budget lineage is already reserved; reconcile spend before another run")
    root = Path(as_str(plan["root"]))
    root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(root / "launch.json", plan)
    write_json_atomic(root / "budget_guard.json", as_dict(plan["budget"]))
    original.sft_runs.commit()


@app.function(**resource_options("coordinator"), timeout=COORDINATOR_SECONDS)
def coordinate(plan: JsonDict, absolute_deadline: float, app_id: str) -> JsonDict:
    try:
        with deadline_alarm(absolute_deadline - 30):
            return coordinate_work(plan, absolute_deadline, app_id)
    except BaseException:
        try:
            original.stop_app(app_id)
        except BaseException as stop_error:
            progress(f"whole-app stop failed: {shared.error_record(stop_error)}")
        raise


def coordinate_work(plan: JsonDict, absolute_deadline: float, app_id: str) -> JsonDict:
    original.sft_runs.reload()
    root = Path(as_str(plan["root"]))
    if shared.read_json(root / "launch.json") != plan or (root / "coordinator.json").exists():
        raise ValueError("reservation differs or coordinator already used")
    if not 0 < absolute_deadline - time.time() <= ABSOLUTE_SECONDS:
        raise ValueError("invalid absolute coordinator deadline")
    stages: JsonDict = {}
    result: JsonDict = {"status": "running", "launch_sha256": shared.digest(plan), "started_at": shared.now(),
              "stages": stages, "coordinator_call_id": modal.current_function_call_id(), "app_id": app_id,
              "absolute_deadline_unix": absolute_deadline, "budget": plan["budget"]}
    active: modal.FunctionCall[JsonDict] | None = None
    phase: str | None = None

    def persist() -> None:
        original.sft_runs.reload()
        write_json_atomic(root / "coordinator.json", result)
        original.sft_runs.commit()

    try:
        persist()
        verify_plan(plan)
        for phase, function in (("prepare", prepare_cpu), ("train", train_h200), ("posteval", posteval_h200)):
            stage_deadline = min(absolute_deadline - 30, time.time() + STAGE_SECONDS[phase] + STARTUP)
            check_deadline(stage_deadline - 75)
            entry: JsonDict = {"status": "starting", "started_at": shared.now(),
                               "deadline_unix": stage_deadline}
            stages[phase] = entry
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
        before = as_dict(dig(stages, "prepare", "result", "result", "baselines"))
        after = as_dict(dig(stages, "posteval", "result", "result", "panels"))
        if any(dig(before, panel, "rows") != dig(after, panel, "rows") for panel in before):
            raise ValueError("comparison requires complete identical panel coverage")
        comparison: JsonDict = {panel: {
            "before": dig(before, panel, "correct"), "after": dig(after, panel, "correct"),
            "rows": dig(before, panel, "rows"), "baseline_complete": True,
            "before_files": dig(before, panel, "files"), "after_files": dig(after, panel, "files"),
        } for panel in before}
        result.update(status="completed", comparison=comparison, additional_updates=512, cumulative_updates=1024)
    except BaseException as exc:
        result.update(status="failed", error=shared.error_record(exc))
        if phase in stages:
            as_dict(stages[phase]).update(status="failed", ended_at=shared.now(), error=result["error"])
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


def stop_run(run_name: str) -> JsonDict:
    shared.validate_cli("", "", shared.MODEL_ID, shared.MODEL_REVISION, run_name, "", PARENT)
    if os.environ.get("MODAL_PROFILE") != "icebear5h":
        raise ValueError("stop requires MODAL_PROFILE=icebear5h")
    local = LOCAL_ROOT / run_name
    plan = shared.read_json(local / "launch.json")
    if plan["schema"] != SCHEMA or plan["run_name"] != run_name:
        raise ValueError("not this extension-r03's local launch receipt")
    state = shared.read_json(local / "orchestration.json")
    if state["launch_sha256"] != shared.digest(plan):
        raise ValueError("stop receipt launch identity differs")
    original.stop_app(recorded_app_id(state["app_id"]))
    state.update(status="app_stop_requested", stop_requested_at=shared.now())
    write_json_atomic(local / "orchestration.json", state)
    return state
