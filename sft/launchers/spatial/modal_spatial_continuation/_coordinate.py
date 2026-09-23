"""Path reservation and the detached CPU coordinator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import modal

from sft.json_types import JsonDict, as_dict
from sft.launchers._json import at, at_dict, at_str
from sft.launchers.full_board.modal_full_board_pilot import (
    app,
)
from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.scripts.train.train_trl_catan_vision import (
    write_json_atomic,
)

from ._base import (
    COORDINATOR_TIMEOUT,
    CPU_OPTIONS,
    GPU_TIMEOUT,
    PREFLIGHT_TIMEOUT,
    STARTUP_TIMEOUT,
    WAIT_GRACE,
)
from ._plan import now, read_json, validate_config
from ._runtime import verify_runtime
from ._workers import compare_panels


@app.function(**CPU_OPTIONS, cpu=(0.25, 0.25), memory=(2048, 2048), timeout=300)
def reserve(plan: JsonDict) -> JsonDict:
    launcher.sft_runs.reload()
    validate_config(at_dict(plan, "config"), at_str(plan, "run_name"))
    if Path(at_str(plan, "config", "output_dir")).exists():
        raise FileExistsError(at(plan, "config", "output_dir"))
    directory = launcher.RUN_ROOT / "pipelines" / at_str(plan, "run_name")
    directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(directory / "launch.json", plan)
    launcher.sft_runs.commit()
    return {"status": "reserved", "reservation_id": plan["reservation_id"]}


@app.function(**CPU_OPTIONS, cpu=(1.0, 1.0), memory=(4096, 4096), timeout=COORDINATOR_TIMEOUT)
def coordinate(plan: JsonDict) -> JsonDict:
    launcher.sft_runs.reload()
    directory = launcher.RUN_ROOT / "pipelines" / at_str(plan, "run_name")
    launch = read_json(directory / "launch.json")
    if launch["reservation_id"] != plan["reservation_id"]:
        raise ValueError("remote launch reservation differs")
    destination = directory / "result.json"
    if destination.exists() or Path(at_str(plan, "config", "output_dir")).exists():
        raise FileExistsError(destination)
    stages: JsonDict = {}
    result: JsonDict = {"status": "running", "phase": "preflight", "started_at": now(), "stages": stages,
              "config": plan["config"], "coordinator_call_id": modal.current_function_call_id()}
    # Persist the full transported plan and coordinator ID before ANY GPU spawn.
    write_json_atomic(directory / "launch.json", {**plan, "coordinator_call_id": result["coordinator_call_id"]})
    launcher.sft_runs.commit()
    active: modal.FunctionCall[JsonDict] | None = None

    def persist() -> None:
        launcher.sft_runs.reload()
        write_json_atomic(destination, result)
        launcher.sft_runs.commit()

    def stage(
        name: str,
        spawn: Callable[[], modal.FunctionCall[JsonDict]],
        timeout: int,
    ) -> JsonDict:
        nonlocal active
        result["phase"] = name
        entry: JsonDict = {"status": "starting", "started_at": now()}
        stages[name] = entry
        persist()
        active = spawn()
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
        audit = stage("preflight", lambda: launcher.continuation_preflight.spawn(plan), PREFLIGHT_TIMEOUT + STARTUP_TIMEOUT + WAIT_GRACE)
        pre = stage("pre", lambda: launcher.evaluate_bounded.spawn(plan, "pre", audit), GPU_TIMEOUT + STARTUP_TIMEOUT + WAIT_GRACE)
        stage("training", lambda: launcher.train_bounded.spawn(plan, audit), GPU_TIMEOUT + STARTUP_TIMEOUT + WAIT_GRACE)
        post = stage("post", lambda: launcher.evaluate_bounded.spawn(plan, "post", audit), GPU_TIMEOUT + STARTUP_TIMEOUT + WAIT_GRACE)
        result["comparison"] = compare_panels({**at_dict(audit, "baselines"), **at_dict(pre, "panels")}, at_dict(post, "panels"))
        result.update(status="completed", phase="completed")
    except BaseException as exc:
        if active is not None:
            try:
                active.cancel(terminate_containers=True)
                result["cancelled_call_id"] = active.object_id
            except BaseException as cancellation:
                result["cancellation_error"] = repr(cancellation)
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        phase = at_str(result, "phase")
        if phase in stages:
            as_dict(stages[phase]).update(status="failed", ended_at=now(), error=result["error"])
        raise
    finally:
        result["ended_at"] = now()
        persist()
    return result
