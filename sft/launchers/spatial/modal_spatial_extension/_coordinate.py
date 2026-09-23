"""Path reservation and the detached CPU coordinator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import modal

from sft.json_types import JsonDict, as_dict
from sft.launchers._json import at_dict, at_str
from sft.launchers.spatial import modal_spatial_extension as extension
from sft.launchers.spatial.modal_spatial_continuation import (
    CPU_OPTIONS,
    PREFLIGHT_TIMEOUT,
    STARTUP_TIMEOUT,
    WAIT_GRACE,
    compare_panels,
    now,
    read_json,
)
from sft.scripts.train.train_trl_catan_vision import (
    write_json_atomic,
)

from ._base import COORDINATOR_TIMEOUT, POST_TIMEOUT, TRAIN_TIMEOUT, app
from ._plan import verify_runtime


@app.function(**CPU_OPTIONS, cpu=(0.25, 0.25), memory=(2048, 2048), timeout=300)
def reserve(plan: JsonDict) -> JsonDict:
    extension.sft_runs.reload()
    verify_runtime(plan)
    output_dir = at_str(plan, "config", "output_dir")
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    directory = extension.RUN_ROOT / "pipelines" / at_str(plan, "run_name")
    directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(directory / "launch.json", plan)
    extension.sft_runs.commit()
    return {"status": "reserved", "reservation_id": plan["reservation_id"]}


@app.function(**CPU_OPTIONS, cpu=(1.0, 1.0), memory=(4096, 4096), timeout=COORDINATOR_TIMEOUT)
def coordinate(plan: JsonDict) -> JsonDict:
    extension.sft_runs.reload()
    directory = extension.RUN_ROOT / "pipelines" / at_str(plan, "run_name")
    if read_json(directory / "launch.json") != plan:
        raise ValueError("remote launch reservation differs")
    destination = directory / "result.json"
    if destination.exists() or Path(at_str(plan, "config", "output_dir")).exists():
        raise FileExistsError(destination)
    stages: JsonDict = {}
    result: JsonDict = {"status": "running", "phase": "preflight", "started_at": now(), "stages": stages,
                        "config": plan["config"], "source_sha256": plan["source_sha256"],
                        "coordinator_call_id": modal.current_function_call_id()}
    write_json_atomic(directory / "launch.json", {**plan, "coordinator_call_id": result["coordinator_call_id"]})
    extension.sft_runs.commit()
    active: modal.FunctionCall[JsonDict] | None = None

    def persist() -> None:
        extension.sft_runs.reload()
        write_json_atomic(destination, result)
        extension.sft_runs.commit()

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
        audit = stage("preflight", lambda: extension.extension_preflight.spawn(plan), PREFLIGHT_TIMEOUT)
        training = stage("training", lambda: extension.train_bounded.spawn(plan, audit), TRAIN_TIMEOUT)
        post = stage("post", lambda: extension.evaluate_bounded.spawn(plan, audit, training), POST_TIMEOUT)
        result["comparison"] = compare_panels(at_dict(audit, "baselines"), at_dict(post, "panels"))
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
