"""Opt-in 256-update r02 extension of r01; local dry runs never contact Modal.

MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2     .venv/bin/python -B -m sft.launchers.board_fluency.modal_board_fluency_extension2     --run-name board-fluency-extension-20260915-r02 --budget-usd 21
Add --execute to reserve and supervise one run, or --stop to stop its entire app.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from sft.json_types import JsonLikeDict, as_str
from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_sft import (
    check_deadline,
    deadline_alarm,
    shared,
)
from sft.scripts.train.train_trl_catan_vision import write_json_atomic

from ._baselines import build_plan, verify_plan
from ._config import (
    ABSOLUTE_SECONDS,
    DEFAULT_RUN_NAME,
    LIMITS,
    LOCAL_ROOT,
    STAGE_SECONDS,
    STARTUP,
    app,
)
from ._coordination import coordinate, reserve, stop_run
from ._training import worker


def launch(run_name: str, budget_usd: float = 21, execute: bool = False) -> JsonLikeDict:
    plan = build_plan(run_name, budget_usd)
    verify_plan(plan)
    if not execute:
        return {"dry_run": True, "remote_calls": False, "plan": plan,
                "deferred_to_cpu": "old mounted receipts/data/baselines; current full parent checkpoint manifest; native cached base audit"}
    if os.environ.get("MODAL_PROFILE") != "icebear5h" or original.HF_SECRET_NAME != "huggingface-secret-2":
        raise ValueError("execute requires MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2")
    local = LOCAL_ROOT / run_name
    local.mkdir(parents=True, exist_ok=False)
    write_json_atomic(local / "launch.json", plan)
    state: JsonLikeDict = {"status": "reserving", "launch_sha256": shared.digest(plan)}
    app_id: str | None = None
    write_json_atomic(local / "orchestration.json", state)
    try:
        with app.run(detach=True):
            app_id = app.app_id
            if app_id is None:
                raise RuntimeError("running Modal app reported no app id")
            state["app_id"] = app_id
            write_json_atomic(local / "orchestration.json", state)
            try:
                with deadline_alarm(time.time() + LIMITS["reservation_seconds"] + STARTUP):
                    reservation = reserve.spawn(plan)
                    state["reservation_call_id"] = reservation.object_id
                    write_json_atomic(local / "orchestration.json", state)
                    reservation.get(timeout=LIMITS["reservation_seconds"] + STARTUP)
                absolute_deadline = time.time() + ABSOLUTE_SECONDS
                with deadline_alarm(absolute_deadline):
                    call = coordinate.spawn(plan, absolute_deadline, app_id)
                    state.update(status="spawned", coordinator_call_id=call.object_id,
                                 absolute_deadline_unix=absolute_deadline,
                                 remote_receipt=as_str(plan["root"]) + "/coordinator.json")
                    write_json_atomic(local / "orchestration.json", state)
                    while True:
                        check_deadline(absolute_deadline)
                        try:
                            result = call.get(timeout=min(30, absolute_deadline - time.time()))
                            break
                        except TimeoutError as polling_timeout:
                            if str(polling_timeout):
                                raise
                    if result["status"] != "completed" or result["launch_sha256"] != shared.digest(plan):
                        raise RuntimeError("coordinator did not complete the reserved extension")
                    write_json_atomic(local / "coordinator.json", result)
                    state.update(status="completed", completed_at=shared.now(), comparison=result["comparison"])
            except BaseException as exc:
                state.update(status="launch_failed", error=shared.error_record(exc))
                if isinstance(exc, KeyboardInterrupt):
                    raise RuntimeError("extension interrupted; stopping dedicated app") from exc
                raise
            finally:
                try:
                    original.stop_app(app_id)
                    state.update(whole_app_stop_requested=True, stop_requested_at=shared.now())
                except BaseException as stop_error:
                    state["stop_error"] = shared.error_record(stop_error)
                write_json_atomic(local / "orchestration.json", state)
    except BaseException as exc:
        state.update(status="launch_failed", error=shared.error_record(exc))
        if app_id is not None and not state.get("whole_app_stop_requested"):
            try:
                original.stop_app(app_id)
                state["whole_app_stop_requested"] = True
            except BaseException as stop_error:
                state["stop_error"] = shared.error_record(stop_error)
        write_json_atomic(local / "orchestration.json", state)
        raise
    return {**state, "local_receipt": str(local / "orchestration.json"),
            "stop": f"MODAL_PROFILE=icebear5h {sys.executable} -B -m sft.launchers.board_fluency.modal_board_fluency_extension2 --run-name {run_name} --stop"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--budget-usd", type=float, default=21)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--stop", action="store_true")
    parser.add_argument("--worker", choices=tuple(STAGE_SECONDS), help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if (os.environ.get("MODAL_IS_REMOTE") != "1" or args.plan is None or args.deadline is None
                or args.execute or args.stop):
            parser.error("internal worker requires a remote container, plan and deadline")
        worker(shared.read_json(args.plan), args.worker, args.deadline)
    else:
        result = stop_run(args.run_name) if args.stop else launch(args.run_name, args.budget_usd, args.execute)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
