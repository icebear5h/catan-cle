"""Opt-in 256-additional-update extension of the completed spatial continuation.

Dry run: python -B -m sft.launchers.spatial.modal_spatial_extension --run-name spatial-continuation-20260912-r01
Add --execute to reserve a new run and detach its bounded CPU coordinator.
The unchanged, already-uploaded parent datasets and all six saved post panels
are pinned by local receipts and checked on CPU before the only training call.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from sft.json_types import JsonDict
from sft.launchers.spatial import modal_spatial_extension as extension
from sft.launchers.spatial.modal_spatial_continuation import (
    DEFAULT_INPUTS,
    now,
)
from sft.scripts.train.train_trl_catan_vision import (
    write_json_atomic,
)

from ._base import DEFAULT_RUN_NAME, app
from ._plan import verify_files, verify_runtime


def launch(*, inputs: str = str(DEFAULT_INPUTS), run_name: str = DEFAULT_RUN_NAME, execute: bool = False) -> JsonDict:
    plan = extension.build_plan(Path(inputs).resolve(), run_name)
    verify_runtime(plan)
    if not execute:
        return {"dry_run": True, "remote_calls": False, "plan": plan,
                "deferred_cpu_checks": ["remote receipt/input/checkpoint hashes and FP32/atlas scope",
                                        "tokenizer IDs and completion lengths", "all six saved original response rescores"]}
    verify_files(plan["input_files_sha256"])
    directory = extension.LOCAL_RUN_ROOT / run_name
    directory.mkdir(parents=True, exist_ok=False)
    plan.update(reservation_id=uuid.uuid4().hex, created_at=now(), status="reserved")
    receipt = directory / "launch.json"
    write_json_atomic(receipt, plan)
    try:
        with app.run(detach=True):
            verify_runtime(plan)
            verify_files(plan["input_files_sha256"])
            extension.reserve.remote(plan)
            call = extension.coordinate.spawn(plan)
            plan.update(status="spawned", coordinator_call_id=call.object_id, spawned_at=now())
            write_json_atomic(receipt, plan)
    except BaseException as exc:
        # A detached coordinator owns cancellation even if writing locally fails.
        write_json_atomic(receipt, {**plan, "status": "launch_failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return {"receipt": str(receipt), "coordinator_call_id": call.object_id,
            "remote_result": str(extension.RUN_ROOT / "pipelines" / run_name / "result.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=str(DEFAULT_INPUTS), help="unchanged parent dataset_inputs.json")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--execute", action="store_true", help="OPT IN to the detached bounded Modal pipeline")
    args = parser.parse_args()
    print(json.dumps(extension.launch(inputs=args.inputs, run_name=args.run_name, execute=args.execute), indent=2, sort_keys=True))
