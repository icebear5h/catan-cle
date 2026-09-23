"""One opt-in, bounded spatial continuation. Importing/dry-running never contacts Modal.

Run ``python -m sft.launchers.spatial.modal_spatial_continuation --help`` locally. Only --execute
reserves remote paths, uploads inputs, and spawns the detached CPU coordinator.
Saved original-image generations are independently rescored before training;
there are no new blank controls, probes, sweeps, or conditional training stages.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from sft.json_types import JsonDict, as_dict
from sft.launchers._json import at_dict, at_str
from sft.launchers._train_config import train_config
from sft.launchers.full_board.modal_full_board_pilot import (
    app,
)
from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.scripts.train.train_trl_catan_vision import (
    write_json_atomic,
)

from ._base import DEFAULT_INPUTS
from ._plan import digest, now
from ._runtime import verify_runtime


def launch(*, inputs: str = str(DEFAULT_INPUTS), run_name: str = "spatial-continuation-20260908-r01", execute: bool = False) -> JsonDict:
    plan = launcher.build_plan(Path(inputs).resolve(), run_name)
    if not execute:
        return {"dry_run": True, "remote_calls": False, "plan": plan,
                "deferred_cpu_checks": ["checkpoint files/hash/profile/token IDs", "actual completion-token lengths",
                                        "remote baseline pixels/records and current-scorer equivalence"]}
    verify_runtime(plan)
    directory = launcher.LOCAL_RUN_ROOT / run_name
    directory.mkdir(parents=True, exist_ok=False)
    plan.update(reservation_id=uuid.uuid4().hex, created_at=now(), status="reserved")
    receipt = directory / "launch.json"
    write_json_atomic(receipt, plan)
    call = None
    try:
        with app.run(detach=True):
            launcher.reserve.remote(plan)
            config = at_dict(plan, "config")
            paths = launcher.upload_training_bundle(Path(at_str(config, "train_jsonl")), Path(at_str(config, "image_root")), Path(at_str(config, "token_inventory")),
                remote_dir=f"catan-vision-sft/datasets/{run_name}/training", require_curriculum=False,
                eval_jsonl=Path(at_str(config, "eval_jsonl")), eval_image_root=Path(at_str(config, "eval_image_root")))
            for label, value in at_dict(plan, "panels").items():
                panel = as_dict(value)
                remote, _ = launcher.upload_eval_jsonl(Path(at_str(panel, "eval_jsonl")), f"catan-vision-sft/datasets/{run_name}/panels/{label}",
                    eval_set_id=f"{run_name}-{label}", image_root=Path(at_str(panel, "image_root")))
                panel.update(eval_jsonl=remote, image_root=None)
            plan["config"] = asdict(replace(train_config(config), train_jsonl=paths[0], eval_jsonl=paths[1],
                image_root=paths[2], eval_image_root=paths[2], token_inventory=paths[3]))
            plan.update(status="ready", uploaded_at=now(), config_sha256=digest(plan["config"]))
            write_json_atomic(receipt, plan)
            call = launcher.coordinate.spawn(plan)
            plan.update(status="spawned", coordinator_call_id=call.object_id, spawned_at=now())
            write_json_atomic(receipt, plan)
    except BaseException as exc:
        # Once detached, do not kill its CPU watchdog because a local receipt
        # write failed. Its remote launch receipt already owns the bounded children.
        plan.update(status="launch_failed", error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(receipt, plan)
        raise
    if call is None:
        raise RuntimeError("coordinator was not spawned")
    return {"receipt": str(receipt), "coordinator_call_id": call.object_id,
            "remote_result": str(launcher.RUN_ROOT / "pipelines" / run_name / "result.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=str(DEFAULT_INPUTS), help="absolute-path dataset_inputs.json contract")
    parser.add_argument("--run-name", default="spatial-continuation-20260908-r01")
    parser.add_argument("--execute", action="store_true", help="OPT IN to uploads and the detached bounded Modal pipeline")
    args = parser.parse_args()
    print(json.dumps(launch(inputs=args.inputs, run_name=args.run_name, execute=args.execute), indent=2, sort_keys=True))
