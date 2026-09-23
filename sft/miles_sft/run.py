"""Plan or execute a bounded, audited Miles/Megatron SFT smoke."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from sft.json_types import JsonLikeDict
from sft.paths import PROJECT_ROOT

from .config import TrainPlan, miles_arguments
from .preflight import inspect_plan, runtime_identity, verify_import_location
from .runtime.receipts import finalize_run


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def execute(plan: TrainPlan, miles_root: Path, megatron_root: Path, bridge_root: Path,
            *, dry_run: bool = True,
            expected_preflight: Mapping[str, object] | None = None) -> JsonLikeDict:
    command = [sys.executable, "-m", "sft.miles_sft.worker", *miles_arguments(plan)]
    if dry_run:
        return {"status": "plan_only", "command": command, "shell_command": shlex.join(command),
                "timeout_seconds": plan.timeout_seconds, "gpu_count": 1,
                "initialization": "merged checkpoint + fresh rank16/alpha32 language LoRA",
                "token_rows": "frozen in merged base", "optimizer_steps": plan.steps}
    started = time.monotonic()
    preflight = inspect_plan(plan)
    if expected_preflight is not None and preflight != expected_preflight:
        raise ValueError("training inputs changed since CPU preflight")
    runtime = runtime_identity(miles_root, megatron_root, bridge_root)
    verify_import_location("miles", miles_root)
    verify_import_location("megatron.bridge", bridge_root)
    output = plan.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipts = output / "receipts"
    receipts.mkdir()
    environment = {
        **os.environ, "CATAN_MILES_ROOT": str(miles_root.resolve()),
        "MILES_SFT_RECEIPT_DIR": str(receipts), "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": os.pathsep.join((str(PROJECT_ROOT), str(miles_root.resolve()),
                                     str(bridge_root.resolve() / "src"), str(megatron_root.resolve()),
                                     os.environ.get("PYTHONPATH", ""))),
    }
    report: JsonLikeDict = {"status": "running", "command": command, "preflight": preflight,
                           "runtime": runtime, "gpu_count": 1, "timeout_seconds": plan.timeout_seconds}
    write_json(output / "launch.json", report)
    try:
        remaining = plan.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("preflight consumed the smoke deadline")
        with (output / "worker.log").open("wb") as log:
            with subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    code = process.wait(timeout=remaining)
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    raise
        report.update(status="completed", result=finalize_run(receipts))
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000])
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "launch.json", report)
    return report
