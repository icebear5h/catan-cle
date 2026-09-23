"""The paid H200 boundary: training, its budgeted twin, and the spend guard."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import modal

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_float, as_int, loads_json
from sft.launchers import modal_catan_vision_sft as launcher
from sft.launchers._train_config import train_config
from sft.launchers.catan_vision._budget import (
    training_budget_plan,
    wait_for_budgeted_call,
)
from sft.launchers.catan_vision._config import (
    _VOLUMES,
    BUDGET_FUNCTION_OPTIONS,
    BUDGET_GUARD_FILE,
    BUDGET_STARTUP_SECONDS,
    BUDGET_TIMEOUT_SECONDS,
    LAUNCH_MANIFEST,
    REMOTE_RUNS,
    app,
    hf_cache,
    hf_secret,
    sft_runs,
    training_image,
)
from sft.launchers.catan_vision._receipts import (
    _canonical_hash,
    _latest_checkpoint,
    _required_artifacts,
    _utc_now,
    _write_json_atomic,
)
from sft.scripts.train.train_trl_catan_vision import run_training


@app.function(
    image=training_image,
    gpu="H200",
    cpu=16.0,
    memory=128 * 1024,
    secrets=[hf_secret],
    volumes=_VOLUMES,
    timeout=60 * 60 * 24,
)
def train_h200(
    config_payload: JsonDict,
    launch_payload: JsonLikeDict,
    resume_latest: bool = False,
) -> JsonLikeDict:
    """Cross the paid boundary and call the native trainer directly."""

    config = train_config(config_payload)
    output_dir = Path(config.output_dir)
    manifest_path = output_dir / LAUNCH_MANIFEST
    identity = _canonical_hash(launch_payload)
    existing = as_dict(loads_json(manifest_path.read_text())) if manifest_path.is_file() else None
    if existing is not None and existing.get("identity") != identity:
        raise RuntimeError(f"refusing to reuse {output_dir} for a different launch")
    if existing is not None and existing.get("status") == "completed":
        return {
            "status": "already_completed",
            "identity": identity,
            "artifacts": _required_artifacts(output_dir),
        }
    if existing is None and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"non-empty output directory has no {LAUNCH_MANIFEST}: {output_dir}")

    resumed_from_checkpoint = config.resume_from_checkpoint
    if resume_latest:
        if existing is None:
            raise RuntimeError("--resume-latest requires an existing interrupted launch")
        latest = _latest_checkpoint(output_dir / "checkpoints")
        if latest is None:
            raise RuntimeError("--resume-latest found no complete Trainer checkpoint")
        resumed_from_checkpoint = str(latest)
        config = replace(config, resume_from_checkpoint=resumed_from_checkpoint)
    elif existing is not None:
        raise RuntimeError(
            "an interrupted launch already exists; pass --resume-latest to resume it safely"
        )

    manifest: JsonLikeDict = {
        **launch_payload,
        "identity": identity,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "attempt": as_int((existing or {}).get("attempt", 0)) + 1,
        "resumed_from_checkpoint": resumed_from_checkpoint,
    }
    _write_json_atomic(manifest_path, manifest)
    try:
        result = run_training(config)
        artifacts = _required_artifacts(output_dir)
    except Exception as exc:
        manifest.update(status="failed", completed_at=_utc_now(), error=repr(exc))
        _write_json_atomic(manifest_path, manifest)
        raise
    else:
        manifest.update(status="completed", completed_at=_utc_now(), artifacts=artifacts)
        _write_json_atomic(manifest_path, manifest)
        return {**result, "identity": identity, "artifacts": artifacts}
    finally:
        hf_cache.commit()
        sft_runs.commit()


@app.function(
    image=training_image, secrets=[hf_secret], volumes=_VOLUMES,
    **BUDGET_FUNCTION_OPTIONS,
)
def train_h200_budgeted(
    config_payload: JsonDict, launch_payload: JsonLikeDict,
    deadline_unix: float,
) -> JsonLikeDict:
    budget = launch_payload.get("budget")
    budget_usd = as_float((as_dict(budget) if budget else {}).get("budget_usd", 0))
    if budget != training_budget_plan(budget_usd):
        raise ValueError("missing or inconsistent budget plan")
    if budget is None or not math.isfinite(deadline_unix) or time.time() >= deadline_unix:
        raise RuntimeError("missing budget or expired training deadline")
    if deadline_unix - time.time() > BUDGET_TIMEOUT_SECONDS + BUDGET_STARTUP_SECONDS + 5:
        raise ValueError("training deadline exceeds the bounded profile")
    # .local executes the existing trainer body IN THIS bounded container;
    # it does not allocate the legacy 24-hour H200 function.
    return launcher.train_h200.local(config_payload, launch_payload, False)


@app.function(
    image=training_image, volumes={REMOTE_RUNS: sft_runs},
    cpu=(0.25, 0.25), memory=(2048, 2048),
    timeout=BUDGET_TIMEOUT_SECONDS + BUDGET_STARTUP_SECONDS + 120,
    startup_timeout=120, retries=0, max_containers=1, scaledown_window=2,
)
def guard_training_budget(
    function_call_id: str, deadline_unix: float, output_dir: str,
) -> JsonLikeDict:
    """Independent CPU-only watch; retries/preemptions cannot reset the deadline."""
    call = modal.FunctionCall.from_id(function_call_id)
    # Outside the training output directory: a pre-start guard must not trip
    # the existing non-empty-output/launch-manifest collision check.
    path = Path(output_dir).parent / f"{Path(output_dir).name}-{BUDGET_GUARD_FILE}"
    state: JsonLikeDict = {"function_call_id": function_call_id, "deadline_unix": deadline_unix,
             "started_at": _utc_now(), "status": "watching"}
    try:
        _write_json_atomic(path, state)
        sft_runs.commit()
        state.update(wait_for_budgeted_call(call, deadline_unix))
    except BaseException as exc:
        call.cancel(terminate_containers=True)
        state.update(status="failed_closed", error=repr(exc))
        raise
    finally:
        state["ended_at"] = _utc_now()
        _write_json_atomic(path, state)
        sft_runs.commit()
    return state
