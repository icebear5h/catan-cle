from __future__ import annotations

import hashlib
import io
import json
import uuid
from pathlib import Path
from types import ModuleType

from sft.board import coordinate_comparison
from sft.json_types import JsonDict, as_str
from sft.launchers.modal_catan_vision_sft import HF_SECRET_NAME, sft_data, sft_runs
from sft.paths import PROJECT_ROOT
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import sha256_file, write_json_atomic

from ._config import (
    BATCH,
    COMPARISON_GPU_DEADLINE,
    CONTEXT,
    DEFAULT_INVENTORY,
    DEFAULT_REVIEW,
    GPU_DEADLINE,
    MODEL_ID,
    MODEL_REVISION,
    NEW_TOKENS,
    ORIGINAL_INVENTORY,
    app,
)
from ._remote import eval_coordinate_h200, eval_h200, prepare_cpu, verify_outputs
from ._validation import (
    digest,
    error_record,
    inspect_inputs,
    json_value,
    now,
    read_json,
    validate_cli,
)


def download_file(remote: str, local: Path) -> bool:
    """Volume.read_file yields byte chunks, not one bytes object."""
    try:
        chunks = iter(sft_runs.read_file(remote))
        first = next(chunks, b"")
    except FileNotFoundError:
        return False
    with local.open("xb") as handle:
        handle.write(first)
        for chunk in chunks:
            handle.write(chunk)
    return True


def module_path(module: ModuleType) -> Path:
    """The source file a hashed module was loaded from."""
    if module.__file__ is None:
        raise ValueError(f"{module.__name__} has no source file to hash")
    return Path(module.__file__)


@app.local_entrypoint()
def main(
    run_name: str,
    hf_repo: str = "",
    hf_revision: str = "",
    adapter_dir: str = "",
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    checkpoint_subdir: str = "",
    eval_jsonl: str = DEFAULT_REVIEW,
    token_inventory: str = "",
    prepare_only: bool = False,
) -> None:
    """Prepare immutable HF caches, then evaluate once; --prepare-only uses CPU only."""
    validate_cli(hf_repo, hf_revision, model_id, model_revision, run_name, checkpoint_subdir, adapter_dir)
    review = Path(eval_jsonl).expanduser().resolve()
    inventory = Path(token_inventory or DEFAULT_INVENTORY).expanduser()
    if not token_inventory and not inventory.is_file():
        inventory = Path(ORIGINAL_INVENTORY)
    inventory = inventory.resolve()
    _, contract = inspect_inputs(review, inventory)
    comparison = contract.get("schema") == coordinate_comparison.SCHEMA
    review_bytes, inventory_bytes = review.read_bytes(), inventory.read_bytes()
    if (hashlib.sha256(review_bytes).hexdigest() != contract["review_sha256"]
            or hashlib.sha256(inventory_bytes).hexdigest() != contract["inventory_sha256"]):
        raise ValueError("local inputs changed during validation")
    local = Path("artifacts/runs/sft") / run_name
    local.mkdir(parents=True, exist_ok=False)
    launch: JsonDict = {
        "schema": "catan_board_fluency_eval_launch/v1", "created_at": now(),
        "run_name": run_name, "hf_repo": hf_repo, "hf_revision": hf_revision,
        "adapter_dir": adapter_dir, "hf_secret_name": HF_SECRET_NAME,
        "checkpoint_subdir": checkpoint_subdir, "model_id": model_id,
        "model_revision": model_revision, "prepare_only": prepare_only, "inputs": json_value(contract),
        "local_review": str(review), "local_inventory": str(inventory),
        "data_dir": f"/data/board-fluency-eval/{run_name}-{uuid.uuid4().hex}",
        "prep_dir": f"/runs/board-fluency-prep/{run_name}",
        "output_dir": f"/runs/board-fluency-eval/{run_name}",
        "source_sha256": {**{name: sha256_file(PROJECT_ROOT / name) for name in (
    "sft/launchers/board_fluency/modal_board_fluency_eval/__init__.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/__main__.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_config.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_validation.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_snapshots.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_remote.py",
    "sft/launchers/board_fluency/modal_board_fluency_eval/_cli.py",
        )},
                          **{module_path(module).name: sha256_file(module_path(module))
                             for module in (evaluator, coordinate_comparison)}},
        "limits": {"gpu": "H200", "cpu": 8, "memory_gib": 64,
                   "execution_seconds": 1800 if comparison else 900,
                   "startup_seconds": 300,
                   "inner_seconds": COMPARISON_GPU_DEADLINE if comparison else GPU_DEADLINE, "retries": 0,
                   "max_containers": 1, "cpu_prepare_seconds": 1200,
                   "context": CONTEXT, "new_tokens": NEW_TOKENS, "batch_size": BATCH},
    }
    write_json_atomic(local / "launch.json", launch)
    state: JsonDict = {"status": "uploading", "launch_sha256": digest(launch), "gpu_calls_requested": 0}
    write_json_atomic(local / "orchestration.json", state)
    try:
        remote = as_str(launch["data_dir"]).removeprefix("/data")
        with sft_data.batch_upload(force=False) as batch:
            batch.put_file(io.BytesIO(review_bytes), remote + "/review.jsonl")
            batch.put_file(io.BytesIO(inventory_bytes), remote + "/trainable_tokens.json")
            batch.put_file(local / "launch.json", remote + "/launch.json")
        state["status"] = "preparing"
        write_json_atomic(local / "orchestration.json", state)
        prepared = prepare_cpu.remote(launch)
        write_json_atomic(local / "prepare.json", prepared)
        if prepared["status"] != "prepared":
            raise RuntimeError(f"CPU preparation failed: {prepared.get('error')}")
        print(json.dumps({"status": "prepared", "checkpoint": prepared["checkpoint_dir"],
                          "base": prepared["base_snapshot"], "max_prompt_tokens": prepared["max_prompt_tokens"],
                          "max_gold_tokens": prepared["max_gold_tokens"], "local_dir": str(local)}, indent=2))
        if prepare_only:
            state["status"] = "prepared_only"
            return
        state.update(status="gpu_requested", gpu_calls_requested=1)
        write_json_atomic(local / "orchestration.json", state)
        worker = eval_coordinate_h200 if comparison else eval_h200
        result = worker.remote(launch, prepared)
        state.update(status=result["status"], result=result)
        if result["status"] != "completed":
            raise RuntimeError(f"GPU eval failed; no retry: {result.get('error')}")
        state["status"] = "downloading"
    except BaseException as exc:
        state.update(status="failed", error=error_record(exc))
        raise
    finally:
        download_errors: JsonDict = {}
        output_dir = as_str(launch["output_dir"])
        targets = {"prepare.json": as_str(launch["prep_dir"]) + "/prepare.json"}
        if state["gpu_calls_requested"]:
            targets.update({name: output_dir + "/" + name
                            for name in ("records.jsonl", "summary.json", "run.json")})
            if state["status"] == "failed":
                targets["evaluator.log"] = output_dir + "/evaluator.log"
        for name, remote_path in targets.items():
            if (local / name).exists():
                continue
            try:
                if not download_file(remote_path.removeprefix("/runs"), local / name):
                    download_errors[name] = "not present on remote volume"
            except Exception as exc:
                download_errors[name] = error_record(exc)
        state.update(ended_at=now(), download_errors=download_errors)
        write_json_atomic(local / "orchestration.json", state)
    try:
        if state["download_errors"]:
            raise RuntimeError(f"artifact download incomplete; do not rerun GPU: {state['download_errors']}")
        verification = verify_outputs(local, launch)
        run = read_json(local / "run.json")
        if (run["status"] != "completed" or run["launch"] != launch
                or run["prepare_sha256"] != digest(prepared) or run["result"] != verification):
            raise ValueError("downloaded run receipt or artifact hashes differ")
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="artifact_verification_failed", error=error_record(exc))
        raise
    finally:
        write_json_atomic(local / "orchestration.json", state)
    print(json.dumps({"status": "completed", "local_dir": str(local), **verification}, indent=2))
