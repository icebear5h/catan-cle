"""CPU merge/preflight and bounded one-H200 Miles SFT in the selected Modal workspace."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import modal

from sft.json_types import JsonDict, as_dict, as_str, load_json_dict
from sft.paths import PROJECT_ROOT

from .config import (
    BASE_REVISION,
    BRIDGE_REVISION,
    IMAGE_REF,
    MEGATRON_REVISION,
    MILES_COMMIT,
    TrainPlan,
)
from .data import prepare_dataset
from .merge import merge_checkpoint
from .merge._contracts import MANIFEST
from .preflight import file_hash, inspect_backend, inspect_plan, runtime_identity, tokenizer_for
from .run import execute as execute_training
from .run import write_json

MILES_ROOT = Path("/opt/catan-miles")
BRIDGE_ROOT = Path("/opt/catan-bridge")
MEGATRON_ROOT = Path("/opt/catan-megatron")
BASE = Path("/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots") / BASE_REVISION
SOURCE = PROJECT_ROOT / "artifacts/generated/sft/symbolic_board_v2/train.jsonl"
RUNS = Path("/runs/miles-sft")
ADAPTER = "/runs/catan-vision-sft/board-fluency-extension-20260915-r04/training/checkpoints/checkpoint-512"

app = modal.App("catan-miles-topology-sft")
cache = modal.Volume.from_name("catan-hf-cache")
runs = modal.Volume.from_name("catan-sft-runs")
image = (
    modal.Image.from_registry(IMAGE_REF)
    .run_commands(
        "test -d /opt && test ! -e /opt/catan-miles && test ! -e /opt/catan-bridge && test ! -e /opt/catan-megatron",
        "git clone --filter=blob:none --no-checkout https://github.com/radixark/miles.git /opt/catan-miles",
        f"git -C /opt/catan-miles checkout --detach {MILES_COMMIT}",
        "git clone --filter=blob:none --no-checkout https://github.com/radixark/Megatron-Bridge.git /opt/catan-bridge",
        f"git -C /opt/catan-bridge checkout --detach {BRIDGE_REVISION}",
        "git clone --filter=blob:none --no-checkout https://github.com/radixark/Megatron-LM.git /opt/catan-megatron",
        f"git -C /opt/catan-megatron checkout --detach {MEGATRON_REVISION}",
        "uv pip install --python /opt/sglang/bin/python --no-deps --no-build-isolation -e /opt/catan-bridge",
        "uv pip install --python /opt/sglang/bin/python modal==1.2.4",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub",
          "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/root:/opt/catan-miles:/opt/catan-bridge/src:/opt/catan-megatron"})
    .add_local_file(SOURCE, "/sft-source/train.jsonl", copy=True)
    .add_local_python_source("sft")
)


def _name(run_name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,100}", run_name) is None:
        raise ValueError("invalid run name")


@app.function(image=image, volumes={"/cache": cache, "/runs": runs}, cpu=8,
              memory=65536, timeout=3600, retries=0, max_containers=1, scaledown_window=2)
def prepare_cpu(run_name: str, adapter_dir: str) -> JsonDict:
    _name(run_name)
    cache.reload()
    runs.reload()
    adapter = Path(adapter_dir).resolve()
    if not adapter.is_relative_to(Path("/runs").resolve()) or not adapter.is_dir():
        raise FileNotFoundError(f"source adapter absent from selected workspace: {adapter}")
    if not BASE.is_dir():
        raise FileNotFoundError(f"pinned cached base absent: {BASE}")
    root = RUNS / run_name
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "prepare.json", {"status": "preparing", "adapter": str(adapter)})
    runs.commit()
    try:
        runtime = runtime_identity(MILES_ROOT, MEGATRON_ROOT, BRIDGE_ROOT)
        merged = root / "merged-base"
        merge_checkpoint(BASE, adapter, merged, base_revision=BASE_REVISION)
        data = prepare_dataset(Path("/sft-source/train.jsonl"), root / "data", tokenizer_for(merged),
                               tokenizer_identity=file_hash(merged / MANIFEST))
        plan = TrainPlan(merged, root / "data", root / "training")
        write_json(root / "prepare.json", {
            "status": "ready", "adapter": str(adapter), "image": IMAGE_REF,
            "runtime": runtime, "dataset": data, "preflight": inspect_plan(plan),
        })
    except BaseException as error:
        write_json(root / "prepare.json", {"status": "failed", "error": str(error)[:2000],
                                           "error_type": type(error).__name__})
        raise
    finally:
        runs.commit()
    return load_json_dict(root / "prepare.json")


@app.function(image=image, volumes={"/cache": cache, "/runs": runs}, gpu="H200", cpu=8,
              memory=131072, timeout=1800, startup_timeout=600, retries=0,
              max_containers=1, scaledown_window=2)
def train_gpu(run_name: str, prepare_sha256: str, training_name: str = "training") -> JsonDict:
    _name(run_name)
    _name(training_name)
    cache.reload()
    runs.reload()
    root = RUNS / run_name
    if file_hash(root / "prepare.json") != prepare_sha256:
        raise ValueError("CPU receipt changed")
    prepared = load_json_dict(root / "prepare.json")
    if prepared.get("status") != "ready":
        raise ValueError("CPU preparation did not pass")
    # Transformer Engine links libcuda even for configuration-only imports. Check
    # these before any expensive weight scan or model allocation on the GPU worker.
    print(json.dumps(inspect_backend(root / "merged-base", MEGATRON_ROOT, BRIDGE_ROOT)), flush=True)
    plan = TrainPlan(root / "merged-base", root / "data", root / training_name)
    try:
        execute_training(plan, MILES_ROOT, MEGATRON_ROOT, BRIDGE_ROOT, dry_run=False,
                         expected_preflight=as_dict(prepared["preflight"]))
    finally:
        runs.commit()
    return load_json_dict(plan.output / "launch.json")


@app.function(image=image, volumes={"/runs": runs}, cpu=2, memory=8192, timeout=300, retries=0)
def prepared_receipt(run_name: str) -> tuple[JsonDict, str]:
    _name(run_name)
    runs.reload()
    path = RUNS / run_name / "prepare.json"
    report = {"runtime": runtime_identity(MILES_ROOT, MEGATRON_ROOT, BRIDGE_ROOT)}
    print(json.dumps(report, sort_keys=True), flush=True)
    return load_json_dict(path), file_hash(path)


@app.local_entrypoint()
def main(run_name: str, adapter_dir: str = ADAPTER, prepare_only: bool = False,
         execute: bool = False, use_prepared: bool = False, training_name: str = "training") -> None:
    """Default prints a plan; --prepare-only runs CPU; --execute also runs two GPU updates."""
    _name(run_name)
    _name(training_name)
    if prepare_only and execute:
        raise ValueError("choose --prepare-only or --execute")
    profile = os.environ.get("MODAL_PROFILE", "tetracorp")
    if not (prepare_only or execute):
        print(json.dumps({"status": "plan_only", "profile": profile, "run_name": run_name,
                          "adapter_dir": adapter_dir, "optimizer_steps": 2, "gpu": "H200:1",
                          "gpu_timeout_seconds": 1800,
                          "command": f"{sys.executable} -m modal run -m sft.miles_sft.modal_run "
                                     f"--run-name {run_name} --execute"}, indent=2))
        return
    local = PROJECT_ROOT / "artifacts/runs/sft" / run_name
    local.mkdir(parents=True, exist_ok=use_prepared)
    if not use_prepared:
        prepare_cpu.remote(run_name, adapter_dir)
    receipt, digest = prepared_receipt.remote(run_name)
    write_json(local / "prepare.json", receipt)
    if receipt.get("status") != "ready":
        raise ValueError(f"CPU preflight failed: {as_str(receipt.get('error'))}")
    if execute:
        result = train_gpu.remote(run_name, digest, training_name)
        write_json(local / f"launch-{training_name}.json", {
            "profile": profile, "app_id": app.app_id, "result": result,
        })
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({"status": "cpu_ready", "profile": profile, "run_name": run_name}, indent=2))
