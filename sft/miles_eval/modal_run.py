"""Bounded one-H200 execution of the stock Cartesian panel using official Miles."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import cast

import modal

from sft.miles_eval.contracts import (
    JsonObject,
    compact,
    json_list,
    json_object,
    parse_json,
    require,
)
from sft.miles_eval.preflight import file_hash, preflight
from sft.miles_eval.run import CONTEXT, MILES_REVISION, NEW_TOKENS, miles_arguments, prepared_panels
from sft.paths import PROJECT_ROOT

IMAGE_REF = "radixark/miles@sha256:6628bff749ffd32e6a62b479a1128daee25a8c0e3c28eb86301a0d620f5dd598"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CHECKPOINT = Path("/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots") / MODEL_REVISION
MILES_ROOT = Path("/opt/catan-miles")
DATA = Path("/eval-data")
RUNS = Path("/runs/miles-eval")
DATASET_NAME = os.environ.get("CATAN_MILES_DATASET", "miles_cartesian_eval_v2")
require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]+", DATASET_NAME) is not None, "invalid dataset name")
DATA_SOURCE = PROJECT_ROOT / "artifacts/generated/sft" / DATASET_NAME
GPU_SECONDS = 1740

cache = modal.Volume.from_name("catan-hf-cache")
runs = modal.Volume.from_name("catan-sft-runs")
volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/cache": cache, "/runs": runs,
}
app = modal.App("catan-miles-cartesian-eval")
image = (
    modal.Image.from_registry(IMAGE_REF)
    .run_commands(
        "test -d /opt && test ! -e /opt/catan-miles",
        "git clone --filter=blob:none --no-checkout https://github.com/radixark/miles.git /opt/catan-miles",
        f"git -C /opt/catan-miles checkout --detach {MILES_REVISION}",
        "uv pip install --python /opt/sglang/bin/python modal==1.2.4 networkx==3.6.1 "
        "flask==3.1.2 flask-cors==6.0.2 flask-socketio==5.6.0 jsonschema==4.25.1",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub",
          "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
          "CATAN_MILES_DATASET": DATASET_NAME,
          "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root:/opt/catan-miles:/root/Megatron-LM"})
    .add_local_dir(DATA_SOURCE, remote_path=str(DATA), copy=True)
    .add_local_python_source("cle", "evals", "sft", "data_pipeline", "playground")
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: JsonObject) -> None:
    path.write_text(compact(payload) + "\n")


def version_report() -> JsonObject:
    revisions = {
        "miles": str(MILES_ROOT), "sglang": "/sgl-workspace/sglang", "megatron": "/root/Megatron-LM",
    }
    return {
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "sglang", "ray", "networkx", "pydantic")},
        "source_revisions": {name: subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip() for name, root in revisions.items()},
    }


@app.function(image=image, volumes=volumes, cpu=(4.0, 4.0), memory=(65536, 65536),
              timeout=1200, startup_timeout=600, retries=0, max_containers=1, scaledown_window=2)
def prepare_cpu(run_name: str) -> JsonObject:
    cache.reload()
    runs.reload()
    root = RUNS / run_name
    root.mkdir(parents=True, exist_ok=False)
    report: JsonObject = {"status": "preparing", "started_at": now(), "image": IMAGE_REF,
                          "dataset_name": DATASET_NAME,
                          "miles_revision": MILES_REVISION, "model_revision": MODEL_REVISION}
    write_json(root / "preflight.json", report)
    runs.commit()
    try:
        report["runtime"] = version_report()
        panels = prepared_panels(DATA)
        report["data_manifest_sha256"] = file_hash(DATA / "manifest.json")
        manifest = parse_json((DATA / "manifest.json").read_text())
        counts = [json_object(item)["rows"] for item in json_object(manifest["panels"]).values()]
        require(all(type(count) is int and count > 0 for count in counts), "invalid panel sizes")
        expected_rows = sum(cast(int, count) for count in counts)
        audit = preflight(CHECKPOINT, panels, context=CONTEXT, new_tokens=NEW_TOKENS)
        lengths = [json_object(item) for item in json_list(audit["token_lengths"])]
        require(len(lengths) == expected_rows and audit["requires_atlas_tokenizer"] is False,
                "expected all admitted stock-only panel rows")
        report.update(status="ready", preflight=audit, expected_rows=expected_rows,
                      panel_names=list(panels),
                      max_prompt_tokens=max(cast(int, item["prompt"]) for item in lengths),
                      max_gold_tokens=max(cast(int, item["gold"]) for item in lengths))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000])
    finally:
        report["ended_at"] = now()
        write_json(root / "preflight.json", report)
        runs.commit()
    return report


@app.function(image=image, volumes=volumes, gpu="H200", cpu=(8.0, 8.0),
              memory=(131072, 131072), timeout=1800, startup_timeout=600,
              retries=0, max_containers=1, scaledown_window=2)
def evaluate(run_name: str, prepared: JsonObject) -> JsonObject:
    started = time.monotonic()
    cache.reload()
    runs.reload()
    root = RUNS / run_name
    require(prepared.get("status") == "ready", "CPU preparation failed")
    require(parse_json((root / "preflight.json").read_text()) == prepared, "preflight receipt changed")
    require(file_hash(DATA / "manifest.json") == prepared["data_manifest_sha256"], "panel manifest changed")
    output = root / "evaluation"
    output.mkdir(exist_ok=False)
    (output / "debug").mkdir()
    command = [sys.executable, "-m", "sft.miles_eval.worker",
               *miles_arguments(CHECKPOINT, prepared_panels(DATA), output)]
    result: JsonObject = {"status": "running", "started_at": now(), "command": list(command),
                          "image": IMAGE_REF, "checkpoint": str(CHECKPOINT), "gpu": "H200"}
    write_json(root / "gpu.json", result)
    runs.commit()
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True,
        ).strip()
        environment = {**os.environ, "CATAN_MILES_ROOT": str(MILES_ROOT)}
        with (root / "worker.log").open("wb") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=max(1, GPU_SECONDS - (time.monotonic() - started)))
        summary = parse_json((output / "results/eval-0/summary.json").read_text())
        require(summary.get("total") == prepared["expected_rows"], "evaluation row count differs")
        require(set(json_object(summary["panels"])) == set(json_list(prepared["panel_names"])),
                "evaluation panels differ")
        result.update(status="completed", summary=summary)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000])
    finally:
        result.update(ended_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(root / "gpu.json", result)
        runs.commit()
    return result


@app.local_entrypoint()
def main(run_name: str = "miles-cartesian-stock-20260921-r01") -> None:
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]+", run_name) is not None, "invalid run name")
    local = PROJECT_ROOT / "artifacts/runs/sft" / run_name
    local.mkdir(parents=True, exist_ok=False)
    receipt: JsonObject = {"run_name": run_name, "app_id": app.app_id, "image": IMAGE_REF,
                           "dataset_name": DATASET_NAME, "profile": os.environ.get("MODAL_PROFILE"),
                           "miles_revision": MILES_REVISION, "model_revision": MODEL_REVISION,
                           "remote_root": str(RUNS / run_name), "started_at": now(),
                           "gpu_timeout_seconds": 1800, "startup_timeout_seconds": 600}
    write_json(local / "launch.json", receipt)
    prepared = prepare_cpu.remote(run_name)
    write_json(local / "preflight.json", prepared)
    require(prepared.get("status") == "ready", f"CPU preflight failed: {prepared.get('error')}")
    call = evaluate.spawn(run_name, prepared)
    receipt.update(gpu_call_id=call.object_id, status="gpu_spawned")
    write_json(local / "launch.json", receipt)
    print(json.dumps({**receipt, "max_prompt_tokens": prepared["max_prompt_tokens"],
                      "max_gold_tokens": prepared["max_gold_tokens"]}, indent=2))
