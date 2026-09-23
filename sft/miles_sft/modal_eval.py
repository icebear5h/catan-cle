"""One-H200 SGLang reload smoke on a completed Miles SFT serving export."""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

from sft.json_types import JsonDict, as_dict, load_json_dict
from sft.miles_eval.preflight import file_hash
from sft.miles_eval.run import MILES_REVISION, launch
from sft.paths import PROJECT_ROOT

from .config import IMAGE_REF
from .evaluation import final_export
from .export import validate_complete_export
from .run import write_json

app = modal.App("catan-miles-sft-reload")
runs = modal.Volume.from_name("catan-sft-runs")
cache = modal.Volume.from_name("catan-hf-cache")
MILES_ROOT = Path("/opt/catan-miles-eval")
SOURCE = PROJECT_ROOT / "artifacts/generated/sft/miles_topology_reload_eval_v1/panels"
image = (
    modal.Image.from_registry(IMAGE_REF)
    .run_commands(
        "test -d /opt && test ! -e /opt/catan-miles-eval",
        "git clone --filter=blob:none --no-checkout https://github.com/radixark/miles.git /opt/catan-miles-eval",
        f"git -C /opt/catan-miles-eval checkout --detach {MILES_REVISION}",
        "uv pip install --python /opt/sglang/bin/python modal==1.2.4 networkx==3.6.1 "
        "flask==3.1.2 flask-cors==6.0.2 flask-socketio==5.6.0 jsonschema==4.25.1",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub",
          "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/root:/opt/catan-miles-eval:/root/Megatron-LM"})
    .add_local_dir(SOURCE, "/reload-panel", copy=True)
    .add_local_python_source("cle", "evals", "sft", "data_pipeline", "playground")
)


@app.function(image=image, volumes={"/cache": cache, "/runs": runs}, cpu=8,
              memory=65536, timeout=1200, retries=0, max_containers=1, scaledown_window=2)
def prepare_cpu(run_name: str, training_name: str = "training") -> str:
    runs.reload()
    training = Path("/runs/miles-sft") / run_name / training_name
    checkpoint = final_export(training)
    receipt = load_json_dict(training / "receipts/run.json")
    export = as_dict(as_dict(receipt["checkpoint"])["hf"])
    validate_complete_export(training.parent / "merged-base", checkpoint,
                             expected_sha256=str(export["composition_manifest_sha256"]))
    return file_hash(training / "receipts/run.json")


@app.function(image=image, volumes={"/cache": cache, "/runs": runs}, gpu="H200", cpu=8,
              memory=131072, timeout=1800, retries=0, max_containers=1, scaledown_window=2)
def evaluate(run_name: str, receipt_sha256: str, training_name: str = "training") -> JsonDict:
    cache.reload()
    runs.reload()
    root = Path("/runs/miles-sft") / run_name
    training = root / training_name
    if file_hash(training / "receipts/run.json") != receipt_sha256:
        raise ValueError("training receipt changed after reload preflight")
    try:
        launch(MILES_ROOT, final_export(training), Path("/reload-panel"), root / "reload-eval", execute=True)
    finally:
        runs.commit()
    return load_json_dict(root / "reload-eval/results/eval-0/summary.json")


@app.local_entrypoint()
def main(run_name: str, training_name: str = "training") -> None:
    if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,100}", name) is None
           for name in (run_name, training_name)):
        raise ValueError("invalid run name")
    receipt = prepare_cpu.remote(run_name, training_name)
    result = evaluate.remote(run_name, receipt, training_name)
    local = PROJECT_ROOT / "artifacts/runs/sft" / run_name
    local.mkdir(parents=True, exist_ok=True)
    write_json(local / "reload-eval.json", {"app_id": app.app_id, "summary": result,
                                           "training_receipt_sha256": receipt})
    print(json.dumps(result, indent=2))
