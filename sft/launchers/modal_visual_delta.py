"""Launcher for bounded CPU-only extraction beside cached HF/checkpoint weights; never trains."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import modal
import torch

from sft.json_types import JsonDict, load_json_dict
from sft.scripts.report.extract_visual_delta import extract, write_json
from sft.scripts.report.extract_visual_delta._base import GroupSummary

app = modal.App("catan-visual-delta-cpu")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("safetensors==0.7.0", "packaging==25.0", "numpy==2.2.6")
    .run_commands("python -c 'import torch; from safetensors.torch import save_file; print(torch.__version__)'")
    .add_local_python_source("sft")
)
runs = modal.Volume.from_name("catan-sft-runs")
cache = modal.Volume.from_name("catan-hf-cache")
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL = "Qwen/Qwen3.8-27B"
CHECKPOINT = "/runs/catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9/checkpoints/checkpoint-384"
DESTINATION = "/runs/catan-vision-diagnostics/visual-delta-gauss-s2-ck384-20260905-fp64"



class DeriveResult(TypedDict):
    """The remote extraction's summary, as returned to the local receipt."""

    output_dir: str
    summary: dict[str, GroupSummary]
    elapsed_seconds: float
    status: str


@app.function(image=image, cpu=8, memory=24576, timeout=1200,
              volumes={"/runs": runs, "/cache": cache}, retries=0)
def derive(hf_metadata: JsonDict) -> DeriveResult:
    torch.set_num_threads(8)
    snapshot = Path("/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots") / REVISION
    root = Path(DESTINATION)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    info = root / "hf_model_info.json"
    write_json(info, hf_metadata)
    write_json(root / "provenance.json", {
        "model_id": MODEL, "hf_revision": REVISION, "checkpoint": CHECKPOINT,
        "revision_evidence": "sole snapshot in training HF cache; refs/main agrees; original training config did not pin revision",
        "base_file_verification": "SHA256 compared to HF revision API LFS metadata",
        "compute": {"gpu": None, "cpu": 8, "memory_mib": 24576, "timeout_seconds": 1200},
        "trained_visual_expected_sha256": "94534c119727435bc2011fb76a6038c1e71facd120abbf482e71bd80a2d252f2",
    })
    try:
        with torch.inference_mode():
            report = extract(
                trained_path=Path(CHECKPOINT) / "visual_model.safetensors",
                trained_sha256="94534c119727435bc2011fb76a6038c1e71facd120abbf482e71bd80a2d252f2",
                base_index=snapshot / "model.safetensors.index.json", base_dir=snapshot,
                hf_info=info, output_dir=root / "derived", revision=REVISION, model_id=MODEL,
                factor_rank=256, base_load_dtype="bfloat16",
            )
        return DeriveResult(output_dir=str(root / "derived"), summary=report["summary"],
                            elapsed_seconds=report["elapsed_seconds"], status=report["status"])
    finally:
        runs.commit()


@app.local_entrypoint()
def main(hf_info: str, output: str, dry_run: bool = True) -> None:
    metadata = load_json_dict(hf_info)
    if metadata.get("sha") != REVISION or metadata.get("id") != MODEL:
        raise ValueError("HF metadata model/revision mismatch")
    plan = {"model_id": MODEL, "revision": REVISION, "checkpoint": CHECKPOINT,
            "destination": DESTINATION, "gpu": None, "cpu": 8, "memory_mib": 24576,
            "timeout_seconds": 1200, "training": False, "dry_run": dry_run}
    print(json.dumps(plan, indent=2))
    if dry_run:
        return
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite local receipt {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = derive.remote(metadata)
    target.write_text(json.dumps({"plan": plan, "result": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))
