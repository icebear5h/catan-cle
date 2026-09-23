"""One capped, no-retry evaluation of the completed Gaussian entity-marker run."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

from sft.analysis.marker_diagnostics import analyze, read_rows, select_rows, selection_manifest
from sft.json_types import JsonLikeDict, as_dict, as_list, as_str
from sft.launchers.qwen_series._eval_support import (
    eval_image,
    hf_cache,
    sft_data,
    sft_runs,
)

app = modal.App("catan-marker-mini-eval")
DATA = "/data/catan-vision-sft/datasets/73b1c523741d"
ADAPTER = "/runs/catan-vision-sft/catan-qwen38-gauss-s1-markers-entity-20260904/73b1c523741d/final"
OUTPUT = "/runs/qwen-series-eval/gauss-s1-entity-final-mini-20260904"


@app.function(
    image=eval_image, gpu="H200", timeout=600, retries=0,
    max_containers=1, scaledown_window=2,
    volumes={"/cache": hf_cache, "/data": sft_data, "/runs": sft_runs},
)
def evaluate(expected_ids_hash: str) -> JsonLikeDict:
    started = time.monotonic()
    output = Path(OUTPUT)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or automatically rerun {output}")
    source = read_rows(f"{DATA}/eval.jsonl")
    selected = select_rows(source)
    manifest = selection_manifest(source, selected)
    if manifest["rows"] != 308 or manifest["tokens"] != 154 or len(as_dict(manifest["boards"])) != 5:
        raise ValueError("unexpected dataset coverage")
    if manifest["row_ids_sha256"] != expected_ids_hash:
        raise ValueError("remote selection differs from local preflight")
    for row in source:
        reference = as_str(row.get("image") or as_list(row["images"])[0])
        path = Path(reference) if Path(reference).is_absolute() else Path(DATA) / "images" / reference
        if not path.is_file():
            raise FileNotFoundError(path)
        row["image"] = str(path)
        row.pop("images", None)
    output.mkdir(parents=True)
    (output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows_path = output / "selected.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in selected))
    command = [
        "python", "-m", "sft.scripts.eval.eval_qwen_vl_adapter",
        "--eval-jsonl", str(rows_path), "--output-dir", OUTPUT,
        "--adapter-dir", ADAPTER, "--model-id", "Qwen/Qwen3.8-27B",
        "--token-inventory", f"{DATA}/trainable_tokens.json",
        "--bits", "16", "--batch-size", "48", "--max-new-tokens", "16",
        "--image-variant", "original",
    ]
    status = "failed"
    error = None
    try:
        subprocess.run(command, check=True, timeout=max(1, 540 - (time.monotonic() - started)))
        status = "completed"
    except subprocess.TimeoutExpired:
        status, error = "timed_out", "evaluation exceeded the inner 9-minute budget; no retry"
    except subprocess.CalledProcessError as exc:
        error = f"evaluator exit code {exc.returncode}; no retry"
    finally:
        # Flush partial records too; do not turn an interrupted sample into a pass.
        try:
            records_path = output / "records.jsonl"
            records = read_rows(records_path) if records_path.exists() else []
            report = analyze(source, selected, records)
            if status == "completed" and not report["complete"]:
                status, error = "incomplete", "missing prediction rows"
            (output / "failures.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            run: JsonLikeDict = {"status": status, "error": error, "elapsed_seconds": time.monotonic() - started,
                   "adapter": ADAPTER, "output_dir": OUTPUT, "command": command,
                   "timeout_seconds": 600, "retries": 0, "expected_row_ids_sha256": expected_ids_hash}
            (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
            sft_runs.commit()
    return run


@app.local_entrypoint()
def main(expected_ids_hash: str) -> None:
    call = evaluate.spawn(expected_ids_hash)
    print(json.dumps({"function_call_id": call.object_id, "output_dir": OUTPUT, "timeout_seconds": 600, "retries": 0}))
