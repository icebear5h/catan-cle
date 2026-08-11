"""Modal launcher for Qwen-VL Catan adapter generation evals."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import modal


APP_NAME = "catan-qwen-series-eval"
REMOTE_WORKDIR = "/workspace"
REMOTE_DATA_MOUNT = "/data"
REMOTE_RUNS_MOUNT = "/runs"
REMOTE_CACHE_MOUNT = "/cache"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("catan-hf-cache", create_if_missing=True)
sft_data = modal.Volume.from_name("catan-sft-data", create_if_missing=True)
sft_runs = modal.Volume.from_name("catan-sft-runs", create_if_missing=True)

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==5.3.0",
        "accelerate==1.10.1",
        "peft==0.15.2",
        "bitsandbytes==0.49.2",
        "qwen-vl-utils==0.0.14",
        "Pillow==11.3.0",
        "av==17.0.1",
        "sentencepiece",
        "protobuf",
    )
    .workdir(REMOTE_WORKDIR)
    .env(
        {
            "HF_HOME": f"{REMOTE_CACHE_MOUNT}/huggingface",
            "HF_HUB_CACHE": f"{REMOTE_CACHE_MOUNT}/huggingface/hub",
            "TRANSFORMERS_CACHE": f"{REMOTE_CACHE_MOUNT}/huggingface/transformers",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": REMOTE_WORKDIR,
        }
    )
    .add_local_python_source("data_pipeline")
    .add_local_python_source("engine")
    .add_local_file(
        "sft/scripts/eval_qwen_vl_adapter.py",
        remote_path=f"{REMOTE_WORKDIR}/sft/scripts/eval_qwen_vl_adapter.py",
    )
)


def _iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def upload_eval_jsonl(eval_jsonl: Path, remote_dir: str) -> str:
    """Upload eval JSONL and referenced images into the Modal data volume."""

    eval_jsonl = eval_jsonl.resolve()
    remote_dir = "/" + remote_dir.strip("/")
    remote_images_dir = f"{remote_dir}/images"
    remote_jsonl = f"{remote_dir}/eval.jsonl"

    rows = []
    image_map: dict[str, str] = {}

    with sft_data.batch_upload(force=True) as batch:
        for row in _iter_jsonl(eval_jsonl):
            local_image = Path(row["image"]).expanduser().resolve()
            if not local_image.exists():
                raise FileNotFoundError(local_image)
            remote_image = image_map.get(str(local_image))
            if remote_image is None:
                suffix = local_image.suffix or ".png"
                remote_image = f"{remote_images_dir}/{len(image_map):06d}{suffix}"
                image_map[str(local_image)] = remote_image
                batch.put_file(local_image, remote_image)

            row = dict(row)
            row["image"] = f"{REMOTE_DATA_MOUNT}{remote_image}"
            rows.append(row)

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            temp_jsonl = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        batch.put_file(temp_jsonl, remote_jsonl)

    print(f"uploaded_rows={len(rows)}")
    print(f"uploaded_images={len(image_map)}")
    print(f"remote_eval_jsonl={REMOTE_DATA_MOUNT}{remote_jsonl}")
    return f"{REMOTE_DATA_MOUNT}{remote_jsonl}"


@app.function(
    image=eval_image,
    gpu="L40S",
    volumes={
        REMOTE_CACHE_MOUNT: hf_cache,
        REMOTE_DATA_MOUNT: sft_data,
        REMOTE_RUNS_MOUNT: sft_runs,
    },
    timeout=60 * 60 * 8,
)
def eval_remote(
    eval_jsonl: str,
    output_dir: str,
    adapter_dir: str | None = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    limit: int | None = None,
    max_new_tokens: int = 256,
) -> dict:
    import subprocess

    command = [
        "python",
        f"{REMOTE_WORKDIR}/sft/scripts/eval_qwen_vl_adapter.py",
        "--eval-jsonl",
        eval_jsonl,
        "--output-dir",
        output_dir,
        "--model-id",
        model_id,
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if adapter_dir:
        command.extend(["--adapter-dir", adapter_dir])
    if limit is not None:
        command.extend(["--limit", str(limit)])

    print("running:", " ".join(command))
    subprocess.run(command, check=True)
    sft_runs.commit()

    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {"output_dir": output_dir}


@app.local_entrypoint()
def main(
    eval_jsonl: str,
    remote_dir: str = "catan-qwen-series-eval/heldout",
    output_dir: str = f"{REMOTE_RUNS_MOUNT}/qwen-series-eval",
    adapter_dir: str | None = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    limit: int | None = None,
    max_new_tokens: int = 256,
):
    remote_eval_jsonl = upload_eval_jsonl(Path(eval_jsonl), remote_dir)
    result = eval_remote.remote(
        eval_jsonl=remote_eval_jsonl,
        output_dir=output_dir,
        adapter_dir=adapter_dir,
        model_id=model_id,
        limit=limit,
        max_new_tokens=max_new_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
