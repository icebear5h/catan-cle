"""Modal launcher for Catan Qwen3-VL SFT.

Usage:

    modal run sft/modal_train.py --train-jsonl sft/data/catan_vlm_sft_train.jsonl --max-steps 5

The local entrypoint uploads the JSONL and referenced images into a Modal Volume,
then starts a capped GPU training job.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import modal


APP_NAME = "catan-qwen-vl-sft"
REMOTE_WORKDIR = "/workspace"
REMOTE_DATA_MOUNT = "/data"
REMOTE_RUNS_MOUNT = "/runs"
REMOTE_CACHE_MOUNT = "/cache"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("catan-hf-cache", create_if_missing=True)
sft_data = modal.Volume.from_name("catan-sft-data", create_if_missing=True)
sft_runs = modal.Volume.from_name("catan-sft-runs", create_if_missing=True)

sft_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "transformers>=4.57.1",
        "trl>=0.21.0",
        "peft>=0.17.0",
        "accelerate>=1.10.0",
        "datasets>=3.0.0",
        "bitsandbytes>=0.46.0",
        "qwen-vl-utils>=0.0.14",
        "Pillow>=10.0.0",
        "pyyaml>=6.0.0",
        "tensorboard>=2.14.0",
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
        }
    )
    .add_local_python_source("data_pipeline")
    .add_local_python_source("engine")
    .add_local_file(
        "sft/scripts/train_qwen_vl_sft.py",
        remote_path=f"{REMOTE_WORKDIR}/sft/scripts/train_qwen_vl_sft.py",
    )
    .add_local_file(
        "sft/configs/qwen3_vl_8b_qlora.yaml",
        remote_path=f"{REMOTE_WORKDIR}/sft/configs/qwen3_vl_8b_qlora.yaml",
    )
    .add_local_dir(
        "configs/training_configs",
        remote_path=f"{REMOTE_WORKDIR}/configs/training_configs",
    )
)


def _iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def upload_train_jsonl(train_jsonl: Path, remote_dir: str) -> str:
    """Upload a local SFT JSONL and its referenced images into the data volume."""

    train_jsonl = train_jsonl.resolve()
    remote_dir = "/" + remote_dir.strip("/")
    remote_images_dir = f"{remote_dir}/images"
    remote_jsonl = f"{remote_dir}/train.jsonl"

    rows = []
    image_map: dict[str, str] = {}

    with sft_data.batch_upload(force=True) as batch:
        for row in _iter_jsonl(train_jsonl):
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
    print(f"remote_train_jsonl={REMOTE_DATA_MOUNT}{remote_jsonl}")
    return f"{REMOTE_DATA_MOUNT}{remote_jsonl}"


@app.function(
    image=sft_image,
    gpu="L40S",
    volumes={
        REMOTE_CACHE_MOUNT: hf_cache,
        REMOTE_DATA_MOUNT: sft_data,
        REMOTE_RUNS_MOUNT: sft_runs,
    },
    timeout=60 * 60 * 8,
)
def train_remote(
    train_jsonl: str,
    config_path: str = f"{REMOTE_WORKDIR}/sft/configs/qwen3_vl_8b_qlora.yaml",
    output_dir: str = f"{REMOTE_RUNS_MOUNT}/qwen3-vl-8b-catan-qlora",
    max_steps: int | None = 5,
    num_train_epochs: float | None = None,
) -> dict[str, str | int | None]:
    import subprocess

    command = [
        "python",
        f"{REMOTE_WORKDIR}/sft/scripts/train_qwen_vl_sft.py",
        "--config",
        config_path,
        "--train-jsonl",
        train_jsonl,
        "--output-dir",
        output_dir,
    ]
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if num_train_epochs is not None:
        command.extend(["--num-train-epochs", str(num_train_epochs)])

    print("running:", " ".join(command))
    subprocess.run(command, check=True)
    sft_runs.commit()
    return {"output_dir": output_dir, "max_steps": max_steps}


@app.local_entrypoint()
def main(
    train_jsonl: str,
    remote_dir: str = "catan-vlm-sft/current",
    config_path: str = f"{REMOTE_WORKDIR}/sft/configs/qwen3_vl_8b_qlora.yaml",
    output_dir: str = f"{REMOTE_RUNS_MOUNT}/qwen3-vl-8b-catan-qlora",
    max_steps: int = 5,
    num_train_epochs: float | None = None,
):
    remote_train_jsonl = upload_train_jsonl(Path(train_jsonl), remote_dir)
    result = train_remote.remote(
        train_jsonl=remote_train_jsonl,
        config_path=config_path,
        output_dir=output_dir,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
    )
    print(result)
