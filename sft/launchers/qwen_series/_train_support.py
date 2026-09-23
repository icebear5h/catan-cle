"""Volumes, image and conversation upload behind the legacy train launcher."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import modal

from sft.json_types import as_dict, as_str
from sft.paths import resolve_dataset_asset, resolve_dataset_image
from sft.scripts.builders.convert_to_qwen_series_sft import convert_row, iter_jsonl

APP_NAME = "catan-qwen-series-sft"
QWEN_REPO = "https://github.com/2U1/Qwen-VL-Series-Finetune.git"
QWEN_COMMIT = "130ad7ccafe06a2ae5377bad9d512027d8225dc5"

REMOTE_WORKDIR = "/workspace"
REMOTE_QWEN_DIR = f"{REMOTE_WORKDIR}/qwen-vl-series-finetune"
REMOTE_DATA_MOUNT = "/data"
REMOTE_RUNS_MOUNT = "/runs"
REMOTE_CACHE_MOUNT = "/cache"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("catan-hf-cache", create_if_missing=True)
sft_data = modal.Volume.from_name("catan-sft-data", create_if_missing=True)
sft_runs = modal.Volume.from_name("catan-sft-runs", create_if_missing=True)

qwen_series_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==5.3.0",
        "accelerate==1.10.1",
        "datasets==3.5.1",
        "peft==0.15.2",
        "trl==0.25.0",
        "bitsandbytes==0.49.2",
        "qwen-vl-utils==0.0.14",
        "ujson==5.10.0",
        "Pillow==11.3.0",
        "av==17.0.1",
        "tensorboard>=2.14.0",
        "sentencepiece",
        "protobuf",
    )
    .run_commands(
        f"git clone --depth 1 {QWEN_REPO} {REMOTE_QWEN_DIR}",
        f"cd {REMOTE_QWEN_DIR} && git fetch --depth 1 origin {QWEN_COMMIT} && git checkout {QWEN_COMMIT}",
    )
    .workdir(REMOTE_QWEN_DIR)
    .env(
        {
            "HF_HOME": f"{REMOTE_CACHE_MOUNT}/huggingface",
            "HF_HUB_CACHE": f"{REMOTE_CACHE_MOUNT}/huggingface/hub",
            "TRANSFORMERS_CACHE": f"{REMOTE_CACHE_MOUNT}/huggingface/transformers",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": f"{REMOTE_QWEN_DIR}/src:{REMOTE_WORKDIR}",
        }
    )
    .add_local_python_source("cle")
    .add_local_python_source("data_pipeline")
    .add_local_python_source("sft")
)



def upload_qwen_series_json(
    train_jsonl: Path,
    remote_dir: str,
    *,
    image_root: Path | None = None,
) -> tuple[str, str]:
    """Upload converted Qwen conversations from an optional explicit image root."""

    train_jsonl = train_jsonl.resolve()
    remote_dir = "/" + remote_dir.strip("/")
    remote_images_dir = f"{remote_dir}/images"
    remote_json = f"{remote_dir}/train_qwen_series.json"

    rows = []
    image_map: dict[str, str] = {}

    with sft_data.batch_upload(force=True) as batch:
        for _, value in iter_jsonl(train_jsonl):
            row = as_dict(value)
            image_name = None
            if row.get("image"):
                image = as_str(row["image"])
                local_image = (
                    resolve_dataset_image(image_root, image)
                    if image_root is not None
                    else resolve_dataset_asset(train_jsonl, image)
                )
                if not local_image.exists():
                    raise FileNotFoundError(local_image)
                image_name = image_map.get(str(local_image))
                if image_name is None:
                    suffix = local_image.suffix or ".png"
                    image_name = f"{len(image_map):06d}{suffix}"
                    image_map[str(local_image)] = image_name
                    batch.put_file(local_image, f"{remote_images_dir}/{image_name}")

            rows.append(convert_row(row, image_name=image_name))

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            temp_json = Path(handle.name)
            json.dump(rows, handle, indent=2, sort_keys=True)
            handle.write("\n")
        batch.put_file(temp_json, remote_json)

    print(f"uploaded_rows={len(rows)}")
    print(f"uploaded_images={len(image_map)}")
    print(f"remote_train_json={REMOTE_DATA_MOUNT}{remote_json}")
    return f"{REMOTE_DATA_MOUNT}{remote_json}", f"{REMOTE_DATA_MOUNT}{remote_images_dir}"
