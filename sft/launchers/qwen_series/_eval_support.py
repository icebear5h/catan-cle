"""Volumes, image and eval-bundle upload behind the Qwen-series eval launcher."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import modal

from sft.json_types import JsonDict, as_dict, loads_json
from sft.paths import resolve_dataset_asset, resolve_dataset_image
from sft.scripts.train.train_trl_catan_vision import (
    ACCELERATE_VERSION,
    HUGGINGFACE_HUB_VERSION,
    PEFT_VERSION,
    PILLOW_VERSION,
    SAFETENSORS_VERSION,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    TRANSFORMERS_VERSION,
    load_token_inventory,
)

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
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"accelerate=={ACCELERATE_VERSION}",
        f"peft=={PEFT_VERSION}",
        f"huggingface-hub=={HUGGINGFACE_HUB_VERSION}",
        f"safetensors=={SAFETENSORS_VERSION}",
        f"Pillow=={PILLOW_VERSION}",
        "bitsandbytes==0.49.2",
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
    .add_local_python_source("cle")
    .add_local_python_source("evals")
    .add_local_python_source("sft")
)



class EvalKwargs(TypedDict):
    """Keyword arguments shared by `eval_remote` and `eval_h200`."""

    eval_jsonl: str
    output_dir: str
    adapter_dir: str | None
    model_id: str
    limit: int | None
    max_new_tokens: int
    bits: int
    token_inventory: str | None
    batch_size: int
    image_variant: str
    occlusion_margin: float
    candidate_scoring: bool
    long_max_new_tokens: int
    long_batch_size: int
    preserve_visual_fp32: bool


def _image_reference(row: JsonDict) -> str:
    if row.get("image"):
        return str(row["image"])
    images = row.get("images")
    if isinstance(images, list) and len(images) == 1:
        return str(images[0])
    raise ValueError("eval row must reference exactly one image")


def _iter_jsonl(path: Path) -> Iterator[JsonDict]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield as_dict(loads_json(line))


def upload_eval_jsonl(
    eval_jsonl: Path,
    remote_dir: str,
    *,
    eval_set_id: str,
    image_root: Path | None = None,
    token_inventory: Path | None = None,
) -> tuple[str, str | None]:
    """Upload eval rows, explicit-root images, and optional token inventory."""

    eval_jsonl = eval_jsonl.resolve()
    with eval_jsonl.open("rb") as handle:
        source_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    remote_dir = "/" + remote_dir.strip("/")
    remote_images_dir = f"{remote_dir}/images"
    remote_jsonl = f"{remote_dir}/eval.jsonl"

    rows: list[JsonDict] = []
    image_map: dict[str, str] = {}

    with sft_data.batch_upload(force=True) as batch:
        for row in _iter_jsonl(eval_jsonl):
            reference = _image_reference(row)
            local_image = (
                resolve_dataset_image(image_root, reference)
                if image_root is not None
                else resolve_dataset_asset(eval_jsonl, reference)
            )
            if not local_image.exists():
                raise FileNotFoundError(local_image)
            remote_image = image_map.get(str(local_image))
            if remote_image is None:
                suffix = local_image.suffix or ".png"
                remote_image = f"{remote_images_dir}/{len(image_map):06d}{suffix}"
                image_map[str(local_image)] = remote_image
                batch.put_file(local_image, remote_image)

            row = dict(row)
            metadata = dict(as_dict(row.get("metadata") or {}))
            metadata.update(
                eval_set_id=eval_set_id,
                eval_source_sha256=source_digest,
            )
            row["metadata"] = metadata
            row["image"] = f"{REMOTE_DATA_MOUNT}{remote_image}"
            row.pop("images", None)
            rows.append(row)

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            temp_jsonl = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        batch.put_file(temp_jsonl, remote_jsonl)
        remote_inventory = None
        if token_inventory is not None:
            load_token_inventory(token_inventory)
            remote_inventory = f"{remote_dir}/trainable_tokens.json"
            batch.put_file(token_inventory.resolve(), remote_inventory)

    print(f"uploaded_rows={len(rows)}")
    print(f"uploaded_images={len(image_map)}")
    print(f"remote_eval_jsonl={REMOTE_DATA_MOUNT}{remote_jsonl}")
    return (
        f"{REMOTE_DATA_MOUNT}{remote_jsonl}",
        f"{REMOTE_DATA_MOUNT}{remote_inventory}" if remote_inventory else None,
    )
