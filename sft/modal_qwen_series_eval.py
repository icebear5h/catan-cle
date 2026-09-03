"""Modal launcher for Qwen-VL Catan adapter generation evals."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import modal

from sft.paths import resolve_dataset_asset, resolve_dataset_image
from sft.scripts.train_trl_catan_vision import (
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


def _image_reference(row: dict) -> str:
    if row.get("image"):
        return str(row["image"])
    images = row.get("images")
    if isinstance(images, list) and len(images) == 1:
        return str(images[0])
    raise ValueError("eval row must reference exactly one image")


def _iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def upload_eval_jsonl(
    eval_jsonl: Path,
    remote_dir: str,
    *,
    image_root: Path | None = None,
    token_inventory: Path | None = None,
) -> tuple[str, str | None]:
    """Upload eval rows, explicit-root images, and optional token inventory."""

    eval_jsonl = eval_jsonl.resolve()
    remote_dir = "/" + remote_dir.strip("/")
    remote_images_dir = f"{remote_dir}/images"
    remote_jsonl = f"{remote_dir}/eval.jsonl"

    rows = []
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


def _run_eval(
    eval_jsonl: str,
    output_dir: str,
    adapter_dir: str | None = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    limit: int | None = None,
    max_new_tokens: int = 16,
    bits: int = 4,
    token_inventory: str | None = None,
    batch_size: int = 1,
    image_variant: str = "original",
    occlusion_margin: float = 0.03,
    candidate_scoring: bool = True,
) -> dict:
    command = [
        "python",
        "-m",
        "sft.scripts.eval_qwen_vl_adapter",
        "--eval-jsonl",
        *eval_jsonl.split(","),
        "--output-dir",
        output_dir,
        "--model-id",
        model_id,
        "--max-new-tokens",
        str(max_new_tokens),
        "--bits",
        str(bits),
        "--batch-size",
        str(batch_size),
        "--image-variant",
        image_variant,
        "--occlusion-margin",
        str(occlusion_margin),
    ]
    if adapter_dir:
        command.extend(["--adapter-dir", adapter_dir])
    if token_inventory:
        command.extend(["--token-inventory", token_inventory])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if not candidate_scoring:
        command.append("--no-candidate-scoring")

    print("running:", " ".join(command))
    subprocess.run(command, check=True)
    sft_runs.commit()

    for name in ("batch_summary.json", "summary.json"):
        summary_path = Path(output_dir) / name
        if summary_path.exists():
            return json.loads(summary_path.read_text())
    return {"output_dir": output_dir}


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
    max_new_tokens: int = 16,
    bits: int = 4,
    token_inventory: str | None = None,
    batch_size: int = 1,
    image_variant: str = "original",
    occlusion_margin: float = 0.03,
    candidate_scoring: bool = True,
) -> dict:
    return _run_eval(
        eval_jsonl,
        output_dir,
        adapter_dir,
        model_id,
        limit,
        max_new_tokens,
        bits,
        token_inventory,
        batch_size,
        image_variant,
        occlusion_margin,
        candidate_scoring,
    )


@app.function(
    image=eval_image,
    gpu="H200",
    volumes={
        REMOTE_CACHE_MOUNT: hf_cache,
        REMOTE_DATA_MOUNT: sft_data,
        REMOTE_RUNS_MOUNT: sft_runs,
    },
    timeout=60 * 60 * 8,
)
def eval_h200(
    eval_jsonl: str,
    output_dir: str,
    adapter_dir: str | None = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    limit: int | None = None,
    max_new_tokens: int = 16,
    bits: int = 16,
    token_inventory: str | None = None,
    batch_size: int = 48,
    image_variant: str = "original",
    occlusion_margin: float = 0.03,
    candidate_scoring: bool = True,
) -> dict:
    return _run_eval(
        eval_jsonl,
        output_dir,
        adapter_dir,
        model_id,
        limit,
        max_new_tokens,
        bits,
        token_inventory,
        batch_size,
        image_variant,
        occlusion_margin,
        candidate_scoring,
    )


@app.local_entrypoint()
def main(
    eval_jsonl: str,
    image_root: str | None = None,
    token_inventory: str | None = None,
    remote_dir: str = "catan-qwen-series-eval/heldout",
    output_dir: str = f"{REMOTE_RUNS_MOUNT}/qwen-series-eval",
    adapter_dir: str | None = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    limit: int | None = None,
    max_new_tokens: int = 16,
    bits: int = 4,
    batch_size: int = 1,
    gpu: str = "l40s",
    image_variant: str = "original",
    occlusion_margin: float = 0.03,
    candidate_scoring: bool = True,
    spawn_eval: bool = False,
):
    if bits not in {4, 8, 16}:
        raise ValueError("--bits must be 4, 8, or 16")
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if gpu not in {"l40s", "h200"}:
        raise ValueError("--gpu must be l40s or h200")
    for variant in image_variant.split(","):
        if variant.strip() not in {
            "original",
            "blank",
            "shuffle",
            "target_occlusion",
            "control_occlusion",
        }:
            raise ValueError(f"unsupported --image-variant: {variant}")
    # Comma-separated eval sets share one image root and token inventory and are
    # scored in one container with a single model load.
    remote_sets = []
    remote_token_inventory = None
    local_sets = [item.strip() for item in eval_jsonl.split(",")]
    roots = [item.strip() for item in image_root.split(",")] if image_root else [None] * len(local_sets)
    if len(roots) == 1 and len(local_sets) > 1:
        roots = roots * len(local_sets)
    if len(roots) != len(local_sets):
        raise ValueError("--image-root must be one root or one root per eval set")
    for index, (local_eval, local_root) in enumerate(zip(local_sets, roots, strict=True)):
        local_path = Path(local_eval)
        set_dir = remote_dir if index == 0 else f"{remote_dir}/{local_path.parent.name}-{local_path.stem}"
        remote_set, remote_inventory = upload_eval_jsonl(
            local_path,
            set_dir,
            image_root=Path(local_root) if local_root else None,
            token_inventory=Path(token_inventory) if token_inventory else None,
        )
        remote_sets.append(remote_set)
        remote_token_inventory = remote_token_inventory or remote_inventory
    kwargs = {
        "eval_jsonl": ",".join(remote_sets),
        "output_dir": output_dir,
        "adapter_dir": adapter_dir,
        "model_id": model_id,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "bits": bits,
        "token_inventory": remote_token_inventory,
        "batch_size": batch_size,
        "image_variant": image_variant,
        "occlusion_margin": occlusion_margin,
        "candidate_scoring": candidate_scoring,
    }
    remote_function = eval_h200 if gpu == "h200" else eval_remote
    if spawn_eval:
        call = remote_function.spawn(**kwargs)
        print(
            json.dumps(
                {
                    "status": "spawned",
                    "function_call_id": call.object_id,
                    "output_dir": output_dir,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = remote_function.remote(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
