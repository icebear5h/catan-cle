"""LEGACY Modal launcher for BF16 Qwen vision-tower and merger SFT.

The active path is ``sft/launchers/modal_catan_vision_sft.py``. This launcher was
superseded before a production run and is retained for historical review.
It is intentionally separate from ``modal_qwen_series_train.py``.
The older entrypoint remains a small 4-bit frozen-vision infrastructure smoke;
this one trains the full visual path and therefore admits only 16-bit profiles.
"""

from __future__ import annotations

import json
import shlex
import subprocess as subprocess
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import modal

from sft.json_types import JsonDict, JsonLikeDict, as_str
from sft.launchers.qwen_series._train_support import (
    QWEN_COMMIT,
    REMOTE_CACHE_MOUNT,
    REMOTE_DATA_MOUNT,
    REMOTE_RUNS_MOUNT,
    hf_cache,
    qwen_series_image,
    sft_data,
    sft_runs,
    upload_qwen_series_json,
)
from sft.qwen_series_vision_sft import (
    DEFAULT_27B_MODEL_ID,
    H200_PROFILE,
    L40S_PROFILE,
    VISION_LANGUAGE_LORA,
    VisionSftConfig,
    build_vision_sft_command,
    default_run_name,
    fingerprint_training_dataset,
    validate_hardware_profile,
)

APP_NAME = "catan-qwen-series-vision-sft"
LAUNCH_MANIFEST = "launch_manifest.json"
TRAINABLE_MANIFEST = "trainable_parameters.json"
HF_SECRET_NAME = "catan-hf"

app = modal.App(APP_NAME)

hf_secret = modal.Secret.from_name(
    HF_SECRET_NAME,
    required_keys=["HF_TOKEN"],
)

_REMOTE_VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    REMOTE_CACHE_MOUNT: hf_cache,
    REMOTE_DATA_MOUNT: sft_data,
    REMOTE_RUNS_MOUNT: sft_runs,
}

from sft.launchers.qwen_series._vision_support import (  # noqa: E402
    _identity_dataset,
    _run_remote,
    _upload_token_inventory,
)


@app.function(
    image=qwen_series_image,
    gpu="L40S",
    cpu=8.0,
    memory=32 * 1024,
    secrets=[hf_secret],
    volumes=_REMOTE_VOLUMES,
    timeout=60 * 60 * 12,
)
def train_l40s_remote(
    train_json: str,
    image_folder: str,
    output_dir: str,
    token_inventory: str,
    config_payload: JsonDict,
    dataset: JsonLikeDict,
) -> JsonLikeDict:
    """Run the 4B architecture smoke on one L40S."""

    return _run_remote(
        hardware=L40S_PROFILE,
        train_json=train_json,
        image_folder=image_folder,
        output_dir=output_dir,
        token_inventory=token_inventory,
        config_payload=config_payload,
        dataset=dataset,
    )


@app.function(
    image=qwen_series_image,
    gpu="H200",
    cpu=16.0,
    memory=128 * 1024,
    secrets=[hf_secret],
    volumes=_REMOTE_VOLUMES,
    timeout=60 * 60 * 12,
)
def train_h200_remote(
    train_json: str,
    image_folder: str,
    output_dir: str,
    token_inventory: str,
    config_payload: JsonDict,
    dataset: JsonLikeDict,
) -> JsonLikeDict:
    """Run the Qwen3.8-27B native BF16 pilot on one H200."""

    return _run_remote(
        hardware=H200_PROFILE,
        train_json=train_json,
        image_folder=image_folder,
        output_dir=output_dir,
        token_inventory=token_inventory,
        config_payload=config_payload,
        dataset=dataset,
    )


@app.local_entrypoint()
def main(
    train_jsonl: str | None = None,
    image_root: str | None = None,
    token_inventory: str | None = None,
    profile: str = VISION_LANGUAGE_LORA,
    hardware: str = H200_PROFILE,
    model_id: str = DEFAULT_27B_MODEL_ID,
    run_name: str | None = None,
    remote_dir: str | None = None,
    output_dir: str | None = None,
    max_steps: int = 1,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    vision_lr: float = 1e-6,
    merger_lr: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    image_min_pixels: int = 256 * 256,
    image_max_pixels: int = 1024 * 1024,
    max_seq_length: int = 4096,
    save_steps: int = 1,
    save_total_limit: int = 3,
    dataloader_num_workers: int = 2,
    seed: int = 42,
    dry_run: bool = True,
) -> None:
    """Plan or launch one immutable vision-SFT run.

    Pass ``--max-steps 0`` to omit the step cap and train for
    ``--num-train-epochs``. A paid GPU launch requires ``--no-dry-run``.
    """

    config = VisionSftConfig(
        profile=profile,
        model_id=model_id,
        max_steps=None if max_steps == 0 else max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        vision_lr=vision_lr,
        merger_lr=merger_lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        max_seq_length=max_seq_length,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
    )
    validate_hardware_profile(config.model_id, hardware)
    run_name = run_name or default_run_name(config)
    output_dir = output_dir or f"{REMOTE_RUNS_MOUNT}/qwen-vision-sft/{run_name}"

    if train_jsonl is None:
        if not dry_run:
            raise ValueError("--train-jsonl is required for a GPU launch")
        dataset: JsonLikeDict = {
            "source_sha256": "DRY_RUN",
            "combined_sha256": "DRY_RUN",
            "rows": 0,
            "unique_images": 0,
            "max_prompt_characters": 0,
            "max_answer_characters": 0,
            "annotation_root": "DRY_RUN",
            "image_root": "DRY_RUN",
            "token_inventory": None,
        }
    else:
        if image_root is None or token_inventory is None:
            raise ValueError("--image-root and --token-inventory are required with --train-jsonl")
        dataset = fingerprint_training_dataset(
            Path(train_jsonl),
            image_root=Path(image_root),
            token_inventory=Path(token_inventory),
        )
        if config.max_steps is None and config.num_train_epochs != 1.0:
            raise ValueError(
                "replay_v1 exports one immutable query plan per epoch; "
                "use exactly one epoch or provide an explicit epoch-specific shard"
            )

    remote_dir = remote_dir or (
        f"catan-qwen-vision-sft/{run_name}/{as_str(dataset['combined_sha256'])[:12]}"
    )
    remote_token_inventory = f"{REMOTE_DATA_MOUNT}/{remote_dir.strip('/')}/trainable_tokens.json"
    plan_command = build_vision_sft_command(
        config,
        train_json=f"{REMOTE_DATA_MOUNT}/{remote_dir.strip('/')}/train_qwen_series.json",
        image_folder=f"{REMOTE_DATA_MOUNT}/{remote_dir.strip('/')}/images",
        output_dir=output_dir,
        token_inventory=remote_token_inventory,
    )
    plan: JsonLikeDict = {
        "schema": "catan_qwen_vision_sft_plan/v2",
        "config": config.as_manifest_dict(),
        "dataset": _identity_dataset(dataset),
        "hardware": hardware,
        "qwen_repo_commit": QWEN_COMMIT,
        "run_name": run_name,
        "remote_dir": remote_dir,
        "output_dir": output_dir,
        "dry_run": dry_run,
        "command": plan_command,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    print("command:", shlex.join(plan_command))
    if dry_run:
        return

    if train_jsonl is None or image_root is None or token_inventory is None:
        raise ValueError(
            "--train-jsonl, --image-root, and --token-inventory are required for launch"
        )
    remote_train_json, remote_image_folder = upload_qwen_series_json(
        Path(train_jsonl),
        remote_dir,
        image_root=Path(image_root),
    )
    uploaded_token_inventory = _upload_token_inventory(
        Path(token_inventory),
        remote_dir,
    )
    if uploaded_token_inventory != remote_token_inventory:
        raise RuntimeError("uploaded token inventory path disagrees with launch plan")
    remote_function = train_h200_remote if hardware == H200_PROFILE else train_l40s_remote
    result = remote_function.remote(
        train_json=remote_train_json,
        image_folder=remote_image_folder,
        output_dir=output_dir,
        token_inventory=uploaded_token_inventory,
        config_payload=asdict(config),
        dataset=_identity_dataset(dataset),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
