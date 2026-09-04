"""LEGACY Modal launcher for BF16 Qwen vision-tower and merger SFT.

The active path is ``sft/modal_catan_vision_sft.py``. This launcher was
superseded before a production run and is retained for historical review.
It is intentionally separate from ``modal_qwen_series_train.py``.
The older entrypoint remains a small 4-bit frozen-vision infrastructure smoke;
this one trains the full visual path and therefore admits only 16-bit profiles.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from sft.modal_qwen_series_train import (
    QWEN_COMMIT,
    REMOTE_CACHE_MOUNT,
    REMOTE_DATA_MOUNT,
    REMOTE_QWEN_DIR,
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
    launch_identity,
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

_REMOTE_VOLUMES = {
    REMOTE_CACHE_MOUNT: hf_cache,
    REMOTE_DATA_MOUNT: sft_data,
    REMOTE_RUNS_MOUNT: sft_runs,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _upload_token_inventory(token_inventory: Path, remote_dir: str) -> str:
    local_path = token_inventory.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    remote_path = f"/{remote_dir.strip('/')}/trainable_tokens.json"
    with sft_data.batch_upload(force=True) as batch:
        batch.put_file(local_path, remote_path)
    return f"{REMOTE_DATA_MOUNT}{remote_path}"


def _artifact_evidence(output_dir: Path) -> dict[str, Any]:
    required_paths = {
        "adapter_config": output_dir / "adapter_config.json",
        "non_lora_state": output_dir / "non_lora_state_dict.bin",
        "trainable_parameters": output_dir / TRAINABLE_MANIFEST,
        "tokenizer_config": output_dir / "tokenizer_config.json",
    }
    adapter_candidates = [
        output_dir / "adapter_model.safetensors",
        output_dir / "adapter_model.bin",
    ]

    missing = [name for name, path in required_paths.items() if not path.is_file()]
    adapter_path = next((path for path in adapter_candidates if path.is_file()), None)
    if adapter_path is None:
        missing.append("adapter_model")
    if missing:
        raise RuntimeError(
            f"Training command completed without required artifacts in {output_dir}: "
            + ", ".join(sorted(missing))
        )

    paths = dict(required_paths)
    paths["adapter_model"] = adapter_path
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(paths.items())
    }


def _identity_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source_sha256",
        "combined_sha256",
        "rows",
        "unique_images",
        "max_prompt_characters",
        "max_answer_characters",
        "annotation_root",
        "image_root",
        "token_inventory",
    )
    return {key: dataset[key] for key in keys}


def _run_remote(
    *,
    hardware: str,
    train_json: str,
    image_folder: str,
    output_dir: str,
    token_inventory: str,
    config_payload: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    config = VisionSftConfig(**config_payload)
    validate_hardware_profile(config.model_id, hardware)
    command = build_vision_sft_command(
        config,
        train_json=train_json,
        image_folder=image_folder,
        output_dir=output_dir,
        token_inventory=token_inventory,
    )

    output_path = Path(output_dir)
    manifest_path = output_path / LAUNCH_MANIFEST
    identity_payload = {
        "schema": "catan_qwen_vision_sft_launch_identity/v2",
        "config": config.as_manifest_dict(),
        "dataset": _identity_dataset(dataset),
        "hardware": hardware,
        "qwen_repo_commit": QWEN_COMMIT,
    }
    identity = launch_identity(identity_payload)

    existing_manifest = None
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest.get("identity") != identity:
            raise RuntimeError(f"Refusing to reuse {output_path} for a different launch identity")
        if existing_manifest.get("status") == "completed":
            artifacts = _artifact_evidence(output_path)
            return {
                "status": "already_completed",
                "identity": identity,
                "output_dir": output_dir,
                "artifacts": artifacts,
            }
    elif output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(
            f"Refusing non-empty output directory without {LAUNCH_MANIFEST}: {output_path}"
        )

    attempts = int((existing_manifest or {}).get("attempts", 0)) + 1
    manifest = {
        **identity_payload,
        "identity": identity,
        "status": "running",
        "attempts": attempts,
        "started_at": _utc_now(),
        "completed_at": None,
        "command": command,
        "output_dir": output_dir,
    }
    _write_json_atomic(manifest_path, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("running:", shlex.join(command))
    try:
        subprocess.run(command, check=True, cwd=REMOTE_QWEN_DIR)
        artifacts = _artifact_evidence(output_path)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = _utc_now()
        manifest["error"] = repr(exc)
        _write_json_atomic(manifest_path, manifest)
        raise
    else:
        manifest["status"] = "completed"
        manifest["completed_at"] = _utc_now()
        manifest["artifacts"] = artifacts
        _write_json_atomic(manifest_path, manifest)
        return {
            "status": "completed",
            "identity": identity,
            "output_dir": output_dir,
            "artifacts": artifacts,
        }
    finally:
        hf_cache.commit()
        sft_runs.commit()


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
    config_payload: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
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
    config_payload: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
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
):
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
        dataset = {
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
        f"catan-qwen-vision-sft/{run_name}/{dataset['combined_sha256'][:12]}"
    )
    remote_token_inventory = f"{REMOTE_DATA_MOUNT}/{remote_dir.strip('/')}/trainable_tokens.json"
    plan_command = build_vision_sft_command(
        config,
        train_json=f"{REMOTE_DATA_MOUNT}/{remote_dir.strip('/')}/train_qwen_series.json",
        image_folder=f"{REMOTE_DATA_MOUNT}/{remote_dir.strip('/')}/images",
        output_dir=output_dir,
        token_inventory=remote_token_inventory,
    )
    plan = {
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
