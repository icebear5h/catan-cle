"""Thin Modal launcher for the native TRL/PEFT Catan vision trainer.

This file owns only infrastructure: the pinned container, Volumes, dataset
upload, immutable launch manifest, and the paid H200 boundary. Model and
training behavior live in ``sft.scripts.train_trl_catan_vision``.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from sft.scripts.train_trl_catan_vision import (
    ACCELERATE_VERSION,
    DATASETS_VERSION,
    HUB_MODEL_ID,
    HUGGINGFACE_HUB_VERSION,
    MODEL_ID,
    PEFT_VERSION,
    PILLOW_VERSION,
    PROFILE_VISION_TOKENS_LORA,
    PROFILES,
    SAFETENSORS_VERSION,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    TRANSFORMERS_VERSION,
    TRL_VERSION,
    TrainConfig,
    inspect_jsonl_contract,
    iter_jsonl,
    load_token_inventory,
    resolve_image_path,
    run_training,
)


APP_NAME = "catan-qwen3-8-vision-sft"
HF_SECRET_NAME = "catan-hf"
REMOTE_CACHE = "/cache"
REMOTE_DATA = "/data"
REMOTE_RUNS = "/runs"
LAUNCH_MANIFEST = "modal_launch.json"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("catan-hf-cache", create_if_missing=True)
sft_data = modal.Volume.from_name("catan-sft-data", create_if_missing=True)
sft_runs = modal.Volume.from_name("catan-sft-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"trl=={TRL_VERSION}",
        f"peft=={PEFT_VERSION}",
        f"datasets=={DATASETS_VERSION}",
        f"accelerate=={ACCELERATE_VERSION}",
        f"huggingface-hub=={HUGGINGFACE_HUB_VERSION}",
        f"safetensors=={SAFETENSORS_VERSION}",
        f"Pillow=={PILLOW_VERSION}",
    )
    .env(
        {
            "HF_HOME": f"{REMOTE_CACHE}/huggingface",
            "HF_HUB_CACHE": f"{REMOTE_CACHE}/huggingface/hub",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Stream trainer prints immediately; otherwise the container is
            # silent until the first tqdm bar on stderr.
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("cle")
    .add_local_python_source("evals")
    .add_local_python_source("sft")
)

_VOLUMES = {
    REMOTE_CACHE: hf_cache,
    REMOTE_DATA: sft_data,
    REMOTE_RUNS: sft_runs,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _dependency_versions() -> dict[str, str]:
    return {
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "trl": TRL_VERSION,
        "peft": PEFT_VERSION,
        "datasets": DATASETS_VERSION,
        "accelerate": ACCELERATE_VERSION,
        "huggingface_hub": HUGGINGFACE_HUB_VERSION,
        "safetensors": SAFETENSORS_VERSION,
        "pillow": PILLOW_VERSION,
    }


def upload_training_bundle(
    train_jsonl: Path,
    image_root: Path,
    token_inventory: Path,
    *,
    remote_dir: str,
    require_curriculum: bool,
    eval_jsonl: Path | None = None,
    eval_image_root: Path | None = None,
) -> tuple[str, str | None, str, str, dict[str, Any], dict[str, Any] | None]:
    """Upload one ordered JSONL and its deduplicated images without reordering."""

    source = train_jsonl.expanduser().resolve()
    root = image_root.expanduser().resolve()
    inventory_path = token_inventory.expanduser().resolve()
    contract = inspect_jsonl_contract(source, root, require_curriculum=require_curriculum)
    eval_source = eval_jsonl.expanduser().resolve() if eval_jsonl is not None else None
    eval_root = eval_image_root.expanduser().resolve() if eval_image_root is not None else None
    if bool(eval_source) != bool(eval_root):
        raise ValueError("eval_jsonl and eval_image_root must be supplied together")
    eval_contract = (
        inspect_jsonl_contract(eval_source, eval_root, require_curriculum=False)
        if eval_source is not None and eval_root is not None
        else None
    )
    load_token_inventory(inventory_path)

    remote_root = "/" + remote_dir.strip("/")
    remote_jsonl = f"{remote_root}/train.jsonl"
    remote_eval_jsonl = f"{remote_root}/eval.jsonl" if eval_source is not None else None
    remote_images = f"{remote_root}/images"
    remote_inventory = f"{remote_root}/trainable_tokens.json"
    image_names: dict[Path, str] = {}

    with tempfile.TemporaryDirectory(prefix="catan-modal-upload-") as temporary_dir:
        rewritten_path = Path(temporary_dir) / "train.jsonl"
        rewritten_eval_path = Path(temporary_dir) / "eval.jsonl"

        def rewrite_jsonl(input_path: Path, input_root: Path, output_path: Path) -> None:
            with output_path.open("w") as output:
                for line_number, row in iter_jsonl(input_path):
                    image_path = resolve_image_path(input_root, row, line_number=line_number)
                    image_name = image_names.get(image_path)
                    if image_name is None:
                        suffix = image_path.suffix.lower() or ".png"
                        image_name = f"{len(image_names):06d}{suffix}"
                        image_names[image_path] = image_name
                    rewritten = dict(row)
                    rewritten.pop("image", None)
                    rewritten["images"] = [image_name]
                    output.write(json.dumps(rewritten, sort_keys=True) + "\n")

        rewrite_jsonl(source, root, rewritten_path)
        if eval_source is not None and eval_root is not None:
            rewrite_jsonl(eval_source, eval_root, rewritten_eval_path)

        with sft_data.batch_upload(force=True) as batch:
            for image_path, image_name in image_names.items():
                batch.put_file(image_path, f"{remote_images}/{image_name}")
            batch.put_file(rewritten_path, remote_jsonl)
            if remote_eval_jsonl is not None:
                batch.put_file(rewritten_eval_path, remote_eval_jsonl)
            batch.put_file(inventory_path, remote_inventory)

    return (
        f"{REMOTE_DATA}{remote_jsonl}",
        f"{REMOTE_DATA}{remote_eval_jsonl}" if remote_eval_jsonl is not None else None,
        f"{REMOTE_DATA}{remote_images}",
        f"{REMOTE_DATA}{remote_inventory}",
        contract,
        eval_contract,
    )


def _required_artifacts(output_dir: Path) -> dict[str, dict[str, Any]]:
    final_dir = output_dir / "final"
    candidates = {
        "adapter_config": final_dir / "adapter_config.json",
        "adapter_model": final_dir / "adapter_model.safetensors",
        "visual_model": final_dir / "visual_model.safetensors",
        "tokenizer_config": final_dir / "tokenizer_config.json",
        "trainable_parameters": final_dir / "trainable_parameters.json",
        "reload_validation": final_dir / "reload_validation.json",
    }
    missing = [name for name, path in candidates.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"training completed without artifacts: {', '.join(missing)}")
    return {
        name: {"path": str(path), "bytes": path.stat().st_size}
        for name, path in candidates.items()
    }


def _latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the highest numbered complete-looking Trainer checkpoint."""

    candidates: list[tuple[int, Path]] = []
    if not checkpoints_dir.is_dir():
        return None
    for path in checkpoints_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        if all((path / name).is_file() for name in required):
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


@app.function(
    image=training_image,
    gpu="H200",
    cpu=16.0,
    memory=128 * 1024,
    secrets=[hf_secret],
    volumes=_VOLUMES,
    timeout=60 * 60 * 24,
)
def train_h200(
    config_payload: dict[str, Any],
    launch_payload: dict[str, Any],
    resume_latest: bool = False,
) -> dict[str, Any]:
    """Cross the paid boundary and call the native trainer directly."""

    config = TrainConfig(**config_payload)
    output_dir = Path(config.output_dir)
    manifest_path = output_dir / LAUNCH_MANIFEST
    identity = _canonical_hash(launch_payload)
    existing = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if existing is not None and existing.get("identity") != identity:
        raise RuntimeError(f"refusing to reuse {output_dir} for a different launch")
    if existing is not None and existing.get("status") == "completed":
        return {
            "status": "already_completed",
            "identity": identity,
            "artifacts": _required_artifacts(output_dir),
        }
    if existing is None and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"non-empty output directory has no {LAUNCH_MANIFEST}: {output_dir}")

    resumed_from_checkpoint = config.resume_from_checkpoint
    if resume_latest:
        if existing is None:
            raise RuntimeError("--resume-latest requires an existing interrupted launch")
        latest = _latest_checkpoint(output_dir / "checkpoints")
        if latest is None:
            raise RuntimeError("--resume-latest found no complete Trainer checkpoint")
        resumed_from_checkpoint = str(latest)
        config = replace(config, resume_from_checkpoint=resumed_from_checkpoint)
    elif existing is not None:
        raise RuntimeError(
            "an interrupted launch already exists; pass --resume-latest to resume it safely"
        )

    manifest = {
        **launch_payload,
        "identity": identity,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "attempt": int((existing or {}).get("attempt", 0)) + 1,
        "resumed_from_checkpoint": resumed_from_checkpoint,
    }
    _write_json_atomic(manifest_path, manifest)
    try:
        result = run_training(config)
        artifacts = _required_artifacts(output_dir)
    except Exception as exc:
        manifest.update(status="failed", completed_at=_utc_now(), error=repr(exc))
        _write_json_atomic(manifest_path, manifest)
        raise
    else:
        manifest.update(status="completed", completed_at=_utc_now(), artifacts=artifacts)
        _write_json_atomic(manifest_path, manifest)
        return {**result, "identity": identity, "artifacts": artifacts}
    finally:
        hf_cache.commit()
        sft_runs.commit()


@app.local_entrypoint()
def main(
    train_jsonl: str | None = None,
    image_root: str | None = None,
    token_inventory: str | None = None,
    eval_jsonl: str | None = None,
    eval_image_root: str | None = None,
    run_name: str = "catan-qwen3-8-27b-spatial-sft",
    model_id: str = MODEL_ID,
    profile: str = PROFILE_VISION_TOKENS_LORA,
    hub_model_id: str = HUB_MODEL_ID,
    initial_bundle: str | None = None,
    max_steps: int = 0,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-4,
    language_lora_learning_rate: float = 1e-4,
    vision_learning_rate: float = 5e-6,
    merger_learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    image_min_pixels: int = 256 * 256,
    image_max_pixels: int = 1024 * 1024,
    save_steps: int = 256,
    eval_steps: int = 128,
    save_total_limit: int = 2,
    dataloader_num_workers: int = 2,
    seed: int = 42,
    patch_loss_weight: float = 0.0,
    patch_temperature: float = 0.07,
    spatial_target_mode: str = "correct",
    require_curriculum: bool = True,
    publish_to_hub: bool = False,
    resume_latest: bool = False,
    spawn_training: bool = False,
    dry_run: bool = True,
) -> None:
    """Print the exact plan by default; ``--no-dry-run`` allocates the H200."""

    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    supplied = (train_jsonl, image_root, token_inventory)
    if any(supplied) and not all(supplied):
        raise ValueError("train_jsonl, image_root, and token_inventory must be supplied together")
    if bool(eval_jsonl) != bool(eval_image_root):
        raise ValueError("eval_jsonl and eval_image_root must be supplied together")
    if not dry_run and not all(supplied):
        raise ValueError("a paid launch requires train_jsonl, image_root, and token_inventory")

    local_contract = None
    local_eval_contract = None
    if all(supplied):
        local_contract = inspect_jsonl_contract(
            Path(train_jsonl),
            Path(image_root),
            require_curriculum=require_curriculum,
        )
        load_token_inventory(Path(token_inventory))
        if eval_jsonl is not None and eval_image_root is not None:
            local_eval_contract = inspect_jsonl_contract(
                Path(eval_jsonl),
                Path(eval_image_root),
                require_curriculum=False,
            )
    source_identity_payload = {
        "train": (local_contract or {}).get("source_sha256", "NO_DATA"),
        "eval": (local_eval_contract or {}).get("source_sha256"),
    }
    source_identity = _canonical_hash(source_identity_payload)[:12]
    # The A/B/C arms share byte-identical data. Key uploads by data identity so
    # unique run names do not consume three copies of the 1024px image bundle.
    remote_dir = f"catan-vision-sft/datasets/{source_identity}"
    remote_train = f"{REMOTE_DATA}/{remote_dir}/train.jsonl"
    remote_eval = f"{REMOTE_DATA}/{remote_dir}/eval.jsonl" if eval_jsonl else None
    remote_images = f"{REMOTE_DATA}/{remote_dir}/images"
    remote_tokens = f"{REMOTE_DATA}/{remote_dir}/trainable_tokens.json"
    output_dir = f"{REMOTE_RUNS}/catan-vision-sft/{run_name}/{source_identity}"

    config = TrainConfig(
        train_jsonl=remote_train,
        image_root=remote_images,
        token_inventory=remote_tokens,
        output_dir=output_dir,
        eval_jsonl=remote_eval,
        eval_image_root=remote_images if remote_eval else None,
        model_id=model_id,
        profile=profile,
        hub_model_id=hub_model_id,
        publish_to_hub=publish_to_hub,
        require_curriculum=require_curriculum,
        initial_bundle=initial_bundle,
        max_steps=None if max_steps == 0 else max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        language_lora_learning_rate=language_lora_learning_rate,
        vision_learning_rate=vision_learning_rate,
        merger_learning_rate=merger_learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        patch_loss_weight=patch_loss_weight,
        patch_temperature=patch_temperature,
        spatial_target_mode=spatial_target_mode,
    )
    config.validate()
    launch = {
        "schema": "catan_modal_trl_vision_sft_launch/v1",
        "hardware": "H200",
        "dependencies": _dependency_versions(),
        "config": asdict(config),
        "dataset": {"train": local_contract, "eval": local_eval_contract},
    }
    plan = {
        **launch,
        "identity": _canonical_hash(launch),
        "dry_run": dry_run,
        "paid_gpu_requested": not dry_run,
        "modal_app": APP_NAME,
        "modal_secret": HF_SECRET_NAME,
        "resume_latest": resume_latest,
        "spawn_training": spawn_training,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if dry_run:
        return

    (
        uploaded_train,
        uploaded_eval,
        uploaded_images,
        uploaded_tokens,
        uploaded_contract,
        uploaded_eval_contract,
    ) = upload_training_bundle(
        Path(train_jsonl),
        Path(image_root),
        Path(token_inventory),
        remote_dir=remote_dir,
        require_curriculum=require_curriculum,
        eval_jsonl=Path(eval_jsonl) if eval_jsonl is not None else None,
        eval_image_root=Path(eval_image_root) if eval_image_root is not None else None,
    )
    if (uploaded_train, uploaded_eval, uploaded_images, uploaded_tokens) != (
        remote_train,
        remote_eval,
        remote_images,
        remote_tokens,
    ):
        raise RuntimeError("uploaded paths disagree with the launch plan")
    if uploaded_contract != local_contract:
        raise RuntimeError("dataset changed between planning and upload")
    if uploaded_eval_contract != local_eval_contract:
        raise RuntimeError("evaluation dataset changed between planning and upload")
    if spawn_training:
        call = train_h200.spawn(asdict(config), launch, resume_latest)
        print(
            json.dumps(
                {
                    "status": "spawned",
                    "function_call_id": call.object_id,
                    "identity": plan["identity"],
                    "output_dir": output_dir,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = train_h200.remote(asdict(config), launch, resume_latest)
    print(json.dumps(result, indent=2, sort_keys=True))
