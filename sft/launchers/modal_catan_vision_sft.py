"""Thin Modal launcher in ``sft.launchers`` for the native TRL/PEFT Catan vision trainer.

This file owns only infrastructure: the pinned container, Volumes, dataset
upload, immutable launch manifest, and the paid H200 boundary. Model and
training behavior live in ``sft.scripts.train.train_trl_catan_vision``.
"""

from __future__ import annotations

import json
import time as time
from dataclasses import asdict
from pathlib import Path

from sft.json_types import JsonLikeDict
from sft.launchers.catan_vision._budget import (
    training_budget_plan as training_budget_plan,
)
from sft.launchers.catan_vision._budget import (
    wait_for_budgeted_call as wait_for_budgeted_call,
)
from sft.launchers.catan_vision._bundle import (
    upload_training_bundle as upload_training_bundle,
)
from sft.launchers.catan_vision._config import (
    APP_NAME as APP_NAME,
)
from sft.launchers.catan_vision._config import (
    BUDGET_EVAL_RESERVE_USD as BUDGET_EVAL_RESERVE_USD,
)
from sft.launchers.catan_vision._config import (
    BUDGET_FUNCTION_OPTIONS as BUDGET_FUNCTION_OPTIONS,
)
from sft.launchers.catan_vision._config import (
    BUDGET_RATE_PER_SECOND as BUDGET_RATE_PER_SECOND,
)
from sft.launchers.catan_vision._config import (
    BUDGET_STARTUP_SECONDS as BUDGET_STARTUP_SECONDS,
)
from sft.launchers.catan_vision._config import (
    BUDGET_TIMEOUT_SECONDS as BUDGET_TIMEOUT_SECONDS,
)
from sft.launchers.catan_vision._config import (
    HF_SECRET_NAME as HF_SECRET_NAME,
)
from sft.launchers.catan_vision._config import (
    REMOTE_DATA as REMOTE_DATA,
)
from sft.launchers.catan_vision._config import (
    REMOTE_RUNS as REMOTE_RUNS,
)
from sft.launchers.catan_vision._config import (
    app as app,
)
from sft.launchers.catan_vision._config import (
    hf_cache as hf_cache,
)
from sft.launchers.catan_vision._config import (
    sft_data as sft_data,
)
from sft.launchers.catan_vision._config import (
    sft_runs as sft_runs,
)
from sft.launchers.catan_vision._config import (
    training_base_image as training_base_image,
)
from sft.launchers.catan_vision._config import (
    training_image as training_image,
)
from sft.launchers.catan_vision._execute import execute_training_run
from sft.launchers.catan_vision._receipts import (
    _canonical_hash,
    _dependency_versions,
    _write_json_atomic,
)
from sft.launchers.catan_vision._remote import (
    guard_training_budget as guard_training_budget,
)
from sft.launchers.catan_vision._remote import (
    train_h200 as train_h200,
)
from sft.launchers.catan_vision._remote import (
    train_h200_budgeted as train_h200_budgeted,
)
from sft.scripts.train.train_trl_catan_vision import (
    HUB_MODEL_ID,
    MODEL_ID,
    PROFILE_VISION_TOKENS_LORA,
    PROFILES,
    TrainConfig,
    inspect_jsonl_contract,
    load_token_inventory,
)


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
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-4,
    language_lora_learning_rate: float = 1e-4,
    vision_learning_rate: float = 5e-6,
    merger_learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
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
    token_init: str = "mean_noise",
    frozen_bundle: str | None = None,
    visual_delta_factors: str | None = None,
    orthogonal_lambda: float = 0.5,
    vision_lora_learning_rate: float = 1e-4,
    require_curriculum: bool = True,
    publish_to_hub: bool = False,
    resume_latest: bool = False,
    spawn_training: bool = False,
    budget_usd: float = 0.0,
    receipt_path: str | None = None,
    dry_run: bool = True,
) -> None:
    """Print the exact plan by default; ``--no-dry-run`` allocates the H200."""

    budget = training_budget_plan(budget_usd)
    receipt = Path(receipt_path).expanduser().resolve() if receipt_path else None
    if receipt is not None and receipt.exists():
        raise FileExistsError(f"refusing to overwrite launch receipt: {receipt}")
    if budget is not None and (resume_latest or publish_to_hub):
        raise ValueError("budgeted launches do not resume or publish; assess spending before a separate run")
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
    if train_jsonl and image_root and token_inventory:
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
    factors_local = Path(visual_delta_factors).expanduser().resolve() if visual_delta_factors else None
    if factors_local is not None and not factors_local.is_file():
        raise FileNotFoundError(factors_local)
    remote_factors = f"{REMOTE_DATA}/{remote_dir}/visual_delta_factors.safetensors" if factors_local is not None else None
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
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        language_lora_learning_rate=language_lora_learning_rate,
        vision_learning_rate=vision_learning_rate,
        merger_learning_rate=merger_learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
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
        token_init=token_init,
        frozen_bundle=frozen_bundle,
        visual_delta_factors=remote_factors,
        orthogonal_lambda=orthogonal_lambda,
        vision_lora_learning_rate=vision_lora_learning_rate,
    )
    config.validate()
    launch: JsonLikeDict = {
        "schema": "catan_modal_trl_vision_sft_launch/v1",
        "hardware": "H200",
        "dependencies": _dependency_versions(),
        "config": asdict(config),
        "dataset": {"train": local_contract, "eval": local_eval_contract},
    }
    if budget is not None:
        launch["budget"] = budget
    plan: JsonLikeDict = {
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
    if receipt is not None:
        _write_json_atomic(receipt, plan)
    if dry_run:
        return
    if not (train_jsonl and image_root and token_inventory):
        raise ValueError("a paid launch requires train_jsonl, image_root, and token_inventory")
    execute_training_run(
        config=config,
        launch=launch,
        plan=plan,
        budget=budget,
        train_jsonl=train_jsonl,
        image_root=image_root,
        token_inventory=token_inventory,
        eval_jsonl=eval_jsonl,
        eval_image_root=eval_image_root,
        remote_dir=remote_dir,
        remote_train=remote_train,
        remote_eval=remote_eval,
        remote_images=remote_images,
        remote_tokens=remote_tokens,
        factors_local=factors_local,
        output_dir=output_dir,
        local_contract=local_contract,
        local_eval_contract=local_eval_contract,
        require_curriculum=require_curriculum,
        resume_latest=resume_latest,
        spawn_training=spawn_training,
        receipt=receipt,
    )
