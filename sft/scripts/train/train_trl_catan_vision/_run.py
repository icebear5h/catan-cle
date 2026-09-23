"""Native Hugging Face TRL/PEFT training for Catan board grounding.

This module owns the complete model-side contract: dataset normalization,
curriculum-order validation, semantic token insertion, PEFT wrapping, optimizer
groups, checkpoint serialization, reload validation, and optional Hub upload.
It intentionally has no Modal or ms-swift dependency.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import torch

from sft.json_types import json_dict
from sft.scripts.train.train_trl_catan_vision._bundles import (
    load_frozen_bundle,
    load_initial_bundle,
    wrap_trainable_model,
)
from sft.scripts.train.train_trl_catan_vision._common import (
    DATASET_REPORT_FILE,
    FROZEN_ADAPTER_DIR,
    FROZEN_BUNDLE_FILE,
    HUB_MODEL_ID,
    INITIAL_BUNDLE_FILE,
    INPUT_MODES,
    MODEL_ID,
    PROFILE_VISION_TOKENS_LORA,
    PROFILES,
    RUN_CONFIG_FILE,
    SPATIAL_TARGET_MODES,
    TOKEN_INIT_MODES,
    TRAINABLE_SCOPE_FILE,
    JsonDict,
    inventory_tokens,
    load_token_inventory,
    write_json_atomic,
)
from sft.scripts.train.train_trl_catan_vision._config import TrainConfig
from sft.scripts.train.train_trl_catan_vision._datasets import (
    load_text_dataset,
    load_training_dataset,
    load_vision_dataset,
)
from sft.scripts.train.train_trl_catan_vision._frozen import load_checkpoint_text_tokenizer
from sft.scripts.train.train_trl_catan_vision._model_tokens import prepare_semantic_tokens
from sft.scripts.train.train_trl_catan_vision._optim import audit_trainable_scope
from sft.scripts.train.train_trl_catan_vision._sft import (
    _load_base_model_and_processor,
    _write_model_card,
    assert_runtime_versions,
    build_sft_config,
    publish_bundle,
    validate_saved_bundle,
)
from sft.scripts.train.train_trl_catan_vision._structure import required_text
from sft.scripts.train.train_trl_catan_vision._trainer import _trainer_class


def run_training(config: TrainConfig, *, extra_callbacks: Sequence[object] | None = None) -> JsonDict:
    config.validate()
    versions = assert_runtime_versions()
    inventory = load_token_inventory(config.token_inventory)
    text_tokenizer = (
        load_checkpoint_text_tokenizer(
            required_text(config.initial_bundle, "initial_bundle"), inventory_tokens(inventory),
        )
        if config.text_only else None
    )
    dataset, dataset_report = load_training_dataset(config, tokenizer=text_tokenizer)
    eval_dataset = None
    eval_dataset_report = None
    if config.text_only and config.eval_jsonl is not None:
        eval_dataset, eval_dataset_report = load_text_dataset(
            config.eval_jsonl, tokenizer=text_tokenizer,
            max_sequence_length=config.max_sequence_length, require_curriculum=False,
        )
    elif config.eval_jsonl is not None and config.eval_image_root is not None:
        eval_dataset, eval_dataset_report = load_vision_dataset(
            config.eval_jsonl,
            config.eval_image_root,
            require_curriculum=False,
        )
    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    final_dir = output_dir / "final"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / DATASET_REPORT_FILE,
        {"train": dataset_report, "eval": eval_dataset_report},
    )
    write_json_atomic(output_dir / RUN_CONFIG_FILE, asdict(config))

    processor, base_model = _load_base_model_and_processor(config, processor=text_tokenizer)
    setup, components = prepare_semantic_tokens(processor, base_model, inventory)
    initial_bundle_report = None
    frozen_bundle_report = None
    if config.olora:
        model, frozen_bundle_report = load_frozen_bundle(base_model, setup, components, config)
        write_json_atomic(output_dir / FROZEN_BUNDLE_FILE, frozen_bundle_report)
    elif config.initial_bundle:
        model, initial_bundle_report = load_initial_bundle(
            base_model,
            setup,
            components,
            config,
        )
        write_json_atomic(output_dir / INITIAL_BUNDLE_FILE, initial_bundle_report)
    else:
        model = wrap_trainable_model(base_model, setup, components, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    audit_trainable_scope(
        model,
        components,
        setup,
        config,
        output_dir / TRAINABLE_SCOPE_FILE,
    )

    args = build_sft_config(
        config,
        checkpoints_dir,
        has_eval_dataset=eval_dataset is not None,
    )
    trainer_type = _trainer_class(config, components, setup)
    trainer = trainer_type(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )
    # Append after constructor-installed callbacks and the existing bundle save machinery.
    for callback in extra_callbacks or ():
        trainer.add_callback(callback)
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    # The periodic eval already ran on the last step whenever max_steps is a
    # multiple of eval_steps; only evaluate again when it did not.
    final_eval_metrics = None
    if eval_dataset is not None:
        last_eval = next(
            (row for row in reversed(trainer.state.log_history) if "eval_loss" in row),
            None,
        )
        if last_eval is not None and last_eval.get("step") == trainer.state.global_step:
            final_eval_metrics = {key: value for key, value in last_eval.items() if key != "step"}
        else:
            final_eval_metrics = trainer.evaluate()
    trainer.save_model(str(final_dir))
    trainer.save_state()
    _write_model_card(final_dir, config)
    metrics = dict(train_result.metrics)
    write_json_atomic(
        final_dir / "train_metrics.json",
        {"train": metrics, "eval": final_eval_metrics},
    )
    if initial_bundle_report is not None:
        write_json_atomic(final_dir / INITIAL_BUNDLE_FILE, initial_bundle_report)
    if frozen_bundle_report is not None:
        write_json_atomic(final_dir / FROZEN_BUNDLE_FILE, {**frozen_bundle_report, "carried_adapter": str(final_dir / FROZEN_ADAPTER_DIR)})

    del trainer, model, base_model, processor, dataset, eval_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reload_report = validate_saved_bundle(config, final_dir, inventory, setup)
    publish_report = publish_bundle(config, final_dir) if config.publish_to_hub else None
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "final_dir": str(final_dir),
        "versions": versions,
        "dataset": {"train": dataset_report, "eval": eval_dataset_report},
        "metrics": json_dict(metrics),
        "eval_metrics": json_dict(final_eval_metrics) if final_eval_metrics is not None else None,
        "initial_bundle": initial_bundle_report,
        "frozen_bundle": frozen_bundle_report,
        "reload_validation": reload_report,
        "hub_publish": publish_report,
    }


def parse_args(argv: Iterable[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="vision")
    parser.add_argument("--max-sequence-length", type=int, help="Required text token budget; never truncates")
    parser.add_argument("--token-inventory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-jsonl")
    parser.add_argument("--eval-image-root")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE_VISION_TOKENS_LORA)
    parser.add_argument("--hub-model-id", default=HUB_MODEL_ID)
    parser.add_argument("--publish-to-hub", action="store_true")
    parser.add_argument("--allow-unstaged", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--initial-bundle")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--language-lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vision-learning-rate", type=float, default=5e-6)
    parser.add_argument("--merger-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--image-min-pixels", type=int, default=256 * 256)
    parser.add_argument("--image-max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--save-steps", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-loss-weight", type=float, default=0.0)
    parser.add_argument("--patch-temperature", type=float, default=0.07)
    parser.add_argument(
        "--spatial-target-mode",
        choices=SPATIAL_TARGET_MODES,
        default="correct",
    )
    parser.add_argument("--token-init", choices=TOKEN_INIT_MODES, default="mean_noise")
    parser.add_argument("--frozen-bundle", help="Finished bundle merged into the base as the frozen task (olora profile)")
    parser.add_argument("--visual-delta-factors", help="safetensors of lora_A/lora_B factors of the frozen task's visual delta; their lora_A rows are the protected vision subspaces")
    parser.add_argument("--orthogonal-lambda", type=float, default=0.5)
    parser.add_argument("--vision-lora-learning-rate", type=float, default=1e-4)
    args = parser.parse_args(argv)
    return TrainConfig(
        train_jsonl=args.train_jsonl,
        image_root=args.image_root,
        input_mode=args.input_mode,
        max_sequence_length=args.max_sequence_length,
        token_inventory=args.token_inventory,
        output_dir=args.output_dir,
        eval_jsonl=args.eval_jsonl,
        eval_image_root=args.eval_image_root,
        model_id=args.model_id,
        profile=args.profile,
        hub_model_id=args.hub_model_id,
        publish_to_hub=args.publish_to_hub,
        require_curriculum=not args.allow_unstaged,
        resume_from_checkpoint=args.resume_from_checkpoint,
        initial_bundle=args.initial_bundle,
        max_steps=None if args.max_steps == 0 else args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        language_lora_learning_rate=args.language_lora_learning_rate,
        vision_learning_rate=args.vision_learning_rate,
        merger_learning_rate=args.merger_learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        patch_loss_weight=args.patch_loss_weight,
        patch_temperature=args.patch_temperature,
        spatial_target_mode=args.spatial_target_mode,
        token_init=args.token_init,
        frozen_bundle=args.frozen_bundle,
        visual_delta_factors=args.visual_delta_factors,
        orthogonal_lambda=args.orthogonal_lambda,
        vision_lora_learning_rate=args.vision_lora_learning_rate,
    )


def main() -> int:
    result = run_training(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
