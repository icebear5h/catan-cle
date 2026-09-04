"""Measure behavior gradient norms and conflicts at one saved Catan adapter."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sft.gradient_diagnostics import (
    PARAMETER_GROUPS,
    PROBE_BEHAVIORS,
    build_gradient_report,
    capture_gradient_snapshot,
    gradient_report_markdown,
    iter_jsonl,
    probe_parameter_group,
)
from sft.scripts.train_trl_catan_vision import (
    RUN_CONFIG_FILE,
    TRAINABLE_SCOPE_FILE,
    VISUAL_STATE_FILE,
    TrainConfig,
    _load_base_model_and_processor,
    _trainer_class,
    audit_trainable_scope,
    build_sft_config,
    load_initial_bundle,
    load_token_inventory,
    load_vision_dataset,
    parameter_category,
    prepare_semantic_tokens,
    sha256_file,
    write_json_atomic,
)


JsonDict = dict[str, Any]


def probe_config_from_bundle(
    adapter_dir: str | Path,
    *,
    probe_jsonl: str,
    image_root: str,
    token_inventory: str,
    output_dir: str,
    microbatch_size: int,
) -> tuple[TrainConfig, JsonDict]:
    bundle = Path(adapter_dir).expanduser().resolve()
    parent_path = bundle / RUN_CONFIG_FILE
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    parent = json.loads(parent_path.read_text())
    valid_fields = {field.name for field in fields(TrainConfig)}
    payload = {name: value for name, value in parent.items() if name in valid_fields}
    payload.update(
        train_jsonl=probe_jsonl,
        image_root=image_root,
        token_inventory=token_inventory,
        output_dir=output_dir,
        eval_jsonl=None,
        eval_image_root=None,
        publish_to_hub=False,
        require_curriculum=False,
        resume_from_checkpoint=None,
        initial_bundle=str(bundle),
        max_steps=1,
        num_train_epochs=1.0,
        per_device_train_batch_size=microbatch_size,
        per_device_eval_batch_size=microbatch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
    )
    config = TrainConfig(**payload)
    config.validate()
    return config, parent


def _validate_selection_rows(rows: list[JsonDict], selection: JsonDict) -> None:
    if len(rows) != int(selection["total_rows"]):
        raise ValueError("probe JSONL row count disagrees with selection manifest")
    for behavior in PROBE_BEHAVIORS:
        spec = selection["behaviors"][behavior]
        observed = [
            str(rows[index].get("row_id") or rows[index].get("id"))
            for index in spec["indices"]
        ]
        if observed != spec["row_ids"]:
            raise ValueError(f"probe JSONL ordering changed for {behavior}")


def _learning_rates(config: TrainConfig) -> dict[str, float]:
    return {
        "vision": config.vision_learning_rate,
        "merger": config.merger_learning_rate,
        "language_lora": config.language_lora_learning_rate,
        "token_rows": config.learning_rate,
    }


def run_gradient_probe(
    *,
    adapter_dir: str,
    probe_jsonl: str,
    image_root: str,
    token_inventory: str,
    selection_manifest_path: str,
    output_dir: str,
    microbatch_size: int = 4,
) -> JsonDict:
    """Run deterministic eval-mode backward passes without an optimizer step."""

    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    selection_path = Path(selection_manifest_path)
    selection = json.loads(selection_path.read_text())
    output = Path(output_dir)
    report_path = output / "gradient_conflicts.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text())
        if (
            Path(existing["adapter_dir"]).resolve() != Path(adapter_dir).resolve()
            or existing["selection"]["identity"] != selection["identity"]
        ):
            raise RuntimeError("existing gradient report has a different adapter or selection")
        return existing
    unexpected = [] if not output.exists() else [
        path.name for path in output.iterdir() if path.name != "modal_launch.json"
    ]
    if unexpected:
        raise RuntimeError(f"refusing non-empty incomplete probe output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    raw_rows = list(iter_jsonl(probe_jsonl))
    _validate_selection_rows(raw_rows, selection)
    for behavior in PROBE_BEHAVIORS:
        count = len(selection["behaviors"][behavior]["indices"])
        if count % microbatch_size:
            raise ValueError(f"{behavior} row count must be divisible by microbatch_size")

    config, parent_config = probe_config_from_bundle(
        adapter_dir,
        probe_jsonl=probe_jsonl,
        image_root=image_root,
        token_inventory=token_inventory,
        output_dir=output_dir,
        microbatch_size=microbatch_size,
    )
    inventory = load_token_inventory(token_inventory)
    dataset, dataset_report = load_vision_dataset(
        probe_jsonl,
        image_root,
        require_curriculum=False,
    )
    processor, base_model = _load_base_model_and_processor(config)
    setup, components = prepare_semantic_tokens(processor, base_model, inventory)
    model, load_report = load_initial_bundle(base_model, setup, components, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    scope = audit_trainable_scope(
        model,
        components,
        setup,
        config,
        output / TRAINABLE_SCOPE_FILE,
    )
    trainer_type = _trainer_class(config, components, setup)
    trainer = trainer_type(
        model=model,
        args=build_sft_config(config, output / "trainer", has_eval_dataset=False),
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.model.eval()

    parameter_counts = {group: 0 for group in PARAMETER_GROUPS}
    for name, parameter in trainer.model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = probe_parameter_group(parameter_category(name, components))
        if group is None:
            raise RuntimeError(f"unroutable trainable parameter: {name}")
        parameter_counts[group] += parameter.numel()

    runs: dict[str, JsonDict] = {}
    for behavior in PROBE_BEHAVIORS:
        indices = list(selection["behaviors"][behavior]["indices"])
        row_ids = list(selection["behaviors"][behavior]["row_ids"])
        snapshots = []
        losses = []
        for start in range(0, len(indices), microbatch_size):
            batch_indices = indices[start : start + microbatch_size]
            examples = [dataset[index] for index in batch_indices]
            trainer.model.zero_grad(set_to_none=True)
            inputs = trainer._prepare_inputs(trainer.data_collator(examples))
            with trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(trainer.model, inputs)
            trainer.accelerator.backward(loss)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            losses.append(float(loss.detach()))
            snapshots.append(
                capture_gradient_snapshot(
                    trainer.model,
                    lambda name: parameter_category(name, components),
                )
            )
            print(
                f"gradient_probe behavior={behavior} batch={start // microbatch_size + 1} "
                f"loss={losses[-1]:.6f}"
            )
            del examples, inputs, loss
        runs[behavior] = {"snapshots": snapshots, "losses": losses, "row_ids": row_ids}

    bundle = Path(adapter_dir)
    context = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adapter_dir": str(bundle),
        "adapter_sha256": sha256_file(bundle / "adapter_model.safetensors"),
        "visual_sha256": sha256_file(bundle / VISUAL_STATE_FILE),
        "parent_training_config": parent_config,
        "selection": selection,
        "dataset": dataset_report,
        "initial_bundle": load_report,
        "trainable_scope_schema": scope["schema"],
        "probe_mode": "eval",
        "optimizer_steps": 0,
        "microbatch_size": microbatch_size,
        "objective": {
            "completion_only_loss": True,
            "loss_type": "chunked_nll",
            "patch_loss_weight": config.patch_loss_weight,
            "patch_temperature": config.patch_temperature,
            "dropout_active": False,
        },
    }
    report = build_gradient_report(
        runs,
        learning_rates=_learning_rates(config),
        parameter_counts=parameter_counts,
        context=context,
    )
    write_json_atomic(report_path, report)
    (output / "gradient_conflicts.md").write_text(gradient_report_markdown(report))

    trainer.model.zero_grad(set_to_none=True)
    del runs, trainer, model, base_model, processor, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--probe-jsonl", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--token-inventory", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--microbatch-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_gradient_probe(
        adapter_dir=args.adapter_dir,
        probe_jsonl=args.probe_jsonl,
        image_root=args.image_root,
        token_inventory=args.token_inventory,
        selection_manifest_path=args.selection_manifest,
        output_dir=args.output_dir,
        microbatch_size=args.microbatch_size,
    )
    print(gradient_report_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
