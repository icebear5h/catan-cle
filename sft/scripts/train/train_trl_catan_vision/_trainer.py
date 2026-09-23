"""Model card and trainer construction."""

from __future__ import annotations

import importlib
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, MutableMapping, cast

import torch

from sft.scripts.train.train_trl_catan_vision._common import (
    FROZEN_ADAPTER_DIR,
    FROZEN_BUNDLE_FILE,
    OPTIMIZER_COVERAGE_FILE,
    PATCH_METRICS_FILE,
    RUN_CONFIG_FILE,
    TRAINABLE_SCOPE_FILE,
    JsonDict,
    write_json_atomic,
)
from sft.scripts.train.train_trl_catan_vision._config import (
    CatanTrainerFactory,
    ModelComponents,
    TokenSetup,
    TrainConfig,
    native_tokenizer,
    validate_resume_mode,
)
from sft.scripts.train.train_trl_catan_vision._frozen import (
    carry_processor_assets,
    freeze_visual_except_lora,
    freeze_visual_weights,
    frozen_adapter_files,
    load_visual_state,
    save_visual_state,
)
from sft.scripts.train.train_trl_catan_vision._model_tokens import (
    answer_token_metrics,
    output_head_weight,
    promote_visual_master_weights,
    trivial_completion_token_ids,
)
from sft.scripts.train.train_trl_catan_vision._optim import (
    audit_optimizer_coverage,
    audit_trainable_scope,
    build_optimizer,
)
from sft.scripts.train.train_trl_catan_vision._structure import (
    as_tensor,
    expose_trainable_tokens_head_to_chunked_nll,
    required_text,
    resolve_wrapped_module,
)
from sft.scripts.train.train_trl_catan_vision._text_data import (
    SpatialTargetCollator,
    TextCompletionCollator,
)
from sft.scripts.train.train_trl_catan_vision._visual import (
    LanguageHiddenCapture,
    VisionPoolerCapture,
    orthogonal_penalty,
    spatial_patch_loss,
)


def _trainer_class(config: TrainConfig, components: ModelComponents,
                   setup: TokenSetup) -> CatanTrainerFactory:
    TRAINING_ARGS_NAME = importlib.import_module("transformers.trainer").TRAINING_ARGS_NAME
    SFTTrainer = importlib.import_module("trl").SFTTrainer

    class CatanSFTTrainer(SFTTrainer):
        data_collator: Callable[[list[JsonDict]], MutableMapping[str, torch.Tensor]]
        optimizer: torch.optim.Optimizer | None

        def __init__(self, *args: object, **kwargs: object) -> None:
            trainer_model = cast("torch.nn.Module | None",
                                 kwargs.get("model", args[0] if args else None))
            if trainer_model is None:
                raise TypeError("CatanSFTTrainer requires a model")
            if config.text_only:
                kwargs["data_collator"] = TextCompletionCollator(
                    native_tokenizer(kwargs["processing_class"]),
                    max_sequence_length=cast("int", config.max_sequence_length),
                )
            with expose_trainable_tokens_head_to_chunked_nll(trainer_model):
                super().__init__(*args, **kwargs)
            self._catan_merge_size = None
            if not config.text_only:
                self.data_collator = SpatialTargetCollator(
                    self.data_collator,
                    setup,
                    target_mode=config.spatial_target_mode,
                )
                vision_module = resolve_wrapped_module(self.model, components.vision)
                self._catan_vision_capture = VisionPoolerCapture(vision_module)
                vision_config = getattr(getattr(self.model, "config", None), "vision_config", None)
                merge_size = getattr(vision_config, "spatial_merge_size", None)
                if merge_size is None:
                    image_processor = getattr(self.processing_class, "image_processor", None)
                    merge_size = getattr(image_processor, "merge_size", 2)
                self._catan_merge_size = int(cast("int", merge_size))
                if self._catan_merge_size <= 0:
                    raise ValueError("vision spatial merge size must be positive")
            self._catan_hidden_capture = LanguageHiddenCapture(
                resolve_wrapped_module(self.model, components.language)
            )
            self._catan_trivial_token_ids = trivial_completion_token_ids(
                native_tokenizer(self.processing_class)
            )
            self._catan_metrics: dict[str, list[float]] = defaultdict(list)
            self._catan_basis_cache: dict[str, torch.Tensor] = {}

        def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: dict[str, object],
            return_outputs: bool = False,
            num_items_in_batch: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, object]:
            if config.text_only:
                # TRL 1.12 prediction_step sets this flag; its compute_loss must
                # receive and pop it before forwarding model kwargs. Do not admit
                # arbitrary underscore-prefixed fields or drop the eval intent.
                if "_prediction_loss_only" in inputs and not isinstance(inputs["_prediction_loss_only"], bool):
                    raise ValueError("_prediction_loss_only must be a trainer-owned boolean")
                if set(inputs) - {"input_ids", "attention_mask", "labels", "_prediction_loss_only"}:
                    raise ValueError("text loss accepts only input_ids, attention_mask, and labels")
            else:
                token_ids = as_tensor(inputs.pop("spatial_target_token_ids"))
                bboxes = as_tensor(inputs.pop("spatial_target_bboxes"))
                target_mask = as_tensor(inputs.pop("spatial_target_mask"))
                image_grid_thw = inputs.get("image_grid_thw")
                self._catan_vision_capture.output = None
            labels = inputs.get("labels")
            self._catan_hidden_capture.output = None
            nll_loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            unwrapped = self.accelerator.unwrap_model(model, keep_torch_compile=False)
            total_loss = nll_loss
            if not config.text_only:
                if image_grid_thw is None:
                    raise RuntimeError("Qwen processor did not return image_grid_thw")
                if self._catan_merge_size is None:
                    raise RuntimeError("vision spatial merge size was not resolved")
                pooled = self._catan_vision_capture.take()
                token_embeddings = unwrapped.get_input_embeddings()(token_ids)
                patch_loss, patch_accuracy, target_count = spatial_patch_loss(
                    pooled,
                    as_tensor(image_grid_thw),
                    token_embeddings,
                    bboxes,
                    target_mask,
                    merge_size=self._catan_merge_size,
                    temperature=config.patch_temperature,
                )
                total_loss = nll_loss + config.patch_loss_weight * patch_loss
                self._catan_metrics["patch_loss"].append(float(patch_loss.detach()))
                self._catan_metrics["patch_top1_tolerant_accuracy"].append(float(patch_accuracy.detach()))
                self._catan_metrics["patch_target_count"].append(float(target_count))
            bases = getattr(unwrapped, "_catan_orthogonal_bases", None)
            if bases is not None and config.orthogonal_lambda > 0:
                orth, fraction, unprotected = orthogonal_penalty(unwrapped, bases, self._catan_basis_cache)
                total_loss = total_loss + config.orthogonal_lambda * orth
                self._catan_metrics["orth_loss"].append(float(orth.detach()))
                self._catan_metrics["orth_fraction"].append(fraction)
                self._catan_metrics["orth_unprotected_modules"].append(float(unprotected))
            if labels is not None:
                weight, bias = output_head_weight(unwrapped.get_base_model().get_output_embeddings())
                answer = answer_token_metrics(
                    self._catan_hidden_capture.take(),
                    as_tensor(labels),
                    weight,
                    bias,
                    self._catan_trivial_token_ids,
                )
                for name, value in answer.items():
                    self._catan_metrics[name].append(value)
            self._catan_metrics["nll_loss"].append(float(nll_loss.detach()))
            return (total_loss, outputs) if return_outputs else total_loss

        def log(self, logs: dict[str, float], *args: object, **kwargs: object) -> None:
            for name, values in self._catan_metrics.items():
                if values:
                    logs[name] = sum(values) / len(values)
            self._catan_metrics.clear()
            super().log(logs, *args, **kwargs)

        def _clip_grad_norm(self, model: torch.nn.Module) -> torch.Tensor | float | None:
            # Record the pre-clip norm of every optimizer group so a dominant
            # group cannot hide behind the single clipped scalar Trainer logs.
            optimizer = self.optimizer
            if optimizer is None:
                raise RuntimeError("gradient clipping ran before the optimizer was created")
            with torch.no_grad():
                for group in optimizer.param_groups:
                    grads = [p.grad.detach().float().norm() for p in group["params"] if p.grad is not None]
                    if grads:
                        norm = torch.norm(torch.stack(grads))
                        self._catan_metrics[f"grad_norm_{group['catan_name']}"].append(float(norm))
            clipped: torch.Tensor | float | None = super()._clip_grad_norm(model)
            return clipped

        def create_optimizer(self,
                             model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
            if self.optimizer is None:
                target = self.model if model is None else model
                self.optimizer = build_optimizer(target, components, config)
                audit_optimizer_coverage(
                    target,
                    self.optimizer,
                    Path(self.args.output_dir) / OPTIMIZER_COVERAGE_FILE,
                )
            return self.optimizer

        def _save(self, output_dir: str | None = None,
                  state_dict: dict[str, torch.Tensor] | None = None) -> None:
            target_dir = Path(output_dir or self.args.output_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            unwrapped.save_pretrained(
                target_dir,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            self.processing_class.save_pretrained(target_dir)
            if config.text_only:
                carry_processor_assets(required_text(config.initial_bundle, "initial_bundle"), target_dir)
            torch.save(self.args, target_dir / TRAINING_ARGS_NAME)
            is_checkpoint = Path(self.args.output_dir).resolve() in target_dir.resolve().parents
            save_visual_state(
                unwrapped,
                components,
                target_dir,
                dtype=None if is_checkpoint or config.text_only else torch.bfloat16,
            )
            audit_trainable_scope(
                unwrapped,
                components,
                setup,
                config,
                target_dir / TRAINABLE_SCOPE_FILE,
            )
            write_json_atomic(target_dir / RUN_CONFIG_FILE, asdict(config))
            frozen_dir = getattr(unwrapped, "_catan_frozen_adapter_dir", None)
            if frozen_dir is not None:
                # A bundle from this profile only reproduces its model on top of the
                # merged parent, so the parent adapter travels inside every save.
                carried = target_dir / FROZEN_ADAPTER_DIR
                carried.mkdir(exist_ok=True)
                for source in frozen_adapter_files(Path(frozen_dir)):
                    shutil.copy2(source, carried / source.name)
                write_json_atomic(target_dir / FROZEN_BUNDLE_FILE, {"schema": "catan_trl_frozen_bundle_pointer/v1", "path": str(frozen_dir), "carried_adapter": str(carried), "orthogonal_lambda": config.orthogonal_lambda})
            if config.text_only:
                return
            write_json_atomic(
                target_dir / PATCH_METRICS_FILE,
                {
                    "schema": "catan_patch_localization_config/v1",
                    "loss_weight": config.patch_loss_weight,
                    "temperature": config.patch_temperature,
                    "target_mode": config.spatial_target_mode,
                    "spatial_merge_size": self._catan_merge_size,
                    "feature_space": "post_merger_lm_hidden",
                    "normalization": "soft_cross_entropy_div_log_patch_count",
                },
            )

        def _load_from_checkpoint(
            self,
            resume_from_checkpoint: str,
            model: torch.nn.Module | None = None,
        ) -> None:
            validate_resume_mode(resume_from_checkpoint, config.input_mode)
            super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            target = self.model if model is None else model
            if config.text_only:
                freeze_visual_weights(target, components)
            elif config.olora:
                freeze_visual_except_lora(target, components)
            else:
                promote_visual_master_weights(target, components)
            load_visual_state(target, resume_from_checkpoint)

    return CatanSFTTrainer
