"""Pinned ms-swift 4.5.2 integration for Catan semantic vision SFT."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import peft
import safetensors.torch
import swift
import torch
from accelerate.data_loader import DataLoaderShard, SkipBatchSampler
from peft import LoraConfig, PeftModel, TrainableTokensConfig, get_peft_model
from swift.arguments import SftArguments
from swift.pipelines.train.sft import SwiftSft
from swift.trainers import Seq2SeqTrainer, TrainerFactory
from swift.tuner_plugin import Tuner, tuners_map
from swift.utils import get_multimodal_target_regex
from torch.utils.data import BatchSampler
from transformers import PreTrainedModel
from transformers.trainer_utils import seed_worker

from sft.board.density_curriculum import SequentialCurriculumSampler, load_curriculum_manifest
from sft.json_types import as_int
from sft.ms_swift_core import (
    CATAN_TUNER_TYPE,
    MS_SWIFT_VERSION,
    PEFT_VERSION,
    SemanticTokenSetup,
    attach_model_components,
    audit_optimizer_coverage,
    audit_trainable_scope,
    discover_model_components,
    get_model_components,
    load_semantic_token_inventory,
    parameter_matches_paths,
    prepare_peft_trainable_token_targets,
    prepare_semantic_tokens,
    resolve_model_module,
)

TRAINABLE_SCOPE_FILE = "trainable_parameters.json"
OPTIMIZER_COVERAGE_FILE = "optimizer_coverage.json"
VISUAL_STATE_FILE = "vit.safetensors"


def assert_ms_swift_version() -> None:
    """Assert the complete released ms-swift/PEFT runtime pair."""

    if swift.__version__ != MS_SWIFT_VERSION:
        raise RuntimeError(
            f"Catan SFT requires ms-swift=={MS_SWIFT_VERSION}; found {swift.__version__}"
        )
    if peft.__version__ != PEFT_VERSION:
        raise RuntimeError(f"Catan SFT requires peft=={PEFT_VERSION}; found {peft.__version__}")


def _set_visual_trainable(model: torch.nn.Module, trainable: bool) -> None:
    components = get_model_components(model)
    for module_path in (*components.vision, *components.aligner):
        resolve_model_module(model, module_path).requires_grad_(trainable)


def _semantic_setup(model: torch.nn.Module) -> SemanticTokenSetup:
    setup = getattr(model, "_catan_semantic_token_setup", None)
    if setup is None:
        setup = getattr(getattr(model, "model", None), "_catan_semantic_token_setup", None)
    if not isinstance(setup, SemanticTokenSetup) or len(setup.token_ids) != 154:
        raise RuntimeError("model was not prepared with 154 semantic token IDs")
    return setup


class CatanVisionTokenTuner(Tuner):
    """Two selective token-row adapters plus visual modules and optional LoRA."""

    @staticmethod
    def prepare_model(args: "CatanSftArguments", model: PreTrainedModel) -> torch.nn.Module:
        assert_ms_swift_version()
        setup = _semantic_setup(model)
        components = get_model_components(model)
        model.requires_grad_(False)
        token_targets = prepare_peft_trainable_token_targets(
            model,
            components,
            setup.token_ids,
            peft_version=peft.__version__,
        )
        config: LoraConfig | TrainableTokensConfig
        if args.catan_language_lora:
            target_regex = get_multimodal_target_regex(
                model,
                freeze_llm=False,
                freeze_vit=True,
                freeze_aligner=True,
            )
            config = LoraConfig(
                task_type=args.task_type.upper(),
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                target_modules=target_regex,
                trainable_token_indices=token_targets,
            )
        else:
            config = TrainableTokensConfig(
                token_indices=list(setup.token_ids),
                target_modules=list(token_targets),
                init_weights=True,
            )
        peft_model = get_peft_model(model, config)
        setattr(peft_model, "_catan_semantic_token_setup", setup)
        attach_model_components(peft_model, components)
        _set_visual_trainable(peft_model, True)
        audit_trainable_scope(
            peft_model,
            language_lora=args.catan_language_lora,
            output_path=Path(args.output_dir) / TRAINABLE_SCOPE_FILE,
        )
        return peft_model

    @staticmethod
    def save_pretrained(
        model: PeftModel,
        save_directory: str,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        safe_serialization: bool = True,
        **kwargs: object,
    ) -> None:
        if state_dict is None:
            state_dict = {
                name: parameter.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        kwargs.setdefault("save_embedding_layers", False)
        # ms-swift forwards its own saver kwargs opaquely; PEFT validates them.
        save_adapter = getattr(model, "save_pretrained")
        save_adapter(
            save_directory,
            state_dict=state_dict,
            safe_serialization=safe_serialization,
            **kwargs,
        )
        components = get_model_components(model)
        visual_state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in state_dict.items()
            if parameter_matches_paths(name, (*components.vision, *components.aligner))
        }
        if not visual_state:
            raise RuntimeError("checkpoint has no full vision/aligner state")
        safetensors.torch.save_file(
            visual_state,
            os.path.join(save_directory, VISUAL_STATE_FILE),
            metadata={"format": "pt", "ms_swift_version": MS_SWIFT_VERSION},
        )

    @staticmethod
    def from_pretrained(
        model: PreTrainedModel,
        model_id: str,
        **kwargs: object,
    ) -> torch.nn.Module:
        assert_ms_swift_version()
        setup = _semantic_setup(model)
        visual_path = os.path.join(model_id, VISUAL_STATE_FILE)
        if not os.path.isfile(visual_path):
            raise FileNotFoundError(visual_path)
        # ms-swift forwards its own loader kwargs opaquely; PEFT validates them.
        load_adapter = getattr(PeftModel, "from_pretrained")
        peft_model = load_adapter(model, model_id, **kwargs)
        if not isinstance(peft_model, PeftModel):
            raise RuntimeError("PeftModel.from_pretrained did not return a PeftModel")
        setattr(peft_model, "_catan_semantic_token_setup", setup)
        visual_state = safetensors.torch.load_file(visual_path)
        incompatible = peft_model.load_state_dict(visual_state, strict=False)
        unexpected = [name for name in incompatible.unexpected_keys if name in visual_state]
        if unexpected:
            raise RuntimeError(f"unexpected visual checkpoint keys: {unexpected[:8]}")
        if kwargs.get("is_trainable", False):
            _set_visual_trainable(peft_model, True)
        # Keep the base LM head frozen without recursively freezing the
        # selective atlas output-row adapter when resuming training.
        peft_model.get_output_embeddings().weight.requires_grad_(False)
        return peft_model


@dataclass
class CatanSftArguments(SftArguments):
    catan_token_inventory: str = ""
    catan_language_lora: bool = True
    catan_curriculum_manifest: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tuner_type != CATAN_TUNER_TYPE:
            raise ValueError(f"--tuner_type must be {CATAN_TUNER_TYPE}")
        if not self.catan_token_inventory:
            raise ValueError("--catan_token_inventory is required")
        if self.new_special_tokens:
            raise ValueError("Catan atlas tokens must be regular tokens, not new_special_tokens")
        super().__post_init__()
        if self.catan_curriculum_manifest:
            if self.global_world_size != 1:
                raise ValueError("the sequential density curriculum currently requires one process")
            if self.dataset_shuffle or self.training_args.train_dataloader_shuffle:
                raise ValueError(
                    "the density curriculum requires --dataset_shuffle false and "
                    "--train_dataloader_shuffle false"
                )
            if self.packing or self.streaming or self.group_by_length:
                raise ValueError(
                    "the density curriculum requires packing, streaming, and group_by_length off"
                )
            if not self.strict or self.split_dataset_ratio != 0:
                raise ValueError(
                    "the density curriculum requires --strict true and --split_dataset_ratio 0"
                )
            if self.max_steps > 0 or self.num_train_epochs != 1:
                raise ValueError("the density curriculum requires exactly one uncapped epoch")


class CatanSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args: object, catan_curriculum_manifest: str | None = None,
                 **kwargs: object) -> None:
        self.catan_curriculum_manifest = catan_curriculum_manifest
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self, skip_batches: int = 0) -> DataLoaderShard:
        if not self.catan_curriculum_manifest:
            return super().get_train_dataloader(skip_batches=skip_batches)
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")
        manifest = load_curriculum_manifest(self.catan_curriculum_manifest)
        sampler = SequentialCurriculumSampler(
            self.train_dataset,
            expected_rows=as_int(manifest["total_rows"]),
        )
        batch_sampler: BatchSampler | SkipBatchSampler = BatchSampler(
            sampler,
            batch_size=self._train_batch_size,
            drop_last=self.args.dataloader_drop_last,
        )
        if skip_batches > 0:
            batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=skip_batches)
        dataloader_params = {
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "batch_sampler": batch_sampler,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        }
        return DataLoaderShard(
            self.train_dataset,
            device=self.accelerator.device,
            **dataloader_params,
        )

    def create_optimizer(self,
                         model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
        optimizer: torch.optim.Optimizer | None = super().create_optimizer(model=model)
        if optimizer is None:
            raise RuntimeError("ms-swift did not create an optimizer")
        audit_optimizer_coverage(
            self.model if model is None else model,
            optimizer,
            output_path=Path(self.args.output_dir) / OPTIMIZER_COVERAGE_FILE,
        )
        return optimizer


class CatanSwiftSft(SwiftSft):
    args_class = CatanSftArguments

    def _prepare_model_tokenizer(self, **kwargs: object) -> None:
        super()._prepare_model_tokenizer(**kwargs)
        if self.model is None:
            raise RuntimeError("Catan semantic SFT requires a loaded model")
        discover_model_components(self.model)
        inventory = load_semantic_token_inventory(self.args.catan_token_inventory)
        setup = prepare_semantic_tokens(self.tokenizer, self.model, inventory)
        setattr(self.model, "_catan_semantic_token_setup", setup)

    def _get_trainer_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = super()._get_trainer_kwargs()
        kwargs["catan_curriculum_manifest"] = self.args.catan_curriculum_manifest
        return kwargs


tuners_map[CATAN_TUNER_TYPE] = CatanVisionTokenTuner
TrainerFactory.TRAINER_MAPPING["causal_lm"] = "sft.ms_swift_plugin.CatanSeq2SeqTrainer"
