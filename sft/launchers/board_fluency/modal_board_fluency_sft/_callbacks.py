from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_int, as_str
from sft.launchers.board_fluency import modal_board_fluency_eval as shared
from sft.launchers.modal_catan_vision_sft import sft_runs
from sft.scripts.train import train_trl_catan_vision as trainer
from sft.scripts.train.train_trl_catan_vision import TrainConfig, write_json_atomic

from ._config import ALLOWED
from ._data import progress
from ._planning import check_deadline
from ._probes import ProbeResult, compare_probes, probe, visual_digest


def require_model(model: torch.nn.Module | None) -> torch.nn.Module:
    if model is None:
        raise RuntimeError("Trainer callback received no model")
    return model


def require_optimizer(optimizer: torch.optim.Optimizer | None) -> torch.optim.Optimizer:
    if optimizer is None:
        raise RuntimeError("Trainer callback received no optimizer")
    return optimizer


def model_components(scope: JsonDict) -> trainer.ModelComponents:
    """`ModelComponents(**scope["components"])` with each field checked."""
    components = as_dict(scope["components"])
    return trainer.ModelComponents(
        input_embedding=as_str(components["input_embedding"]),
        output_head=as_str(components["output_head"]),
        language=as_str(components["language"]), vision=as_str(components["vision"]),
        merger=as_str(components["merger"]), vocab_size=as_int(components["vocab_size"]),
        hidden_size=as_int(components["hidden_size"]),
    )


class CheckpointCallback(TrainerCallback):
    def __init__(self, config: TrainConfig, deadline: float) -> None:
        self.config, self.deadline = config, deadline
        self.saved_steps: list[int] = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        **kwargs: object,
    ) -> None:
        check_deadline(self.deadline)
        model, optimizer = require_model(model), require_optimizer(optimizer)
        if state.global_step != 0 or optimizer.state:
            raise RuntimeError("continuation requires a fresh optimizer at step zero")
        scope = shared.read_json(Path(self.config.output_dir) / trainer.TRAINABLE_SCOPE_FILE)
        if scope["errors"]:
            raise RuntimeError("trainer scope audit failed")
        # Discovery must run before PEFT wraps Embedding/Linear. Reuse the
        # trainer's actual pre-wrap component audit instead of rediscovering it.
        self.components = model_components(scope)
        groups = Counter(trainer.parameter_category(n, self.components) for n, p in model.named_parameters() if p.requires_grad)
        if set(groups) != ALLOWED or groups["atlas_input_rows"] != 1 or groups["atlas_output_rows"] != 1:
            raise RuntimeError(f"unapproved trainable scope: {groups}")
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        optimized = [id(p) for g in optimizer.param_groups for p in g["params"]]
        if set(optimized) != trainable or len(set(optimized)) != len(optimized):
            raise RuntimeError("optimizer must cover exactly the approved trainable tensors once")
        for group in optimizer.param_groups:
            if not group["params"]:
                continue
            expected = self.config.language_lora_learning_rate if group["catan_name"] == "language_lora" else self.config.learning_rate
            if group.get("initial_lr", group["lr"]) != expected:
                raise RuntimeError("optimizer group learning rate differs")

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        check_deadline(self.deadline)

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        check_deadline(self.deadline)

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        check_deadline(self.deadline)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        check_deadline(self.deadline)
        if logs:
            if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in logs.values()):
                raise RuntimeError("nonfinite training/evaluation metric")
            progress(f"step {state.global_step}: {json.dumps(logs, sort_keys=True)}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        # Trainer emits on_save after adapter, visual state AND optimizer/state
        # files are complete. Caller callbacks run after its bundle-saving hooks.
        if args.output_dir is None:
            raise RuntimeError("Trainer saved without an output directory")
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        shared.volume_bundle(str(checkpoint))
        for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(f"incomplete Trainer checkpoint: {checkpoint / name}")
        if shared.read_json(checkpoint / "trainer_state.json")["global_step"] != state.global_step:
            raise ValueError("checkpoint global step differs")
        self.saved_steps.append(state.global_step)
        write_json_atomic(Path(self.config.output_dir) / "committed_checkpoints.json",
                          {"steps": self.saved_steps, "latest": str(checkpoint), "at": shared.now()})
        sft_runs.commit()
        progress(f"committed complete Trainer checkpoint {state.global_step}")
        check_deadline(self.deadline)


class GateCallback(CheckpointCallback):
    def __init__(
        self,
        config: TrainConfig,
        deadline: float,
        rows: list[JsonDict],
        before: ProbeResult,
    ) -> None:
        super().__init__(config, deadline)
        self.rows, self.before = rows, before
        self.initial: dict[str, torch.Tensor] = {}
        self.frozen: dict[str, tuple[int, int]] = {}
        self.losses: list[float] = []
        self.new_b_gradient = 0.0
        self.group_gradients = dict.fromkeys(ALLOWED, 0.0)
        self.after: ProbeResult | None = None
        self.report: JsonLikeDict = {}

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        **kwargs: object,
    ) -> None:
        super().on_train_begin(args, state, control, model=model, optimizer=optimizer, **kwargs)
        model = require_model(model)
        processing_class = kwargs.get("processing_class")
        if not isinstance(processing_class, PreTrainedTokenizerBase):
            raise RuntimeError("gate callback requires the Trainer's tokenizer")
        self.tokenizer = processing_class
        self.report["trainer_initial_equivalence"] = compare_probes(self.before, probe(model, self.tokenizer, self.rows))
        self.visual_before = visual_digest(model)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.initial[name] = parameter.detach().cpu().clone()
                if ".lora_B." in name and torch.count_nonzero(parameter[:, 8:]).item():
                    raise RuntimeError("new rank-B columns must begin at zero")
            else:
                self.frozen[name] = (id(parameter), parameter._version)

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **kwargs: object,
    ) -> None:
        check_deadline(self.deadline)
        for name, parameter in require_model(model).named_parameters():
            if name in self.frozen:
                if parameter.requires_grad or parameter.grad is not None or (id(parameter), parameter._version) != self.frozen[name]:
                    raise RuntimeError(f"frozen base/visual tensor changed: {name}")
            elif parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"nonfinite gate gradient: {name}")
                category = trainer.parameter_category(name, self.components)
                self.group_gradients[category] = max(
                    self.group_gradients[category], float(parameter.grad.abs().max()),
                )
                if ".lora_B." in name:
                    self.new_b_gradient = max(self.new_b_gradient, float(parameter.grad[:, 8:].abs().max()))

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        super().on_log(args, state, control, logs=logs, **kwargs)
        if logs and "nll_loss" in logs:
            self.losses.append(float(logs["nll_loss"]))

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        super().on_save(args, state, control, **kwargs)
        found = kwargs.get("model")
        model = require_model(found if isinstance(found, torch.nn.Module) else None)
        updates, new_b_update = dict.fromkeys(ALLOWED, 0.0), 0.0
        for name, parameter in model.named_parameters():
            if name in self.frozen:
                if parameter.grad is not None or (id(parameter), parameter._version) != self.frozen[name]:
                    raise RuntimeError(f"frozen tensor changed during gate: {name}")
            else:
                current = parameter.detach().cpu()
                if not torch.isfinite(current).all():
                    raise RuntimeError("nonfinite updated gate parameter")
                delta = (current - self.initial[name]).abs()
                category = trainer.parameter_category(name, self.components)
                updates[category] = max(updates[category], float(delta.max()))
                if ".lora_B." in name:
                    new_b_update = max(new_b_update, float(delta[:, 8:].max()))
        if (state.global_step != 2 or len(self.losses) != 2 or not all(math.isfinite(v) for v in self.losses)
                or not all(v > 0 for v in updates.values())
                or not all(v > 0 for v in self.group_gradients.values())
                or new_b_update <= 0 or self.new_b_gradient <= 0):
            raise RuntimeError(f"gate did not learn in all approved groups: updates={updates}, new_B={new_b_update}")
        self.after = probe(model, self.tokenizer, self.rows)
        visual_after = visual_digest(model)
        if self.visual_before != visual_after:
            raise ValueError("visual tensors changed during the two training steps")
        self.report.update(losses=self.losses, max_abs_update=updates,
                            max_abs_gradient=self.group_gradients,
                           new_rank_b_max_gradient=self.new_b_gradient,
                           new_rank_b_max_update=new_b_update, visual_tensor_sha256=visual_after,
                           frozen_tensors_checked=len(self.frozen), added_rank_a_gradient_required=False)
        write_json_atomic(Path(self.config.output_dir) / "gate_updates.json", self.report)
        sft_runs.commit()
