"""Wrap the real optimizer and train step, preserving their return values exactly."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch

from sft.json_types import JsonLikeDict

from .audits import adapter_parameters, gradient_witness, update_witness
from .contracts import OptimizerResult, OptimizerView, RuntimeArgs, StepResult, TrainStep, require
from .storage import write_receipt


def finite_scalar(value: float | torch.Tensor | None, label: str) -> float:
    require(value is not None, f"missing {label}")
    if value is None:
        raise ValueError(label)
    result = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    require(math.isfinite(result), f"nonfinite {label}")
    return result


def audited_train_step(
    original: TrainStep, directory: Path, args: RuntimeArgs, rollout_id: int, step_id: int,
    data_iterator: Sequence[object], model: Sequence[torch.nn.Module],
    optimizer: OptimizerView | None, opt_param_scheduler: object,
    num_microbatches: int, num_rollouts: int, witness_info: object, attempt: int,
    ft_test_action_executor: object = None,
) -> StepResult:
    require(optimizer is not None, "real optimizer required")
    if optimizer is None:
        raise ValueError("missing optimizer")
    params, scope = adapter_parameters(model, args.tensor_model_parallel_size)
    prefix = f"rollout-{rollout_id}-step-{step_id}-rank-{args.rank}"
    evidence: list[JsonLikeDict] = []
    original_step = optimizer.step

    def step() -> OptimizerResult:
        require(not evidence, "train_one_step invoked the optimizer more than once")
        gradients = gradient_witness(params)
        snapshots = {name: param.detach().clone() for name, param in params.items()}
        write_receipt(directory / f"{prefix}-before.json", {
            "rollout_id": rollout_id, "step_id": step_id, "rank": args.rank,
            "attempt": attempt, **gradients,
        })
        result = original_step()
        require(result[0] is True, "optimizer reported an unsuccessful update")
        grad_norm = finite_scalar(result[1], "optimizer grad norm")
        update = update_witness(snapshots, params)
        evidence.append({**gradients, **update, "optimizer_grad_norm": grad_norm})
        return result

    optimizer.step = step
    try:
        result = original(
            args, rollout_id, step_id, data_iterator, model, optimizer, opt_param_scheduler,
            num_microbatches, num_rollouts, witness_info, attempt, ft_test_action_executor,
        )
    finally:
        optimizer.step = original_step
    require(len(evidence) == 1, "no successful optimizer step was witnessed")
    losses, grad_norm, outcome = result
    require(outcome.name == "NORMAL" and bool(losses), "discarded step or missing real loss metrics")
    measured_losses = {key: finite_scalar(value, f"loss {key}") for key, value in losses.items()}
    _, after_scope = adapter_parameters(model, args.tensor_model_parallel_size)
    require(scope == after_scope, "trainable scope changed during optimizer step")
    write_receipt(directory / f"{prefix}-after.json", {
        "rollout_id": rollout_id, "step_id": step_id, "rank": args.rank, "attempt": attempt,
        "successful": True, "losses": measured_losses, "loss_timing": "forward_before_update",
        "grad_norm": finite_scalar(grad_norm, "step grad norm"), **evidence[0],
    })
    return result
