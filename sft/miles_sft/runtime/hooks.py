"""Trainer hooks loaded only inside the real Miles/Megatron worker."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import torch

from sft.json_types import JsonLikeDict, load_json_dict
from sft.miles_sft.export import assemble_complete_export

from .admission import validate_args
from .artifacts import audit_hf_export, audit_native
from .audits import adapter_parameters, audit_fresh_adapter
from .contracts import MilesModel, OptimizerView, RuntimeArgs, StepResult, require
from .storage import file_hash, receipt_directory, verify_miles_revision, write_receipt
from .witness import audited_train_step


class BridgeAdapters(Protocol):
    ParallelLinearAdapter: type[torch.nn.Module]


MODEL = cast(MilesModel, importlib.import_module("miles.backends.megatron_utils.model"))
BRIDGE = cast(BridgeAdapters, importlib.import_module("megatron.bridge.peft.utils"))
_ORIGINAL_STEP = MODEL.train_one_step
_INITIALIZED = False
_SCOPE: JsonLikeDict | None = None


def initialize(args: RuntimeArgs) -> None:
    """--custom-megatron-init-path: validate before building fresh Bridge LoRA."""
    global _INITIALIZED
    require(not _INITIALIZED, "runtime initialized twice")
    validate_args(args)
    revision = verify_miles_revision(MODEL.__file__)
    expected = ["args", "rollout_id", "step_id", "data_iterator", "model", "optimizer",
                "opt_param_scheduler", "num_microbatches", "num_rollouts", "witness_info",
                "attempt", "ft_test_action_executor"]
    require(list(inspect.signature(_ORIGINAL_STEP).parameters) == expected,
            "pinned Miles train_one_step signature changed")
    write_receipt(receipt_directory() / f"initialize-rank-{args.rank}.json", {
        "schema": "catan_miles_sft_runtime/v1", "miles_commit": revision,
        "rank": args.rank, "world_size": args.world_size, "hf_checkpoint": args.hf_checkpoint,
        "save": args.save, "save_hf": args.save_hf, "num_rollout": args.num_rollout,
        "steps_per_rollout": args.rollout_batch_size // args.global_batch_size,
        "rollout_batch_size": args.rollout_batch_size, "prompt_data": args.prompt_data,
        "input_sha256": file_hash(Path(args.prompt_data)),
        "lora_type": args.lora_type, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "targets": list(args.target_modules),
    })
    MODEL.train_one_step = _train_step
    _INITIALIZED = True


def _train_step(
    args: RuntimeArgs, rollout_id: int, step_id: int, data_iterator: Sequence[object],
    model: Sequence[torch.nn.Module], optimizer: OptimizerView | None, opt_param_scheduler: object,
    num_microbatches: int, num_rollouts: int, witness_info: object, attempt: int,
    ft_test_action_executor: object = None,
) -> StepResult:
    return audited_train_step(
        _ORIGINAL_STEP, receipt_directory(), args, rollout_id, step_id, data_iterator, model,
        optimizer, opt_param_scheduler, num_microbatches, num_rollouts, witness_info, attempt,
        ft_test_action_executor,
    )


def before_train_step(
    args: RuntimeArgs, rollout_id: int, step_id: int, model: Sequence[torch.nn.Module],
    optimizer: OptimizerView | None, opt_param_scheduler: object,
) -> None:
    """Pinned before-step signature: inspect the built model, never change its scope."""
    global _SCOPE
    require(_INITIALIZED and MODEL.train_one_step is _train_step, "audited step wrapper not installed")
    validate_args(args)
    require(optimizer is not None and opt_param_scheduler is not None, "missing optimizer/scheduler")
    params, scope = adapter_parameters(model, args.tensor_model_parallel_size)
    for name, module in model[0].named_modules():
        if name + ".linear_in.weight" in params:
            require(isinstance(module, BRIDGE.ParallelLinearAdapter),
                    f"expected real Bridge ParallelLinearAdapter: {name}")
            require(getattr(module, "dim", None) == 16 and getattr(module, "alpha", None) == 32,
                    f"live adapter rank/alpha mismatch: {name}")
    if _SCOPE is None:
        require(rollout_id == step_id == 0, "first fresh step must be rollout0/step0")
        audit_fresh_adapter(params)
        write_receipt(receipt_directory() / f"scope-rank-{args.rank}.json", scope)
        _SCOPE = scope
    else:
        require(_SCOPE == scope, "trainable scope drifted")


def post_save(
    args: RuntimeArgs, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str | None,
) -> None:
    """Record model artifacts only. The driver saves the data-source cursor later."""
    require(_INITIALIZED and _SCOPE is not None and args.rank == 0, "missing live scope audit")
    require(hf_checkpoint_dir is not None and args.save_hf is not None, "merged HF export required")
    if hf_checkpoint_dir is None or args.save_hf is None:
        raise ValueError("missing export")
    expected = Path(args.save) / f"iter_{rollout_id:07d}"
    require(Path(checkpoint_dir).resolve() == expected.resolve(), "native save path mismatch")
    require(Path(hf_checkpoint_dir).resolve() == Path(args.save_hf.format(rollout_id=rollout_id)).resolve(),
            "HF save path mismatch")
    directory = receipt_directory()
    scopes = [load_json_dict(directory / f"scope-rank-{rank}.json") for rank in range(args.world_size)]
    native = audit_native(expected, rollout_id, scopes)
    bridge_path = Path(hf_checkpoint_dir).resolve()
    final_path = bridge_path.parent / "model"
    assemble_complete_export(Path(args.hf_checkpoint), bridge_path, final_path)
    exported = audit_hf_export(Path(args.hf_checkpoint), final_path, bridge_export=bridge_path)
    write_receipt(directory / f"checkpoint-{rollout_id}.json", {
        "rollout_id": rollout_id, "status": "model_saved_cursor_pending", "complete": False,
        "checkpoint_dir": str(expected.resolve()), "bridge_checkpoint_dir": str(bridge_path),
        "hf_checkpoint_dir": str(final_path),
        "native": native, "hf": exported,
    })
