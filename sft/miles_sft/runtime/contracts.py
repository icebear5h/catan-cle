"""Structural interfaces for the pinned Miles API, never substitute implementations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from typing import ClassVar, Protocol

import torch

MILES_COMMIT = "24ec7d8666c2c4ca9986addd197523b89d67621d"
TARGET_SUFFIXES = (
    "self_attention.linear_qkv", "self_attention.linear_proj",
    "self_attention.in_proj", "self_attention.out_proj", "mlp.linear_fc1", "mlp.linear_fc2",
)
REQUIRED_TARGETS = tuple(f"language_model.decoder.layers.*.{s}" for s in TARGET_SUFFIXES)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class RolloutArgs(Protocol):
    rollout_global_dataset: bool
    rollout_batch_size: int
    n_samples_per_prompt: int
    apply_chat_template: bool
    input_key: str
    metadata_key: str
    rollout_max_prompt_len: int | None
    seq_length: int


class RuntimeArgs(RolloutArgs, Protocol):
    train_backend: str
    megatron_to_hf_mode: str
    debug_train_only: bool
    debug_rollout_only: bool
    debug_disable_optimizer: bool
    lora_train_only: bool
    lora_type: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_adapter_path: str | None
    target_modules: Sequence[str]
    exclude_modules: Sequence[str] | None
    loss_type: str
    calculate_per_token_loss: bool
    compute_advantages_and_returns: bool
    enable_mtp_training: bool
    use_critic: bool
    actor_num_nodes: int
    actor_num_gpus_per_node: int
    rollout_num_gpus: int
    eval_num_gpus: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int | None
    virtual_pipeline_model_parallel_size: int | None
    rank: int
    world_size: int
    hf_checkpoint: str
    prompt_data: str
    load: str | None
    save: str
    save_hf: str | None
    num_rollout: int
    start_rollout_id: int
    global_batch_size: int
    no_save_optim: bool


class SampleView(Protocol):
    Status: ClassVar[type[Enum]]
    prompt: object
    metadata: object
    tokens: list[int]
    response: str
    response_length: int
    loss_mask: list[int] | None
    status: Enum
    reward: float | object
    multimodal_inputs: object
    multimodal_train_inputs: object

    def validate(self) -> None: ...


class DataSource(Protocol):
    def get_samples(self, num_samples: int) -> list[list[SampleView]]: ...


class RolloutOutput(Protocol):
    samples: list[list[SampleView]]


class MilesTypes(Protocol):
    Sample: type[SampleView]


class MilesRollout(Protocol):
    def RolloutFnTrainOutput(self, *, samples: list[list[SampleView]]) -> RolloutOutput: ...


OptimizerResult = tuple[bool, float | torch.Tensor | None, float | torch.Tensor | None]


class OptimizerView(Protocol):
    step: Callable[[], OptimizerResult]


StepResult = tuple[dict[str, float], float | torch.Tensor, Enum]


class TrainStep(Protocol):
    def __call__(
        self, args: RuntimeArgs, rollout_id: int, step_id: int,
        data_iterator: Sequence[object], model: Sequence[torch.nn.Module],
        optimizer: OptimizerView | None, opt_param_scheduler: object,
        num_microbatches: int, num_rollouts: int, witness_info: object, attempt: int,
        ft_test_action_executor: object = None,
    ) -> StepResult: ...


class MilesModel(Protocol):
    __file__: str
    train_one_step: TrainStep
