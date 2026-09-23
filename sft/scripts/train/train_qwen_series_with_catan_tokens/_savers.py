"""savers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _patch_non_deepspeed_savers(upstream: ModuleType) -> None:
    """Avoid requiring DeepSpeed for single-GPU Modal smoke runs."""

    def maybe_clone(param: torch.Tensor) -> torch.Tensor:
        return param.detach().cpu().clone()

    def is_peft_adapter_param(name: str) -> bool:
        return "lora_" in name or "token_adapter" in name or "trainable_tokens" in name

    def get_peft_state_maybe_zero_3(
        named_params: Iterable[tuple[str, torch.Tensor]],
        bias: str,
    ) -> dict[str, torch.Tensor]:
        if bias == "none":
            selected = {k: t for k, t in named_params if is_peft_adapter_param(k)}
        elif bias == "all":
            selected = {k: t for k, t in named_params if is_peft_adapter_param(k) or "bias" in k}
        else:
            selected = {k: t for k, t in named_params if is_peft_adapter_param(k)}
        return {k: maybe_clone(v) for k, v in selected.items()}

    def get_peft_state_non_lora_maybe_zero_3(
        named_params: Iterable[tuple[str, torch.Tensor]],
        require_grad_only: bool = True,
    ) -> dict[str, torch.Tensor]:
        selected = {k: t for k, t in named_params if not is_peft_adapter_param(k)}
        if require_grad_only:
            selected = {k: t for k, t in selected.items() if t.requires_grad}
        return {k: maybe_clone(v) for k, v in selected.items()}

    setattr(upstream, "get_peft_state_maybe_zero_3", get_peft_state_maybe_zero_3)
    setattr(
        upstream, "get_peft_state_non_lora_maybe_zero_3", get_peft_state_non_lora_maybe_zero_3
    )

    trainer_module = importlib.import_module(upstream.QwenSFTTrainer.__module__)
    setattr(trainer_module, "get_peft_state_maybe_zero_3", get_peft_state_maybe_zero_3)
    setattr(
        trainer_module,
        "get_peft_state_non_lora_maybe_zero_3",
        get_peft_state_non_lora_maybe_zero_3,
    )


def _resize_token_embeddings_without_shrinking(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Resize only when Catan tokens exceed Qwen's padded vocab rows."""

    target_rows = len(tokenizer)
    input_embeddings = model.get_input_embeddings()
    current_rows = input_embeddings.num_embeddings
    if not isinstance(current_rows, int):
        raise TypeError("Input embeddings do not expose an integer num_embeddings")

    if target_rows <= current_rows:
        print(
            f"token_embeddings_already_cover_tokenizer={current_rows}; tokenizer_len={target_rows}"
        )
        return

    model.resize_token_embeddings(target_rows, pad_to_multiple_of=64)
    resized_rows = model.get_input_embeddings().num_embeddings
    print(f"resized_token_embeddings={current_rows}->{resized_rows}; tokenizer_len={target_rows}")
