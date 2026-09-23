"""main."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from evals.catan_board_bench.tokens import added_tokens
from sft.json_types import as_list, as_str
from sft.qwen_series_vision_sft import (
    load_recognition_token_inventory,
)

from ._args import _arg_value, _pop_flag, _pop_option, _validate_profile_arguments
from ._audit import _patch_trainer_scope_audit
from ._peft import _patch_peft_for_catan_tokens, _patch_peft_for_catan_tokens_only
from ._savers import _patch_non_deepspeed_savers, _resize_token_embeddings_without_shrinking


def main() -> int:
    model_id = _arg_value("--model_id")
    if not model_id:
        raise SystemExit("Missing required upstream argument: --model_id")

    transformers = importlib.import_module("transformers")
    upstream = importlib.import_module("train.train_sft")

    profile = _pop_option("--catan_trainable_profile")
    token_inventory_path = _pop_option("--catan_token_inventory")
    token_adapter_only = _pop_flag("--catan_token_adapter_only")
    patch_no_deepspeed_savers = _pop_flag("--catan_patch_no_deepspeed_savers")
    output_dir = None
    if profile is not None:
        output_dir = _validate_profile_arguments(
            profile,
            token_adapter_only,
            token_inventory_path,
        )
    elif token_adapter_only:
        raise SystemExit("--catan_token_adapter_only requires --catan_trainable_profile")

    processor: ProcessorMixin = transformers.AutoProcessor.from_pretrained(model_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("Qwen processor does not expose a Hugging Face tokenizer")
    tokenizer.padding_side = "right"
    if token_inventory_path is not None:
        inventory = load_recognition_token_inventory(Path(token_inventory_path))
        catan_tokens = [as_str(token) for token in as_list(inventory["tokens"])]
    else:
        catan_tokens = added_tokens()
    added = tokenizer.add_tokens(catan_tokens)

    original_load_model = upstream.load_qwen_vl_generation_model

    def load_model_and_resize(*args: object, **kwargs: object) -> torch.nn.Module:
        model: PreTrainedModel = original_load_model(*args, **kwargs)
        if added:
            _resize_token_embeddings_without_shrinking(model, tokenizer)
            print(f"added_catan_tokens={added}")
        return model

    class ProcessorShim:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> ProcessorMixin:
            return processor

    setattr(upstream, "load_qwen_vl_generation_model", load_model_and_resize)
    setattr(upstream, "AutoProcessor", ProcessorShim)
    if token_adapter_only:
        _patch_peft_for_catan_tokens_only(upstream, tokenizer, catan_tokens)
    else:
        _patch_peft_for_catan_tokens(upstream, tokenizer, catan_tokens)
    if patch_no_deepspeed_savers:
        _patch_non_deepspeed_savers(upstream)
    if profile is not None and output_dir is not None:
        if len(catan_tokens) != 220:
            raise RuntimeError(
                f"vision recognition profiles require exactly 220 tokens, got {len(catan_tokens)}"
            )
        _patch_trainer_scope_audit(
            upstream,
            profile,
            output_dir,
            catan_tokens,
        )

    upstream.train()
    return 0
