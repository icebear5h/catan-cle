"""Run the upstream Qwen-VL-Series-Finetune SFT trainer with Catan tokens.

This wrapper intentionally keeps the trainer implementation upstream-owned. It
loads the processor before the upstream training function, adds Catan atlas
tokens, and patches the upstream model loader so token embeddings are resized
before PEFT wraps the model.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from data_pipeline.catan_board_bench.tokens import add_tokens_to_tokenizer, added_tokens


def _arg_value(name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(sys.argv):
        if arg == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return default


def _patch_non_deepspeed_savers(upstream: Any) -> None:
    """Avoid requiring DeepSpeed for single-GPU Modal smoke runs."""

    def maybe_clone(param):
        return param.detach().cpu().clone()

    def is_peft_adapter_param(name):
        return "lora_" in name or "token_adapter" in name or "trainable_tokens" in name

    def get_peft_state_maybe_zero_3(named_params, bias):
        if bias == "none":
            selected = {k: t for k, t in named_params if is_peft_adapter_param(k)}
        elif bias == "all":
            selected = {k: t for k, t in named_params if is_peft_adapter_param(k) or "bias" in k}
        else:
            selected = {k: t for k, t in named_params if is_peft_adapter_param(k)}
        return {k: maybe_clone(v) for k, v in selected.items()}

    def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
        selected = {k: t for k, t in named_params if not is_peft_adapter_param(k)}
        if require_grad_only:
            selected = {k: t for k, t in selected.items() if t.requires_grad}
        return {k: maybe_clone(v) for k, v in selected.items()}

    upstream.get_peft_state_maybe_zero_3 = get_peft_state_maybe_zero_3
    upstream.get_peft_state_non_lora_maybe_zero_3 = get_peft_state_non_lora_maybe_zero_3


def _resize_token_embeddings_without_shrinking(model: Any, tokenizer: Any) -> None:
    """Resize only when Catan tokens exceed Qwen's padded vocab rows."""

    target_rows = len(tokenizer)
    input_embeddings = model.get_input_embeddings()
    current_rows = input_embeddings.num_embeddings

    if target_rows <= current_rows:
        print(
            f"token_embeddings_already_cover_tokenizer={current_rows}; tokenizer_len={target_rows}"
        )
        return

    model.resize_token_embeddings(target_rows, pad_to_multiple_of=64)
    resized_rows = model.get_input_embeddings().num_embeddings
    print(f"resized_token_embeddings={current_rows}->{resized_rows}; tokenizer_len={target_rows}")


def _find_language_embed_tokens_name(model: Any) -> str:
    """Return the concrete embedding module PEFT should train token rows on."""

    candidates = []
    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and "visual" not in name:
            candidates.append((name, module))

    if not candidates:
        raise RuntimeError("Could not find language embed_tokens module")

    if len(candidates) == 1:
        return candidates[0][0]

    for name, _ in candidates:
        if "language_model" in name:
            return name

    return candidates[0][0]


def _patch_peft_for_catan_tokens(upstream: Any, tokenizer: Any, catan_tokens: list[str]) -> None:
    """Attach trainable Catan token rows to the upstream LoRA config."""

    token_ids = tokenizer.convert_tokens_to_ids(catan_tokens)
    token_ids = [token_id for token_id in token_ids if token_id != tokenizer.unk_token_id]
    if not token_ids:
        return

    original_get_peft_model = upstream.get_peft_model

    def get_peft_model_with_catan_tokens(model, peft_config, *args, **kwargs):
        if hasattr(peft_config, "trainable_token_indices"):
            embed_name = _find_language_embed_tokens_name(model)
            peft_config.trainable_token_indices = {embed_name: token_ids}
            print(f"catan_trainable_token_indices={len(token_ids)} on {embed_name}")
        else:
            print("warning=trainable_token_indices_not_supported_by_peft")
        return original_get_peft_model(model, peft_config, *args, **kwargs)

    upstream.get_peft_model = get_peft_model_with_catan_tokens


def main() -> int:
    model_id = _arg_value("--model_id")
    if not model_id:
        raise SystemExit("Missing required upstream argument: --model_id")

    transformers = importlib.import_module("transformers")
    upstream = importlib.import_module("train.train_sft")

    processor = transformers.AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
    added = add_tokens_to_tokenizer(processor.tokenizer)
    catan_tokens = added_tokens()

    original_load_model = upstream.load_qwen_vl_generation_model

    def load_model_and_resize(*args, **kwargs):
        model = original_load_model(*args, **kwargs)
        if added:
            _resize_token_embeddings_without_shrinking(model, processor.tokenizer)
            print(f"added_catan_tokens={added}")
        return model

    class ProcessorShim:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return processor

    upstream.load_qwen_vl_generation_model = load_model_and_resize
    upstream.AutoProcessor = ProcessorShim
    _patch_peft_for_catan_tokens(upstream, processor.tokenizer, catan_tokens)
    if "--catan_patch_no_deepspeed_savers" in sys.argv:
        sys.argv.remove("--catan_patch_no_deepspeed_savers")
        _patch_non_deepspeed_savers(upstream)

    upstream.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
