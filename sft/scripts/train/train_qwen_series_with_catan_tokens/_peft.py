"""peft."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ._modules import _catan_token_ids, _language_token_target_names

if TYPE_CHECKING:
    from peft import PeftConfig, PeftModel


class GetPeftModel(Protocol):
    """Upstream `get_peft_model` as the Qwen trainer calls it."""

    def __call__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        /,
        *args: object,
        **kwargs: object,
    ) -> PeftModel: ...


def _patch_peft_for_catan_tokens(
    upstream: ModuleType,
    tokenizer: PreTrainedTokenizerBase,
    catan_tokens: list[str],
) -> None:
    """Attach selective Catan input/output rows to the upstream LoRA config."""

    token_ids = _catan_token_ids(tokenizer, catan_tokens)
    if not token_ids:
        return

    original_get_peft_model: GetPeftModel = upstream.get_peft_model

    def get_peft_model_with_catan_tokens(
        model: PreTrainedModel,
        peft_config: PeftConfig,
        *args: object,
        **kwargs: object,
    ) -> PeftModel:
        if hasattr(peft_config, "trainable_token_indices"):
            target_names = _language_token_target_names(model)
            peft_config.trainable_token_indices = {
                target_name: token_ids for target_name in target_names
            }
            print(f"catan_trainable_token_indices={len(token_ids)} " f"on {','.join(target_names)}")
        else:
            print("warning=trainable_token_indices_not_supported_by_peft")
        return original_get_peft_model(model, peft_config, *args, **kwargs)

    setattr(upstream, "get_peft_model", get_peft_model_with_catan_tokens)


def _patch_peft_for_catan_tokens_only(
    upstream: ModuleType,
    tokenizer: PreTrainedTokenizerBase,
    catan_tokens: list[str],
) -> None:
    """Replace broad LoRA with standalone selective token-row PEFT."""

    token_ids = _catan_token_ids(tokenizer, catan_tokens)
    if not token_ids:
        raise RuntimeError("No Catan token IDs were available for token-only PEFT")

    peft = importlib.import_module("peft")
    original_get_peft_model: GetPeftModel = upstream.get_peft_model

    def get_peft_model_with_catan_tokens_only(
        model: PreTrainedModel,
        _peft_config: PeftConfig,
        *args: object,
        **kwargs: object,
    ) -> PeftModel:
        target_names = _language_token_target_names(model)
        token_config = peft.TrainableTokensConfig(
            target_modules=target_names,
            token_indices=token_ids,
            init_weights=True,
        )
        print(f"catan_token_adapter_only={len(token_ids)} " f"on {','.join(target_names)}")
        return original_get_peft_model(model, token_config, *args, **kwargs)

    setattr(upstream, "get_peft_model", get_peft_model_with_catan_tokens_only)
