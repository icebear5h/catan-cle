"""Run the upstream Qwen-VL-Series-Finetune SFT trainer with Catan tokens.

This wrapper intentionally keeps the trainer implementation upstream-owned. It
loads the processor before the upstream training function, adds Catan atlas
tokens, and patches the upstream model loader so token embeddings are resized
before PEFT wraps the model.
"""

from __future__ import annotations

from ._args import _arg_value as _arg_value
from ._args import _pop_flag as _pop_flag
from ._args import _pop_option as _pop_option
from ._args import _validate_profile_arguments as _validate_profile_arguments
from ._audit import _audit_trainable_parameters as _audit_trainable_parameters
from ._audit import _patch_trainer_scope_audit as _patch_trainer_scope_audit
from ._audit import _trainable_parameter_group as _trainable_parameter_group
from ._base import VISION_LANGUAGE_LORA as VISION_LANGUAGE_LORA
from ._base import VISION_ONLY as VISION_ONLY
from ._base import VISION_SFT_PROFILES as VISION_SFT_PROFILES
from ._base import Any as Any
from ._base import Path as Path
from ._base import added_tokens as added_tokens
from ._base import importlib as importlib
from ._base import json as json
from ._base import load_recognition_token_inventory as load_recognition_token_inventory
from ._base import sys as sys
from ._main import main as main
from ._modules import _catan_token_ids as _catan_token_ids
from ._modules import _find_language_embed_tokens_name as _find_language_embed_tokens_name
from ._modules import _find_module_name as _find_module_name
from ._modules import _language_token_target_names as _language_token_target_names
from ._peft import _patch_peft_for_catan_tokens as _patch_peft_for_catan_tokens
from ._peft import _patch_peft_for_catan_tokens_only as _patch_peft_for_catan_tokens_only
from ._savers import _patch_non_deepspeed_savers as _patch_non_deepspeed_savers
from ._savers import (
    _resize_token_embeddings_without_shrinking as _resize_token_embeddings_without_shrinking,
)

__all__ = [
    "Any",
    "Path",
    "VISION_LANGUAGE_LORA",
    "VISION_ONLY",
    "VISION_SFT_PROFILES",
    "added_tokens",
    "importlib",
    "json",
    "load_recognition_token_inventory",
    "main",
    "sys",
]
