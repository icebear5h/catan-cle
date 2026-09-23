"""Schemas, versions and dataclasses shared by the ms-swift integration."""

from __future__ import annotations

import json as json
from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from pathlib import Path as Path
from typing import Any as Any
from typing import Iterable as Iterable

import torch as torch

from evals.catan_board_bench.tokens import (
    semantic_recognition_token_inventory as semantic_recognition_token_inventory,
)
from sft.json_types import JsonDict as JsonDict

MS_SWIFT_VERSION = "4.5.2"

PEFT_VERSION = "0.17.1"

CATAN_TUNER_TYPE = "catan_vision_tokens"

COMPONENT_SCHEMA = "catan_ms_swift_model_components/v1"

SCOPE_SCHEMA = "catan_ms_swift_trainable_scope/v2"

OPTIMIZER_COVERAGE_SCHEMA = "catan_ms_swift_optimizer_coverage/v1"

_COMPONENT_ATTRIBUTE = "_catan_ms_swift_model_components"

@dataclass(frozen=True)
class SemanticTokenSetup:
    token_ids: tuple[int, ...]
    original_tokenizer_size: int
    tokenizer_size: int
    model_vocab_size: int
    added_tokens: int

    def as_dict(self) -> JsonDict:
        return asdict(self)

@dataclass(frozen=True)
class ModelComponentPaths:
    """Validated, model-independent paths for one ms-swift multimodal model."""

    architecture: str
    input_embedding: str
    output_head: str
    language: tuple[str, ...]
    vision: tuple[str, ...]
    aligner: tuple[str, ...]
    vocab_size: int
    hidden_size: int

    def as_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["schema"] = COMPONENT_SCHEMA
        return payload
