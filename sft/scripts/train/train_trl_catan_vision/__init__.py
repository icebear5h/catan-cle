"""Native Hugging Face TRL/PEFT training for Catan board grounding.

This module owns the complete model-side contract: dataset normalization,
curriculum-order validation, semantic token insertion, PEFT wrapping, optimizer
groups, checkpoint serialization, reload validation, and optional Hub upload.
It intentionally has no Modal or ms-swift dependency.
"""

from __future__ import annotations

import argparse as argparse
import gc as gc
import hashlib as hashlib
import importlib as importlib
import json as json
import math as math
import os as os
import shutil as shutil
import types as types
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor
from contextlib import contextmanager as contextmanager
from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from pathlib import Path as Path
from typing import Any as Any
from typing import Callable as Callable
from typing import Iterable as Iterable
from typing import Iterator as Iterator
from typing import Sequence as Sequence

import torch as torch
from huggingface_hub import HfApi as HfApi
from peft import LoraConfig as LoraConfig
from peft import PeftModel as PeftModel
from peft import TrainableTokensConfig as TrainableTokensConfig
from peft import get_peft_model as get_peft_model
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer as TrainableTokensLayer
from peft.tuners.tuners_utils import BaseTunerLayer as BaseTunerLayer
from peft.tuners.tuners_utils import check_target_module_exists as check_target_module_exists
from peft.utils.other import TrainableTokensWrapper as TrainableTokensWrapper
from safetensors.torch import load_file as load_file
from safetensors.torch import save_file as save_file

from evals.catan_board_bench.tokens import (
    semantic_recognition_token_inventory as semantic_recognition_token_inventory,
)
from sft.board_state_readout import score_board_state as score_board_state
from sft.lora_expansion import SUPPORTED_TEXT_LORA_RANKS as SUPPORTED_TEXT_LORA_RANKS
from sft.lora_expansion import expected_adapter_shapes as expected_adapter_shapes
from sft.lora_expansion import standard_lora_rank as standard_lora_rank
from sft.lora_expansion import tensor_headers as tensor_headers

from ._bundles import load_frozen_bundle as load_frozen_bundle
from ._bundles import load_initial_bundle as load_initial_bundle
from ._bundles import wrap_trainable_model as wrap_trainable_model
from ._common import ACCELERATE_VERSION as ACCELERATE_VERSION
from ._common import ANSWER_METRIC_TRIVIAL_TOKENS as ANSWER_METRIC_TRIVIAL_TOKENS
from ._common import CURRICULUM_STAGES as CURRICULUM_STAGES
from ._common import DATASET_REPORT_FILE as DATASET_REPORT_FILE
from ._common import DATASETS_VERSION as DATASETS_VERSION
from ._common import FAMILY_WORDS as FAMILY_WORDS
from ._common import FROZEN_ADAPTER_DIR as FROZEN_ADAPTER_DIR
from ._common import FROZEN_BUNDLE_FILE as FROZEN_BUNDLE_FILE
from ._common import HUB_MODEL_ID as HUB_MODEL_ID
from ._common import HUGGINGFACE_HUB_VERSION as HUGGINGFACE_HUB_VERSION
from ._common import IMAGE_HASH_WORKERS as IMAGE_HASH_WORKERS
from ._common import INITIAL_BUNDLE_FILE as INITIAL_BUNDLE_FILE
from ._common import INPUT_MODES as INPUT_MODES
from ._common import LORA_PROFILES as LORA_PROFILES
from ._common import MAX_ANSWER_CHARACTERS as MAX_ANSWER_CHARACTERS
from ._common import MAX_PROMPT_CHARACTERS as MAX_PROMPT_CHARACTERS
from ._common import MODEL_ID as MODEL_ID
from ._common import OPTIMIZER_COVERAGE_FILE as OPTIMIZER_COVERAGE_FILE
from ._common import PATCH_METRICS_FILE as PATCH_METRICS_FILE
from ._common import PEFT_VERSION as PEFT_VERSION
from ._common import PILLOW_VERSION as PILLOW_VERSION
from ._common import PROCESSOR_ASSET_FILES as PROCESSOR_ASSET_FILES
from ._common import PROFILE_OLORA_FROZEN_BUNDLE as PROFILE_OLORA_FROZEN_BUNDLE
from ._common import PROFILE_VISION_TOKENS as PROFILE_VISION_TOKENS
from ._common import PROFILE_VISION_TOKENS_LORA as PROFILE_VISION_TOKENS_LORA
from ._common import PROFILES as PROFILES
from ._common import RELOAD_REPORT_FILE as RELOAD_REPORT_FILE
from ._common import RUN_CONFIG_FILE as RUN_CONFIG_FILE
from ._common import SAFETENSORS_VERSION as SAFETENSORS_VERSION
from ._common import SEMANTIC_ROW_NOISE_SCALE as SEMANTIC_ROW_NOISE_SCALE
from ._common import SPATIAL_TARGET_MODES as SPATIAL_TARGET_MODES
from ._common import TEXT_MEDIA_KEYS as TEXT_MEDIA_KEYS
from ._common import TOKEN_INIT_MODES as TOKEN_INIT_MODES
from ._common import TORCH_VERSION as TORCH_VERSION
from ._common import TORCHVISION_VERSION as TORCHVISION_VERSION
from ._common import TRAINABLE_SCOPE_FILE as TRAINABLE_SCOPE_FILE
from ._common import TRANSFORMERS_VERSION as TRANSFORMERS_VERSION
from ._common import TRL_VERSION as TRL_VERSION
from ._common import VISION_LORA_SUFFIXES as VISION_LORA_SUFFIXES
from ._common import VISUAL_STATE_FILE as VISUAL_STATE_FILE
from ._common import JsonDict as JsonDict
from ._common import inventory_tokens as inventory_tokens
from ._common import iter_jsonl as iter_jsonl
from ._common import load_token_inventory as load_token_inventory
from ._common import sha256_file as sha256_file
from ._common import write_json_atomic as write_json_atomic
from ._config import ModelComponents as ModelComponents
from ._config import TokenSetup as TokenSetup
from ._config import TrainConfig as TrainConfig
from ._config import native_tokenizer as native_tokenizer
from ._config import normalize_training_config as normalize_training_config
from ._config import validate_resume_mode as validate_resume_mode
from ._config import validate_text_budget as validate_text_budget
from ._datasets import load_text_dataset as load_text_dataset
from ._datasets import load_training_dataset as load_training_dataset
from ._datasets import load_vision_dataset as load_vision_dataset
from ._frozen import apply_frozen_adapter as apply_frozen_adapter
from ._frozen import carry_processor_assets as carry_processor_assets
from ._frozen import freeze_visual_except_lora as freeze_visual_except_lora
from ._frozen import freeze_visual_weights as freeze_visual_weights
from ._frozen import frozen_adapter_files as frozen_adapter_files
from ._frozen import load_checkpoint_text_tokenizer as load_checkpoint_text_tokenizer
from ._frozen import load_visual_state as load_visual_state
from ._frozen import processor_asset_hashes as processor_asset_hashes
from ._frozen import save_visual_state as save_visual_state
from ._frozen import validate_checkpoint_tokenizer as validate_checkpoint_tokenizer
from ._frozen import validate_text_adapter as validate_text_adapter
from ._frozen import validate_text_context_budget as validate_text_context_budget
from ._model_tokens import answer_token_metrics as answer_token_metrics
from ._model_tokens import initialize_semantic_token_rows as initialize_semantic_token_rows
from ._model_tokens import language_linear_targets as language_linear_targets
from ._model_tokens import output_head_weight as output_head_weight
from ._model_tokens import prepare_semantic_tokens as prepare_semantic_tokens
from ._model_tokens import promote_visual_master_weights as promote_visual_master_weights
from ._model_tokens import trivial_completion_token_ids as trivial_completion_token_ids
from ._model_tokens import vision_linear_targets as vision_linear_targets
from ._optim import audit_optimizer_coverage as audit_optimizer_coverage
from ._optim import audit_trainable_scope as audit_trainable_scope
from ._optim import build_optimizer as build_optimizer
from ._optim import parameter_category as parameter_category
from ._run import main as main
from ._run import parse_args as parse_args
from ._run import run_training as run_training
from ._sft import _load_base_model_and_processor as _load_base_model_and_processor
from ._sft import _write_model_card as _write_model_card
from ._sft import assert_runtime_versions as assert_runtime_versions
from ._sft import build_sft_config as build_sft_config
from ._sft import publish_bundle as publish_bundle
from ._sft import validate_saved_bundle as validate_saved_bundle
from ._structure import _ChunkedNLLTrainableTokensHead as _ChunkedNLLTrainableTokensHead
from ._structure import _matches_path as _matches_path
from ._structure import discover_components as discover_components
from ._structure import (
    expose_trainable_tokens_head_to_chunked_nll as expose_trainable_tokens_head_to_chunked_nll,
)
from ._structure import module_name_for_instance as module_name_for_instance
from ._structure import resolve_wrapped_module as resolve_wrapped_module
from ._text_data import SpatialTargetCollator as SpatialTargetCollator
from ._text_data import TextCompletionCollator as TextCompletionCollator
from ._text_data import _content_text as _content_text
from ._text_data import _message_pair as _message_pair
from ._text_data import _text_content as _text_content
from ._text_data import encode_text_pair as encode_text_pair
from ._text_data import pad_text_inputs as pad_text_inputs
from ._text_data import text_chat_ids as text_chat_ids
from ._trainer import _trainer_class as _trainer_class
from ._vision_data import _image_reference as _image_reference
from ._vision_data import inspect_jsonl_contract as inspect_jsonl_contract
from ._vision_data import resolve_image_path as resolve_image_path
from ._vision_data import validate_spatial_targets as validate_spatial_targets
from ._visual import LanguageHiddenCapture as LanguageHiddenCapture
from ._visual import VisionPoolerCapture as VisionPoolerCapture
from ._visual import _merged_patch_counts as _merged_patch_counts
from ._visual import load_protected_bases as load_protected_bases
from ._visual import module_key as module_key
from ._visual import orthogonal_penalty as orthogonal_penalty
from ._visual import orthonormal_rows as orthonormal_rows
from ._visual import soft_patch_target as soft_patch_target
from ._visual import spatial_patch_loss as spatial_patch_loss
from ._visual import split_merged_visual_features as split_merged_visual_features
