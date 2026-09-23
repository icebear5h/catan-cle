"""Evaluate a Qwen-VL adapter on local Catan SFT JSONL rows."""

from __future__ import annotations

import argparse as argparse
import importlib as importlib
import json as json
import math as math
import re as re
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from contextlib import nullcontext as nullcontext
from datetime import datetime as datetime
from datetime import timezone as timezone
from pathlib import Path as Path
from typing import Any as Any

from safetensors import safe_open as safe_open

from sft.analysis.behavior_diagnostics import summarize_behaviors as summarize_behaviors
from sft.board.board_fluency_scoring import SCHEMA as SCHEMA
from sft.board.board_fluency_scoring import score_board_fluency as score_board_fluency
from sft.board.board_fluency_scoring import (
    validate_board_fluency_metadata as validate_board_fluency_metadata,
)
from sft.board.coordinate_comparison import paired_summary as paired_summary
from sft.board.coordinate_comparison import (
    score_coordinate_comparison as score_coordinate_comparison,
)
from sft.board.coordinate_comparison import (
    validate_comparison_metadata as validate_comparison_metadata,
)
from sft.board.coordinate_comparison import validate_comparison_rows as validate_comparison_rows
from sft.board.spatial_tasks import score_spatial_task as score_spatial_task
from sft.board.symbolic_board_tasks import SYMBOLIC_TASKS as SYMBOLIC_TASKS
from sft.board.symbolic_board_tasks import score_symbolic_task as score_symbolic_task
from sft.board.symbolic_board_tasks import symbolic_task_role as symbolic_task_role
from sft.board_state_readout import score_board_state as score_board_state
from sft.board_state_readout import summarize_board_states as summarize_board_states
from sft.paths import resolve_dataset_asset as resolve_dataset_asset
from sft.paths import resolve_dataset_image as resolve_dataset_image
from sft.scripts.train.train_trl_catan_vision import INPUT_MODES as INPUT_MODES
from sft.scripts.train.train_trl_catan_vision import RUN_CONFIG_FILE as RUN_CONFIG_FILE
from sft.scripts.train.train_trl_catan_vision import VISUAL_STATE_FILE as VISUAL_STATE_FILE
from sft.scripts.train.train_trl_catan_vision import _message_pair as _message_pair
from sft.scripts.train.train_trl_catan_vision import (
    assert_runtime_versions as assert_runtime_versions,
)
from sft.scripts.train.train_trl_catan_vision import encode_text_pair as encode_text_pair
from sft.scripts.train.train_trl_catan_vision import freeze_visual_weights as freeze_visual_weights
from sft.scripts.train.train_trl_catan_vision import (
    load_checkpoint_text_tokenizer as load_checkpoint_text_tokenizer,
)
from sft.scripts.train.train_trl_catan_vision import load_token_inventory as load_token_inventory
from sft.scripts.train.train_trl_catan_vision import load_visual_state as load_visual_state
from sft.scripts.train.train_trl_catan_vision import native_tokenizer as native_tokenizer
from sft.scripts.train.train_trl_catan_vision import pad_text_inputs as pad_text_inputs
from sft.scripts.train.train_trl_catan_vision import (
    prepare_semantic_tokens as prepare_semantic_tokens,
)
from sft.scripts.train.train_trl_catan_vision import (
    resolve_wrapped_module as resolve_wrapped_module,
)
from sft.scripts.train.train_trl_catan_vision import text_chat_ids as text_chat_ids
from sft.scripts.train.train_trl_catan_vision import validate_text_adapter as validate_text_adapter
from sft.scripts.train.train_trl_catan_vision import validate_text_budget as validate_text_budget
from sft.scripts.train.train_trl_catan_vision import (
    validate_text_context_budget as validate_text_context_budget,
)

from ._candidates import ATLAS_TOKEN_RE as ATLAS_TOKEN_RE
from ._candidates import ENTITY_PREFIXES as ENTITY_PREFIXES
from ._candidates import IMAGE_VARIANTS as IMAGE_VARIANTS
from ._candidates import MARKER_LETTERS as MARKER_LETTERS
from ._candidates import _shuffled_image_map as _shuffled_image_map
from ._candidates import _spatial_target as _spatial_target
from ._candidates import build_qwen_messages as build_qwen_messages
from ._candidates import candidate_answers as candidate_answers
from ._candidates import evaluation_image as evaluation_image
from ._candidates import generate_responses as generate_responses
from ._candidates import image_reference as image_reference
from ._cli import main as main
from ._cli import parse_args as parse_args
from ._cli import run_eval as run_eval
from ._jobs import eval_jobs as eval_jobs
from ._jobs import run_eval_job as run_eval_job
from ._model import load_model as load_model
from ._model import load_non_lora_adapter_weights as load_non_lora_adapter_weights
from ._model import normalize_non_lora_state_dict as normalize_non_lora_state_dict
from ._reporting import BOARD_FLUENCY_SCHEMAS as BOARD_FLUENCY_SCHEMAS
from ._reporting import COORDINATE_COMPARISON_SCHEMA as COORDINATE_COMPARISON_SCHEMA
from ._reporting import OCCUPANCY_CATEGORIES as OCCUPANCY_CATEGORIES
from ._reporting import SPATIAL_TASK_TYPES as SPATIAL_TASK_TYPES
from ._reporting import evaluation_metadata as evaluation_metadata
from ._reporting import occupancy_class as occupancy_class
from ._reporting import summarize as summarize
from ._reporting import summarize_occupancy_classes as summarize_occupancy_classes
from ._reporting import summarize_readout_items as summarize_readout_items
from ._scoring import BOARD_FLUENCY_SCHEMA as BOARD_FLUENCY_SCHEMA
from ._scoring import FULL_BOARD_TASK as FULL_BOARD_TASK
from ._scoring import LONG_ANSWER_CHARACTERS as LONG_ANSWER_CHARACTERS
from ._scoring import LONG_ANSWER_TASK_TYPES as LONG_ANSWER_TASK_TYPES
from ._scoring import READOUT_ITEM_RE as READOUT_ITEM_RE
from ._scoring import candidate_token_ids as candidate_token_ids
from ._scoring import expected_text as expected_text
from ._scoring import extract_json_object as extract_json_object
from ._scoring import is_long_answer as is_long_answer
from ._scoring import iter_jsonl as iter_jsonl
from ._scoring import normalize_text as normalize_text
from ._scoring import readout_items as readout_items
from ._scoring import score_candidates as score_candidates
from ._scoring import score_readout as score_readout
from ._scoring import score_response as score_response
from ._scoring import user_text as user_text
from ._summaries import NEIGHBOR_CONFUSION_BUCKETS as NEIGHBOR_CONFUSION_BUCKETS
from ._summaries import NEIGHBOR_CONFUSION_CATEGORIES as NEIGHBOR_CONFUSION_CATEGORIES
from ._summaries import is_neighbor_confusion_record as is_neighbor_confusion_record
from ._summaries import piece_answer as piece_answer
from ._summaries import summarize_dimension as summarize_dimension
from ._summaries import summarize_neighbor_confusion as summarize_neighbor_confusion
