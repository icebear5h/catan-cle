"""Prepare an immutable, CPU/offline symbolic atlas pilot from full_board_diverse_v1.

CLI: --dry-run audits sources and constructs/scorers all rows without writing;
--validate verifies a previously built output and its pinned source files.
Default execution creates a new output only after validation. No overwrite mode.
Internal build API: build_dataset(output_dir, root=..., seed=..., dry_run=False,
checkpoint=None). The checkpoint is a proposed paired-experiment input, not loaded
or selected by this builder; final training budget/configuration belongs to integration.
"""

from __future__ import annotations

import argparse as argparse
import json as json
import random as random
import re as re
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from itertools import combinations as combinations
from pathlib import Path as Path

from data_pipeline.board_recognition.full_board_readout import board_answer as board_answer
from data_pipeline.board_recognition.replay_dataset import dense_labels as dense_labels
from data_pipeline.board_recognition.replay_dataset import static_board_facts as static_board_facts
from data_pipeline.board_recognition.sources import DEFAULT_LEAKAGE_LEDGER as DEFAULT_LEAKAGE_LEDGER
from data_pipeline.board_recognition.sources import DEFAULT_SOURCE_LOCK as DEFAULT_SOURCE_LOCK
from data_pipeline.board_recognition.sources import canonical_sha256 as canonical_sha256
from data_pipeline.board_recognition.sources import file_sha256 as file_sha256
from data_pipeline.board_recognition.sources import load_leakage_ledger as load_leakage_ledger
from data_pipeline.board_recognition.sources import (
    source_lock_matches_metadata as source_lock_matches_metadata,
)
from data_pipeline.board_recognition.sources import (
    validate_replay_source_lock as validate_replay_source_lock,
)
from data_pipeline.board_recognition.sources import visible_board_facts as visible_board_facts
from evals.catan_board_bench.tokens import atlas_tokens as atlas_tokens
from sft.board.symbolic_board_tasks import DIRECTIONS as DIRECTIONS
from sft.board.symbolic_board_tasks import OFFSETS as OFFSETS
from sft.board.symbolic_board_tasks import RESOURCES_LOWER as RESOURCES_LOWER
from sft.board.symbolic_board_tasks import STATIC_TASKS as STATIC_TASKS
from sft.board.symbolic_board_tasks import TRAIN_TASKS as TRAIN_TASKS
from sft.board.symbolic_board_tasks import TRANSFER_TASKS as TRANSFER_TASKS
from sft.board.symbolic_board_tasks import PhysicalStateError as PhysicalStateError
from sft.board.symbolic_board_tasks import atlas_geometry as atlas_geometry
from sft.board.symbolic_board_tasks import decode_state as decode_state
from sft.board.symbolic_board_tasks import owned_route as owned_route
from sft.board.symbolic_board_tasks import score_symbolic_task as score_symbolic_task
from sft.board.symbolic_board_tasks import strict_json as strict_json
from sft.board.symbolic_board_tasks import symbolic_answer as symbolic_answer
from sft.board.symbolic_board_tasks import symbolic_prompt as symbolic_prompt
from sft.board.symbolic_board_tasks import symbolic_task_role as symbolic_task_role
from sft.board.symbolic_board_tasks import validate_contract as validate_contract

from ._build import _write as _write
from ._build import build_dataset as build_dataset
from ._build import main as main
from ._components import _balanced as _balanced
from ._components import _component_options as _component_options
from ._components import _route_query as _route_query
from ._components import _selectors as _selectors
from ._components import component_slots as component_slots
from ._components import directional_pair_manifest as directional_pair_manifest
from ._components import known_direction_exposure as known_direction_exposure
from ._components import selector_nodes as selector_nodes
from ._components import static_queries as static_queries
from ._sampling import ComponentSampler as ComponentSampler
from ._sampling import _ordered_sources as _ordered_sources
from ._sampling import _row as _row
from ._sampling import component_rows as component_rows
from ._sources import DEFAULT_OUTPUT as DEFAULT_OUTPUT
from ._sources import DEFAULT_ROOT as DEFAULT_ROOT
from ._sources import DENSITIES as DENSITIES
from ._sources import EVAL_QUOTAS as EVAL_QUOTAS
from ._sources import KNOWN_DIRECTION_TRAINING as KNOWN_DIRECTION_TRAINING
from ._sources import PROJECT_ROOT as PROJECT_ROOT
from ._sources import SOURCE_COUNTS as SOURCE_COUNTS
from ._sources import SPLITS as SPLITS
from ._sources import TOKEN_PATTERN as TOKEN_PATTERN
from ._sources import TRAIN_QUOTAS as TRAIN_QUOTAS
from ._sources import VERSION as VERSION
from ._sources import _asset as _asset
from ._sources import _check as _check
from ._sources import _engine_source_digest as _engine_source_digest
from ._sources import _hash as _hash
from ._sources import _unique_pairs as _unique_pairs
from ._sources import audit_source_state as audit_source_state
from ._sources import load_sources as load_sources
from ._sources import read_json as read_json
from ._sources import read_jsonl as read_jsonl
from ._transfer import _diverse_candidates as _diverse_candidates
from ._transfer import graph_case_coverage as graph_case_coverage
from ._transfer import nx as nx
from ._transfer import transfer_projection as transfer_projection
from ._transfer import transfer_rows as transfer_rows
from ._transfer import transfer_weights as transfer_weights
from ._validate import component_profile as component_profile
from ._validate import preserved_v1_artifacts as preserved_v1_artifacts
from ._validate import validate_component_balance as validate_component_balance
from ._validate import validate_dataset as validate_dataset
from ._validate import validate_row_declarations as validate_row_declarations
from ._validate import validate_rows as validate_rows
