"""Compose the approved mixed 128-step continuation from existing local images."""

from __future__ import annotations

import argparse as argparse
import copy as copy
import json as json
import random as random
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from itertools import combinations as combinations
from pathlib import Path as Path

from data_pipeline.board_recognition.full_board_readout import board_answer as board_answer
from data_pipeline.board_recognition.replay_dataset import dense_labels as dense_labels
from data_pipeline.board_recognition.replay_dataset import read_jsonl as read_jsonl
from data_pipeline.board_recognition.replay_dataset import static_board_facts as static_board_facts
from data_pipeline.board_recognition.replay_dataset import write_json as write_json
from data_pipeline.board_recognition.replay_dataset import write_jsonl as write_jsonl
from data_pipeline.board_recognition.sources import canonical_sha256 as canonical_sha256
from data_pipeline.board_recognition.sources import file_sha256 as file_sha256
from data_pipeline.board_recognition.sources import (
    validate_public_board_contract as validate_public_board_contract,
)
from data_pipeline.board_recognition.sources import visible_board_facts as visible_board_facts
from data_pipeline.board_recognition.spatial_robber import spatial_query_bank as spatial_query_bank
from evals.catan_board_bench.tokens import atlas_tokens as atlas_tokens
from sft.board.spatial_tasks import atlas_node_graph as atlas_node_graph
from sft.board.spatial_tasks import dice_production as dice_production
from sft.board.spatial_tasks import local_node_tiles as local_node_tiles
from sft.board.spatial_tasks import node_tile_tokens as node_tile_tokens
from sft.board.spatial_tasks import score_spatial_task as score_spatial_task
from sft.board.spatial_tasks import shortest_node_path as shortest_node_path
from sft.board_state_readout import score_board_state as score_board_state

from ._build import build_dataset as build_dataset
from ._build import main as main
from ._production import _coverage as _coverage
from ._production import _production_options as _production_options
from ._production import _row as _row
from ._production import _select_production as _select_production
from ._queries import _bank_queries as _bank_queries
from ._queries import _path_pairs as _path_pairs
from ._queries import _query as _query
from ._queries import _select_paths as _select_paths
from ._queries import _varied_states as _varied_states
from ._sources import BANK_QUOTAS as BANK_QUOTAS
from ._sources import DEFAULT_OUTPUT as DEFAULT_OUTPUT
from ._sources import DEFAULT_ROOT as DEFAULT_ROOT
from ._sources import FROZEN_BOARD_EVAL as FROZEN_BOARD_EVAL
from ._sources import NEW_PANELS as NEW_PANELS
from ._sources import PROJECT_ROOT as PROJECT_ROOT
from ._sources import STATIC_GUIDANCE as STATIC_GUIDANCE
from ._sources import STEP_QUOTAS as STEP_QUOTAS
from ._sources import VERSION as VERSION
