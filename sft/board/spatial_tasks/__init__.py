"""Canonical atlas solvers and strict, task-specific spatial answer scoring."""

from __future__ import annotations

import json as json
from collections import deque as deque
from functools import lru_cache as lru_cache

from cle.game_engine.models.enums import CITY as CITY
from cle.game_engine.models.enums import RESOURCES as RESOURCES
from cle.game_engine.models.enums import SETTLEMENT as SETTLEMENT
from cle.game_engine.models.player import Color as Color
from cle.game_engine.public_board import (
    PUBLIC_BOARD_CONTRACT_SCHEMA as PUBLIC_BOARD_CONTRACT_SCHEMA,
)
from evals.catan_board_bench.tokens import atlas_metadata as atlas_metadata
from sft.board.spatial_tasks._contracts import _check_color_roll as _check_color_roll
from sft.board.spatial_tasks._contracts import _check_ids as _check_ids
from sft.board.spatial_tasks._contracts import _check_resource_number as _check_resource_number
from sft.board.spatial_tasks._contracts import _contract_tiles as _contract_tiles
from sft.board.spatial_tasks._contracts import _indexed_rows as _indexed_rows
from sft.board.spatial_tasks._contracts import dice_production as dice_production
from sft.board.spatial_tasks._contracts import local_node_tiles as local_node_tiles
from sft.board.spatial_tasks._scoring import _json_answer as _json_answer
from sft.board.spatial_tasks._scoring import _reject_json_number as _reject_json_number
from sft.board.spatial_tasks._scoring import _token_answer as _token_answer
from sft.board.spatial_tasks._scoring import _unique_object as _unique_object
from sft.board.spatial_tasks._scoring import _valid_path as _valid_path
from sft.board.spatial_tasks._scoring import score_spatial_task as score_spatial_task
from sft.board.spatial_tasks._topology import COLORS as COLORS
from sft.board.spatial_tasks._topology import RESOURCE_KEYS as RESOURCE_KEYS
from sft.board.spatial_tasks._topology import TASK_TYPES as TASK_TYPES
from sft.board.spatial_tasks._topology import _topology as _topology
from sft.board.spatial_tasks._topology import atlas_node_graph as atlas_node_graph
from sft.board.spatial_tasks._topology import node_tile_tokens as node_tile_tokens
from sft.board.spatial_tasks._topology import shortest_node_path as shortest_node_path

__all__: list[str] = [
    "CITY",
    "COLORS",
    "Color",
    "PUBLIC_BOARD_CONTRACT_SCHEMA",
    "RESOURCES",
    "RESOURCE_KEYS",
    "SETTLEMENT",
    "TASK_TYPES",
    "annotations",
    "atlas_metadata",
    "atlas_node_graph",
    "deque",
    "dice_production",
    "json",
    "local_node_tiles",
    "lru_cache",
    "node_tile_tokens",
    "score_spatial_task",
    "shortest_node_path",
]
