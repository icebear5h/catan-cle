"""Text-only atlas tasks, detached engine oracles, and recomputing strict scoring.

Metadata target is exactly ``{"state": state_or_null, "query": query}``.
A state contains only ``board`` (the canonical 155-entry board_answer string)
and four participant ``colors``. Geometry lives in this module, never in model
input. Known malformed targets raise ValueError; cached expected text is ignored.
Settlement and edge-simple longest-trail tasks are compositional transfer only.
"""

from __future__ import annotations

import copy as copy
import json as json
from collections import deque as deque
from functools import lru_cache as lru_cache

from cle.game_engine.models.board import STATIC_GRAPH as STATIC_GRAPH
from cle.game_engine.models.board import Board as Board
from cle.game_engine.models.enums import CITY as CITY
from cle.game_engine.models.enums import RESOURCES as RESOURCES
from cle.game_engine.models.enums import SETTLEMENT as SETTLEMENT
from cle.game_engine.models.map import CatanMap as CatanMap
from cle.game_engine.models.player import Color as Color
from cle.game_engine.public_board import (
    PUBLIC_BOARD_CONTRACT_SCHEMA as PUBLIC_BOARD_CONTRACT_SCHEMA,
)
from data_pipeline.board_recognition.full_board_readout import board_answer as board_answer
from evals.catan_board_bench.tokens import atlas_metadata as atlas_metadata
from evals.catan_board_bench.tokens import atlas_tokens as atlas_tokens
from evals.catan_board_bench.tokens import base_catan_map as base_catan_map
from sft.board.symbolic_board_tasks._constants import COLORS as COLORS
from sft.board.symbolic_board_tasks._constants import DIRECTIONS as DIRECTIONS
from sft.board.symbolic_board_tasks._constants import OFFSETS as OFFSETS
from sft.board.symbolic_board_tasks._constants import RESOURCES_LOWER as RESOURCES_LOWER
from sft.board.symbolic_board_tasks._constants import ROUTE_RULES as ROUTE_RULES
from sft.board.symbolic_board_tasks._constants import SET_FORMAT as SET_FORMAT
from sft.board.symbolic_board_tasks._constants import SETTLEMENT_RULES as SETTLEMENT_RULES
from sft.board.symbolic_board_tasks._constants import STATIC_TASKS as STATIC_TASKS
from sft.board.symbolic_board_tasks._constants import SYMBOLIC_TASKS as SYMBOLIC_TASKS
from sft.board.symbolic_board_tasks._constants import TRAIL_RULES as TRAIL_RULES
from sft.board.symbolic_board_tasks._constants import TRAIN_TASKS as TRAIN_TASKS
from sft.board.symbolic_board_tasks._constants import TRANSFER_TASKS as TRANSFER_TASKS
from sft.board.symbolic_board_tasks._constants import PhysicalStateError as PhysicalStateError
from sft.board.symbolic_board_tasks._constants import _bad_number as _bad_number
from sft.board.symbolic_board_tasks._constants import _keys as _keys
from sft.board.symbolic_board_tasks._constants import _require as _require
from sft.board.symbolic_board_tasks._constants import _unique_object as _unique_object
from sft.board.symbolic_board_tasks._constants import strict_json as strict_json
from sft.board.symbolic_board_tasks._contracts import _decode as _decode
from sft.board.symbolic_board_tasks._contracts import decode_state as decode_state
from sft.board.symbolic_board_tasks._contracts import detached_board as detached_board
from sft.board.symbolic_board_tasks._contracts import validate_contract as validate_contract
from sft.board.symbolic_board_tasks._geometry import _atlas as _atlas
from sft.board.symbolic_board_tasks._geometry import _participants as _participants
from sft.board.symbolic_board_tasks._geometry import _rows as _rows
from sft.board.symbolic_board_tasks._geometry import _same as _same
from sft.board.symbolic_board_tasks._geometry import _token as _token
from sft.board.symbolic_board_tasks._geometry import atlas_geometry as atlas_geometry
from sft.board.symbolic_board_tasks._scoring import _route_valid as _route_valid
from sft.board.symbolic_board_tasks._scoring import score_symbolic_task as score_symbolic_task
from sft.board.symbolic_board_tasks._scoring import symbolic_answer as symbolic_answer
from sft.board.symbolic_board_tasks._scoring import symbolic_prompt as symbolic_prompt
from sft.board.symbolic_board_tasks._solve import _near as _near
from sft.board.symbolic_board_tasks._solve import _relation as _relation
from sft.board.symbolic_board_tasks._solve import _solve as _solve
from sft.board.symbolic_board_tasks._solve import _transfer_facts as _transfer_facts
from sft.board.symbolic_board_tasks._solve import owned_route as owned_route
from sft.board.symbolic_board_tasks._solve import symbolic_task_role as symbolic_task_role

__all__: list[str] = [
    "Board",
    "CITY",
    "COLORS",
    "CatanMap",
    "Color",
    "DIRECTIONS",
    "OFFSETS",
    "PUBLIC_BOARD_CONTRACT_SCHEMA",
    "PhysicalStateError",
    "RESOURCES",
    "RESOURCES_LOWER",
    "ROUTE_RULES",
    "SETTLEMENT",
    "SETTLEMENT_RULES",
    "SET_FORMAT",
    "STATIC_GRAPH",
    "STATIC_TASKS",
    "SYMBOLIC_TASKS",
    "TRAIL_RULES",
    "TRAIN_TASKS",
    "TRANSFER_TASKS",
    "annotations",
    "atlas_geometry",
    "atlas_metadata",
    "atlas_tokens",
    "base_catan_map",
    "board_answer",
    "copy",
    "decode_state",
    "deque",
    "detached_board",
    "json",
    "lru_cache",
    "owned_route",
    "score_symbolic_task",
    "strict_json",
    "symbolic_answer",
    "symbolic_prompt",
    "symbolic_task_role",
    "validate_contract",
]
