"""Catan board observation contracts and benchmark generation.

The core idea is that the engine remains the oracle. This package turns a live
``GameEngine`` into a public board contract, deterministic question/answer items,
and optionally 512x512 board images rendered through the existing frontend. The
question groups, replay loading, and selection helpers live in sibling modules."""

from __future__ import annotations
from __future__ import annotations as annotations

import asyncio as asyncio
import contextlib as contextlib
import importlib as importlib
import io as io
import json as json
import math as math
from collections import Counter as Counter
from dataclasses import dataclass as dataclass
from datetime import datetime as datetime
from datetime import timezone as timezone
from pathlib import Path as Path
from typing import Any as Any
from typing import Dict as Dict
from typing import Iterable as Iterable
from typing import List as List
from typing import Optional as Optional
from typing import Sequence as Sequence
from typing import Tuple as Tuple

# Names the pre-split module also exposed, kept importable at this path.
from PIL import Image as Image
from PIL import ImageOps as ImageOps

from cle.game_engine.game import GameEngine as GameEngine
from cle.game_engine.models.board import get_edges as get_edges
from cle.game_engine.models.enums import CITY as CITY
from cle.game_engine.models.enums import SETTLEMENT as SETTLEMENT
from cle.game_engine.models.map import NUM_EDGES as NUM_EDGES
from cle.game_engine.models.map import NUM_NODES as NUM_NODES
from cle.game_engine.models.map import PORT_DIRECTION_TO_NODEREFS as PORT_DIRECTION_TO_NODEREFS
from cle.game_engine.models.map import CatanMap as CatanMap
from cle.game_engine.models.map import LandTile as LandTile
from cle.game_engine.models.map import Port as Port
from cle.game_engine.models.player import Color as Color
from cle.game_engine.state_functions import get_largest_army as get_largest_army
from cle.game_engine.state_functions import get_longest_road_color as get_longest_road_color
from cle.game_engine.state_functions import get_longest_road_length as get_longest_road_length
from cle.game_engine.state_functions import get_played_dev_cards as get_played_dev_cards
from cle.game_engine.state_functions import get_visible_victory_points as get_visible_victory_points
from cle.players.legacy import SimplePlayer as SimplePlayer
from cle.replay.colonist.coordinates import create_map_from_colonist as create_map_from_colonist
from cle.replay.colonist.event_parser import (
    parse_colonist_events_to_actions as parse_colonist_events_to_actions,
)
from cle.replay.runtime.step_executor import replay_step_logic as replay_step_logic
from evals.catan_board_bench.builder.benchmark_builder import (
    CatanBoardBenchBuilder as CatanBoardBenchBuilder,
)
from evals.catan_board_bench.builder.benchmark_builder import (
    build_catan_board_bench_sync as build_catan_board_bench_sync,
)
from evals.catan_board_bench.builder.constants import COLONIST_COLOR_NAMES as COLONIST_COLOR_NAMES
from evals.catan_board_bench.builder.constants import (
    COLONIST_TO_ENGINE_COLOR as COLONIST_TO_ENGINE_COLOR,
)
from evals.catan_board_bench.builder.constants import DEFAULT_OUTPUT_DIR as DEFAULT_OUTPUT_DIR
from evals.catan_board_bench.builder.constants import DEFAULT_QUESTION_DIR as DEFAULT_QUESTION_DIR
from evals.catan_board_bench.builder.constants import DEFAULT_REPLAY_DIR as DEFAULT_REPLAY_DIR
from evals.catan_board_bench.builder.constants import (
    FALLBACK_ENGINE_COLORS as FALLBACK_ENGINE_COLORS,
)
from evals.catan_board_bench.builder.constants import BenchmarkBuildResult as BenchmarkBuildResult
from evals.catan_board_bench.builder.constants import EdgeId as EdgeId
from evals.catan_board_bench.builder.constants import JsonDict as JsonDict
from evals.catan_board_bench.builder.records import _answer_view as _answer_view
from evals.catan_board_bench.builder.records import _question_view as _question_view
from evals.catan_board_bench.builder.records import _quiet_context as _quiet_context
from evals.catan_board_bench.builder.records import _write_json as _write_json
from evals.catan_board_bench.builder.records import _write_jsonl as _write_jsonl
from evals.catan_board_bench.builder.records import _write_square_png as _write_square_png
from evals.catan_board_bench.builder.replay import _project_relative_path as _project_relative_path
from evals.catan_board_bench.builder.replay import load_colonist_replay as load_colonist_replay
from evals.catan_board_bench.builder.replay import step_replay as step_replay
from evals.catan_board_bench.builder.selection import (
    _disconnected_node_pair as _disconnected_node_pair,
)
from evals.catan_board_bench.builder.selection import _find_by_id as _find_by_id
from evals.catan_board_bench.builder.selection import _pick as _pick
from evals.catan_board_bench.builder.selection import (
    _port_occupancy_answer as _port_occupancy_answer,
)
from evals.catan_board_bench.builder.selection import (
    _port_occupied_nodes_target as _port_occupied_nodes_target,
)
from evals.catan_board_bench.builder.selection import _positive_first as _positive_first
from evals.catan_board_bench.builder.selection import (
    _resource_number_answer as _resource_number_answer,
)
from evals.catan_board_bench.builder.selection import _road_edges_for_color as _road_edges_for_color
from evals.catan_board_bench.builder.selection import _spaced_steps as _spaced_steps
from evals.catan_board_bench.builder.selection import (
    _tile_occupied_nodes_answer as _tile_occupied_nodes_answer,
)
from evals.catan_board_bench.builder.selection import (
    _tile_occupied_nodes_target as _tile_occupied_nodes_target,
)
from evals.catan_board_bench.builder.suite import CatanObservationSuite as CatanObservationSuite
from evals.catan_board_bench.builder.topology import _node_ports as _node_ports
from evals.catan_board_bench.builder.topology import _owned_road_count as _owned_road_count
from evals.catan_board_bench.builder.topology import _playable_edges as _playable_edges
from evals.catan_board_bench.builder.topology import _port_coordinates as _port_coordinates
from evals.catan_board_bench.builder.topology import _port_nodes_by_id as _port_nodes_by_id
from evals.catan_board_bench.builder.topology import _tile_coordinates as _tile_coordinates
from evals.catan_board_bench.paths import DATASETS_DIR as DATASETS_DIR
from evals.catan_board_bench.paths import PROJECT_ROOT as PROJECT_ROOT
from evals.catan_board_bench.tokens import base_edges as base_edges
from evals.catan_board_bench.tokens import building_token as building_token
from evals.catan_board_bench.tokens import canonical_edge as canonical_edge
from evals.catan_board_bench.tokens import color_token as color_token
from evals.catan_board_bench.tokens import edge_token as edge_token
from evals.catan_board_bench.tokens import node_token as node_token
from evals.catan_board_bench.tokens import object_token as object_token
from evals.catan_board_bench.tokens import port_token as port_token
from evals.catan_board_bench.tokens import resource_token as resource_token
from evals.catan_board_bench.tokens import tile_token as tile_token
from playground.game_viewer.state import ServerState as ServerState
