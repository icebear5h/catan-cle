"""Build post-atlas node visual-grounding data.

The purpose of this dataset is not to imitate real games or teach the atlas from
scratch. It assumes the fixed Catan atlas already exists, then trains visual
readout around stable atlas handles:

    stable token: <N41>
    transient visual state: EMPTY / <RED> <SETTLEMENT> / <BLUE> <CITY> / ...
    varied context: shuffled tile resources, numbers, ports, robber, and roads

This guards against the bad shortcut where a model learns that a node token
permanently "has" a piece. Each atlas node appears under many contradictory
transient states.
"""

from __future__ import annotations

import argparse as argparse
import json as json
import random as random
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from datetime import datetime as datetime
from datetime import timezone as timezone
from pathlib import Path as Path
from typing import Any as Any

from cle.game_engine.models.enums import CITY as CITY
from cle.game_engine.models.enums import SETTLEMENT as SETTLEMENT
from cle.game_engine.models.player import Color as Color
from evals.catan_board_bench.annotations import (
    annotation_payload_for_contract as annotation_payload_for_contract,
)
from evals.catan_board_bench.tokens import atlas_metadata as atlas_metadata
from evals.catan_board_bench.tokens import building_token as building_token
from evals.catan_board_bench.tokens import canonical_edge as canonical_edge
from evals.catan_board_bench.tokens import color_token as color_token
from evals.catan_board_bench.tokens import edge_token as edge_token
from evals.catan_board_bench.tokens import node_token as node_token
from evals.catan_board_bench.tokens import object_token as object_token
from evals.catan_board_bench.tokens import port_token as port_token
from evals.catan_board_bench.tokens import resource_token as resource_token
from evals.catan_board_bench.tokens import tile_token as tile_token
from sft.paths import GENERATED_SFT_ROOT as GENERATED_SFT_ROOT

from ._build import main as main
from ._build import parse_args as parse_args
from ._contract import build_contract as build_contract
from ._qas import adjacent_tile_answer as adjacent_tile_answer
from ._qas import board_atlas_bbox_payload as board_atlas_bbox_payload
from ._qas import build_qas as build_qas
from ._qas import compact_json as compact_json
from ._qas import find_annotation as find_annotation
from ._qas import incident_road_answer as incident_road_answer
from ._qas import local_edge_payloads as local_edge_payloads
from ._qas import local_port_payloads as local_port_payloads
from ._qas import local_state_payload as local_state_payload
from ._qas import local_tile_payloads as local_tile_payloads
from ._qas import local_tiles as local_tiles
from ._qas import occupancy_answer as occupancy_answer
from ._qas import tile_phrase as tile_phrase
from ._sources import CATEGORY_GROUPS as CATEGORY_GROUPS
from ._sources import CURRICULUM_STAGE as CURRICULUM_STAGE
from ._sources import DATASET_NAME as DATASET_NAME
from ._sources import DATASET_ROLE as DATASET_ROLE
from ._sources import DATASET_SCHEMA as DATASET_SCHEMA
from ._sources import DEFAULT_COLORS as DEFAULT_COLORS
from ._sources import NUMBER_DECK as NUMBER_DECK
from ._sources import PORT_DECK as PORT_DECK
from ._sources import PROMPT_PREFIX as PROMPT_PREFIX
from ._sources import REQUIRES_STAGE as REQUIRES_STAGE
from ._sources import RESOURCE_DECK as RESOURCE_DECK
from ._sources import AtlasIndices as AtlasIndices
from ._sources import build_indices as build_indices
from ._sources import category_group as category_group
from ._sources import occupancy_cases as occupancy_cases
from ._sources import validate_colors as validate_colors
from ._variants import choose_distractor_buildings as choose_distractor_buildings
from ._variants import choose_roads as choose_roads
from ._variants import player_summaries as player_summaries
from ._views import answer_view as answer_view
from ._views import message_view as message_view
from ._views import question_view as question_view
from ._views import write_json as write_json
from ._views import write_jsonl as write_jsonl
from ._views import write_readme as write_readme
