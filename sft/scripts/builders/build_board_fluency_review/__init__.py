"""Build the bounded, text-only 200-example board-fluency human review batch.

This is a review schema, not an admitted trainer task schema. Only the first
MAX_SOURCE_ROWS physical source lines are read; the full source hash is merely
reported from its manifest. Source states and original provenance stay intact.
"""

from __future__ import annotations

import argparse as argparse
import copy as copy
import hashlib as hashlib
import json as json
import random as random
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from collections import deque as deque
from dataclasses import dataclass as dataclass
from itertools import combinations as combinations
from itertools import islice as islice
from pathlib import Path as Path

from data_pipeline.board_recognition.replay_dataset import static_board_facts as static_board_facts
from data_pipeline.board_recognition.sources import canonical_sha256 as canonical_sha256
from data_pipeline.board_recognition.sources import file_sha256 as file_sha256
from data_pipeline.board_recognition.sources import repository_relative as repository_relative
from data_pipeline.board_recognition.sources import visible_board_facts as visible_board_facts
from evals.catan_board_bench.annotations import contract_to_render_state as contract_to_render_state
from sft.board.spatial_tasks import dice_production as dice_production
from sft.board.spatial_tasks import local_node_tiles as local_node_tiles
from sft.board.symbolic_board_tasks import ROUTE_RULES as ROUTE_RULES
from sft.board.symbolic_board_tasks import TRAIN_TASKS as TRAIN_TASKS
from sft.board.symbolic_board_tasks import atlas_geometry as atlas_geometry
from sft.board.symbolic_board_tasks import decode_state as decode_state
from sft.board.symbolic_board_tasks import owned_route as owned_route
from sft.board.symbolic_board_tasks import symbolic_answer as symbolic_answer
from sft.board.symbolic_board_tasks import validate_contract as validate_contract

from ._audit import audit_contract as audit_contract
from ._audit import contract_components as contract_components
from ._audit import reference_answer as reference_answer
from ._audit import validate_render as validate_render
from ._build import build as build
from ._build import main as main
from ._build import publish as publish
from ._candidates import Candidate as Candidate
from ._candidates import candidates_for as candidates_for
from ._candidates import select as select
from ._facts import Facts as Facts
from ._facts import pips as pips
from ._facts import production_change as production_change
from ._facts import richness as richness
from ._facts import text_answer as text_answer
from ._prompt import answer as answer
from ._prompt import prompt as prompt
from ._prompt import question as question
from ._sources import CELLS as CELLS
from ._sources import CHANGE_FORMAT as CHANGE_FORMAT
from ._sources import COMPONENT_RULES as COMPONENT_RULES
from ._sources import COVERAGE_RULES as COVERAGE_RULES
from ._sources import FAMILIES as FAMILIES
from ._sources import MAX_SOURCE_ROWS as MAX_SOURCE_ROWS
from ._sources import OPERATION_FAMILY as OPERATION_FAMILY
from ._sources import OUTPUT as OUTPUT
from ._sources import OWN_FILES as OWN_FILES
from ._sources import PIP_RULES as PIP_RULES
from ._sources import PRODUCTION_RULES as PRODUCTION_RULES
from ._sources import RESOURCES as RESOURCES
from ._sources import ROOT as ROOT
from ._sources import SCHEMA as SCHEMA
from ._sources import SEED as SEED
from ._sources import SET_FORMAT as SET_FORMAT
from ._sources import SOURCE as SOURCE
from ._sources import VECTOR_FORMAT as VECTOR_FORMAT
from ._sources import VERSION as VERSION
from ._sources import Donor as Donor
from ._sources import QueryDict as QueryDict
from ._sources import compact as compact
from ._sources import load_prefix as load_prefix
from ._sources import require as require
