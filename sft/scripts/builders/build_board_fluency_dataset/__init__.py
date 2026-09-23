"""Build the approved, admitted symbolic board-fluency SFT corpus locally.

Reuse source admission and the reviewed candidates/questions/independent oracles.
No source generation, model calls, or historical artifact writes. --dry-run checks
the complete proposed corpus without writing; --validate is read-only admission
for launchers (the messages-only trainer cannot enforce these declarations).
"""

from __future__ import annotations

import argparse as argparse
import json as json
import random as random
import re as re
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from dataclasses import dataclass as dataclass
from pathlib import Path as Path

from data_pipeline.board_recognition.sources import canonical_sha256 as canonical_sha256
from data_pipeline.board_recognition.sources import file_sha256 as file_sha256
from sft.board.board_fluency_scoring import SFT_SCHEMA as SFT_SCHEMA
from sft.board.board_fluency_scoring import score_board_fluency as score_board_fluency
from sft.board.symbolic_board_tasks import atlas_geometry as atlas_geometry
from sft.board.symbolic_board_tasks import decode_state as decode_state
from sft.scripts.builders.build_symbolic_board_dataset import DEFAULT_ROOT as DEFAULT_ROOT
from sft.scripts.builders.build_symbolic_board_dataset import (
    graph_case_coverage as graph_case_coverage,
)
from sft.scripts.builders.build_symbolic_board_dataset import load_sources as load_sources
from sft.scripts.builders.build_symbolic_board_dataset import read_json as read_json
from sft.scripts.builders.build_symbolic_board_dataset import read_jsonl as read_jsonl

from ._build import build_dataset as build_dataset
from ._build import main as main
from ._build import validate_dataset as validate_dataset
from ._rows import exposure as exposure
from ._rows import make_rows as make_rows
from ._rows import profile as profile
from ._rows import validation_subset as validation_subset
from ._select import interleave as interleave
from ._select import nx as nx
from ._select import select as select
from ._selection import apportion as apportion
from ._selection import eligible_sources as eligible_sources
from ._selection import quotas as quotas
from ._selection import review_exclusions as review_exclusions
from ._selection import roster_positions as roster_positions
from ._sources import ATLAS_PATTERN as ATLAS_PATTERN
from ._sources import COUNTS as COUNTS
from ._sources import INVENTORY as INVENTORY
from ._sources import OPERATIONS as OPERATIONS
from ._sources import OUTPUT as OUTPUT
from ._sources import PIP_OPERATIONS as PIP_OPERATIONS
from ._sources import REVIEW as REVIEW
from ._sources import REVIEW_SHA256 as REVIEW_SHA256
from ._sources import ROOT as ROOT
from ._sources import SCHEMA as SCHEMA
from ._sources import SEED as SEED
from ._sources import SPLITS as SPLITS
from ._sources import VERSION as VERSION
from ._sources import Donor as Donor
from ._sources import donor_for as donor_for
from ._sources import pin as pin
from ._sources import require as require
from ._sources import review as review
from ._validate import validate_rows as validate_rows
