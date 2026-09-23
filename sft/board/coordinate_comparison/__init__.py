"""Matched atlas-ID/integer-coordinate inference diagnostics, without trainer imports.

Only ``messages[0]`` is model input. Canonical targets and source receipts stay in
metadata; the crosswalk is a separate artifact. This is a representation-usability
comparison on an atlas-trained checkpoint, not an equal-training-budget experiment.
"""

from __future__ import annotations

import copy as copy
import hashlib as hashlib
import json as json
import re as re
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from functools import lru_cache as lru_cache
from pathlib import Path as Path

from data_pipeline.board_recognition.sources import canonical_sha256 as canonical_sha256
from data_pipeline.board_recognition.sources import file_sha256 as file_sha256
from sft.board.board_fluency_scoring import _strip_transport as _strip_transport
from sft.board.coordinate_comparison._constants import ATLAS_ATOM as ATLAS_ATOM
from sft.board.coordinate_comparison._constants import ATOM_INSTRUCTION as ATOM_INSTRUCTION
from sft.board.coordinate_comparison._constants import COMPARISON_FIELDS as COMPARISON_FIELDS
from sft.board.coordinate_comparison._constants import COMPLETE_TEST_TASKS as COMPLETE_TEST_TASKS
from sft.board.coordinate_comparison._constants import COORDINATE_ATOM as COORDINATE_ATOM
from sft.board.coordinate_comparison._constants import DEFAULT_SOURCE as DEFAULT_SOURCE
from sft.board.coordinate_comparison._constants import GEOMETRY_CONVENTION as GEOMETRY_CONVENTION
from sft.board.coordinate_comparison._constants import INCIDENCE_RELATIONS as INCIDENCE_RELATIONS
from sft.board.coordinate_comparison._constants import INTEGER as INTEGER
from sft.board.coordinate_comparison._constants import OPERATION_AREA as OPERATION_AREA
from sft.board.coordinate_comparison._constants import PAIR_QUOTAS as PAIR_QUOTAS
from sft.board.coordinate_comparison._constants import PROJECT_ROOT as PROJECT_ROOT
from sft.board.coordinate_comparison._constants import REPRESENTATIONS as REPRESENTATIONS
from sft.board.coordinate_comparison._constants import SCHEMA as SCHEMA
from sft.board.coordinate_comparison._constants import SOURCE_ROW_FIELDS as SOURCE_ROW_FIELDS
from sft.board.coordinate_comparison._constants import VERSION as VERSION
from sft.board.coordinate_comparison._mapping import _compact as _compact
from sft.board.coordinate_comparison._mapping import _geometry as _geometry
from sft.board.coordinate_comparison._mapping import _inverse_mapping as _inverse_mapping
from sft.board.coordinate_comparison._mapping import _mapping as _mapping
from sft.board.coordinate_comparison._mapping import _require as _require
from sft.board.coordinate_comparison._mapping import _same as _same
from sft.board.coordinate_comparison._mapping import _static_fact_sha256 as _static_fact_sha256
from sft.board.coordinate_comparison._mapping import _unique_object as _unique_object
from sft.board.coordinate_comparison._mapping import coordinate_mapping as coordinate_mapping
from sft.board.coordinate_comparison._mapping import mapping_artifact as mapping_artifact
from sft.board.coordinate_comparison._mapping import mapping_sha256 as mapping_sha256
from sft.board.coordinate_comparison._mapping import project_text as project_text
from sft.board.coordinate_comparison._mapping import read_jsonl as read_jsonl
from sft.board.coordinate_comparison._rows import _relation_key as _relation_key
from sft.board.coordinate_comparison._rows import _required_cells as _required_cells
from sft.board.coordinate_comparison._rows import _sampling_cell as _sampling_cell
from sft.board.coordinate_comparison._rows import _source_row as _source_row
from sft.board.coordinate_comparison._rows import _source_semantics as _source_semantics
from sft.board.coordinate_comparison._rows import build_comparison_rows as build_comparison_rows
from sft.board.coordinate_comparison._rows import comparison_prompt as comparison_prompt
from sft.board.coordinate_comparison._rows import load_source_rows as load_source_rows
from sft.board.coordinate_comparison._rows import select_source_cases as select_source_cases
from sft.board.coordinate_comparison._scoring import _pair_identity as _pair_identity
from sft.board.coordinate_comparison._scoring import _parse_response as _parse_response
from sft.board.coordinate_comparison._scoring import (
    score_coordinate_comparison as score_coordinate_comparison,
)
from sft.board.coordinate_comparison._scoring import (
    validate_comparison_metadata as validate_comparison_metadata,
)
from sft.board.coordinate_comparison._scoring import (
    validate_comparison_rows as validate_comparison_rows,
)
from sft.board.coordinate_comparison._summaries import _summarize_pairs as _summarize_pairs
from sft.board.coordinate_comparison._summaries import paired_summary as paired_summary
from sft.board.symbolic_board_tasks import STATIC_TASKS as STATIC_TASKS
from sft.board.symbolic_board_tasks import atlas_geometry as atlas_geometry
from sft.board.symbolic_board_tasks import score_symbolic_task as score_symbolic_task
from sft.board.symbolic_board_tasks import symbolic_answer as symbolic_answer
from sft.board.symbolic_board_tasks import symbolic_prompt as symbolic_prompt
from sft.board.symbolic_board_tasks import symbolic_task_role as symbolic_task_role

__all__: list[str] = [
    "ATLAS_ATOM",
    "ATOM_INSTRUCTION",
    "COMPARISON_FIELDS",
    "COMPLETE_TEST_TASKS",
    "COORDINATE_ATOM",
    "Counter",
    "DEFAULT_SOURCE",
    "GEOMETRY_CONVENTION",
    "INCIDENCE_RELATIONS",
    "INTEGER",
    "OPERATION_AREA",
    "PAIR_QUOTAS",
    "PROJECT_ROOT",
    "Path",
    "REPRESENTATIONS",
    "SCHEMA",
    "SOURCE_ROW_FIELDS",
    "STATIC_TASKS",
    "VERSION",
    "annotations",
    "atlas_geometry",
    "build_comparison_rows",
    "canonical_sha256",
    "comparison_prompt",
    "coordinate_mapping",
    "copy",
    "defaultdict",
    "file_sha256",
    "hashlib",
    "json",
    "load_source_rows",
    "lru_cache",
    "mapping_artifact",
    "mapping_sha256",
    "paired_summary",
    "project_text",
    "re",
    "read_jsonl",
    "score_coordinate_comparison",
    "score_symbolic_task",
    "select_source_cases",
    "symbolic_answer",
    "symbolic_prompt",
    "symbolic_task_role",
    "validate_comparison_metadata",
    "validate_comparison_rows",
]
