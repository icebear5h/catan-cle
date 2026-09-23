"""Implementation of the spatial and robber supplement, re-exported by its module.

``data_pipeline/board_recognition/spatial_robber.py`` stays a physical file
because an SFT builder opens that exact path and records its sha256 into the
manifests it generates. This subpackage holds the implementation.
"""

from __future__ import annotations

from data_pipeline.board_recognition.robber_impl._bank import _relation_bank, spatial_query_bank
from data_pipeline.board_recognition.robber_impl._config import (
    ATLAS_TOKENS,
    AUDIT_SCHEMA,
    CURRICULUM_STAGES,
    DEFAULT_OUTPUT_NAME,
    EXPORT_SCHEMA,
    ROBBER_ROWS_PER_STATE,
    SMOKE_ROWS_PER_STAGE,
    SPATIAL_ROWS_PER_EMPTY_STATE,
    STAGE_BY_DENSITY,
    JsonDict,
    SpatialRobberError,
)
from data_pipeline.board_recognition.robber_impl._export import (
    export_spatial_robber_supplement,
)
from data_pipeline.board_recognition.robber_impl._geometry import (
    _atlas_geometry,
    _direction,
    _graph_distance,
    _opposite,
    _pixel,
)
from data_pipeline.board_recognition.robber_impl._io import (
    _stable_rank,
    _training_row,
    _write_json,
    _write_jsonl,
)
from data_pipeline.board_recognition.robber_impl._queries import (
    _audit_row,
    robber_queries_for_state,
    spatial_queries_for_state,
)
from data_pipeline.board_recognition.robber_impl._smoke import (
    _diverse_indices,
    build_curriculum_smoke_rows,
)
from data_pipeline.board_recognition.robber_impl._validate import (
    validate_spatial_robber_supplement,
)

__all__ = [
    "ATLAS_TOKENS",
    "AUDIT_SCHEMA",
    "CURRICULUM_STAGES",
    "DEFAULT_OUTPUT_NAME",
    "EXPORT_SCHEMA",
    "ROBBER_ROWS_PER_STATE",
    "SMOKE_ROWS_PER_STAGE",
    "SPATIAL_ROWS_PER_EMPTY_STATE",
    "STAGE_BY_DENSITY",
    "JsonDict",
    "SpatialRobberError",
    "_atlas_geometry",
    "_audit_row",
    "_direction",
    "_diverse_indices",
    "_graph_distance",
    "_opposite",
    "_pixel",
    "_relation_bank",
    "_stable_rank",
    "_training_row",
    "_write_json",
    "_write_jsonl",
    "build_curriculum_smoke_rows",
    "export_spatial_robber_supplement",
    "robber_queries_for_state",
    "spatial_queries_for_state",
    "spatial_query_bank",
    "validate_spatial_robber_supplement",
]
