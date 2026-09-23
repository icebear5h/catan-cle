"""Deterministic spatial-grounding and robber-localization SFT supplement.

This module stays a physical file at this exact path: an SFT builder opens it
and records its sha256 into the manifests it generates. The implementation
lives in ``data_pipeline.board_recognition.robber_impl``; every name the module
used to define is re-exported here unchanged.
"""

from __future__ import annotations

from data_pipeline.board_recognition.replay_dataset import (
    ALL_SPLITS,
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.robber_impl import (
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
    _atlas_geometry,
    _audit_row,
    _direction,
    _diverse_indices,
    _graph_distance,
    _opposite,
    _pixel,
    _relation_bank,
    _stable_rank,
    _training_row,
    _write_json,
    _write_jsonl,
    build_curriculum_smoke_rows,
    export_spatial_robber_supplement,
    robber_queries_for_state,
    spatial_queries_for_state,
    spatial_query_bank,
    validate_spatial_robber_supplement,
)
from evals.catan_board_bench.tokens import atlas_metadata, atlas_tokens

__all__ = [
    "ALL_SPLITS",
    "ATLAS_TOKENS",
    "AUDIT_SCHEMA",
    "CURRICULUM_STAGES",
    "DEFAULT_OUTPUT_NAME",
    "EXPORT_SCHEMA",
    "JsonDict",
    "ROBBER_ROWS_PER_STATE",
    "SMOKE_ROWS_PER_STAGE",
    "SPATIAL_ROWS_PER_EMPTY_STATE",
    "STAGE_BY_DENSITY",
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
    "atlas_metadata",
    "atlas_tokens",
    "build_curriculum_smoke_rows",
    "export_spatial_robber_supplement",
    "file_sha256",
    "read_jsonl",
    "robber_queries_for_state",
    "spatial_queries_for_state",
    "spatial_query_bank",
    "validate_replay_v1_dataset",
    "validate_spatial_robber_supplement",
]
