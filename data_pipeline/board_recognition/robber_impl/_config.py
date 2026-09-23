"""Schemas, stage tables, row budgets, and the supplement error type."""

from __future__ import annotations

from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.tokens import atlas_tokens

__all__ = ["JsonDict"]


EXPORT_SCHEMA = "catan_spatial_robber_supplement/v1"
AUDIT_SCHEMA = "catan_spatial_robber_audit/v1"
DEFAULT_OUTPUT_NAME = "spatial_robber_v1"
SPATIAL_ROWS_PER_EMPTY_STATE = 24
ROBBER_ROWS_PER_STATE = 3
SMOKE_ROWS_PER_STAGE = 8
CURRICULUM_STAGES = (
    "spatial_grounding",
    "clean_board_grounding",
    "pieces_and_colors",
    "real_game_distribution",
)
STAGE_BY_DENSITY = {
    "empty": "clean_board_grounding",
    "setup": "clean_board_grounding",
    "sparse": "pieces_and_colors",
    "dense": "real_game_distribution",
}
ATLAS_TOKENS = frozenset(atlas_tokens())


class SpatialRobberError(RuntimeError):
    __module__ = "data_pipeline.board_recognition.spatial_robber"
    """Raised when the supplement violates its deterministic contract."""
