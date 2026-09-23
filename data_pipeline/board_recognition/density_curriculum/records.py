"""Typed records for the ordered density curriculum."""

from __future__ import annotations

from typing import TypedDict

from data_pipeline.json_types import JsonDict


class StageDefinition(TypedDict):
    """One density stage of the ordered curriculum."""

    stage: str
    stage_index: int
    density_bins: tuple[str, ...]
    minimum_piece_count: int
    maximum_piece_count: int

class StageRecord(TypedDict):
    """One annotation/audit pair with its position in the source order."""

    annotation: JsonDict
    audit: JsonDict
    original_index: int
