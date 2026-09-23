"""The two record shapes this package reads and writes."""

from typing import TypedDict

__all__ = ["SemanticTrace", "Utterance"]


class Utterance(TypedDict):
    """One transcript segment, carrying where it came from in the source."""

    text: str
    start_s: float
    end_s: float
    source_start_index: int
    source_end_index: int
    source_segment_count: int


class SemanticTrace(TypedDict):
    """One coherent stretch of talk, titled by the role that dominates it."""

    trace_id: str
    role: str
    title: str
    start_s: float
    end_s: float
    overlaps_previous: bool
    text: str
    evidence: list[Utterance]
