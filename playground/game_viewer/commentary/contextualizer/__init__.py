"""Causal blind-then-reveal stepping for engine-grounded commentary."""

from .evidence import CommentarySessionBase, EvidenceSelector
from .models import (
    BlindContext,
    CommentaryContextError,
    CommentarySpan,
    CommitToken,
    EvidenceSegment,
    EvidenceSelectionInput,
    RevealedEvent,
    _color_name,
    _engine_color_for_colonist,
    _PendingContext,
    _state_fingerprint,
)
from .session import CausalCommentarySession

__all__ = [
    "BlindContext",
    "CausalCommentarySession",
    "CommentaryContextError",
    "CommentarySessionBase",
    "CommentarySpan",
    "CommitToken",
    "EvidenceSegment",
    "EvidenceSelectionInput",
    "EvidenceSelector",
    "RevealedEvent",
    "_PendingContext",
    "_color_name",
    "_engine_color_for_colonist",
    "_state_fingerprint",
]
