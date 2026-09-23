"""Compact semantic grouping for causally available transcript evidence."""

from .patterns import (
    BOARD,
    COMMITMENT,
    MAX_TRACE_CHARS,
    MAX_VISIBLE_TRACES,
    MIN_TRACE_CHARS,
    NEGOTIATION,
    NUMBER_REFERENCE,
    OPPONENT,
    OPTION,
    PLAN,
    RECENT_CONTEXT_SECONDS,
    ROLE_TITLES,
    STOP_WORDS,
    STRONG_ROLES,
    TARGET_TRACE_CHARS,
    UPDATE,
    WORD,
)
from .segmentation import (
    _boundary_score,
    _dominant_role,
    _partition_utterances,
    _should_share_boundary,
    _tokens,
    _utterance_role,
)
from .traces import _trace_payload, build_semantic_transcript_traces
from .utterances import SemanticTrace, Utterance

__all__ = [
    "BOARD",
    "COMMITMENT",
    "MAX_TRACE_CHARS",
    "MAX_VISIBLE_TRACES",
    "MIN_TRACE_CHARS",
    "NEGOTIATION",
    "NUMBER_REFERENCE",
    "OPPONENT",
    "OPTION",
    "PLAN",
    "RECENT_CONTEXT_SECONDS",
    "ROLE_TITLES",
    "STOP_WORDS",
    "STRONG_ROLES",
    "SemanticTrace",
    "TARGET_TRACE_CHARS",
    "UPDATE",
    "Utterance",
    "WORD",
    "_BOARD",
    "_COMMITMENT",
    "_MAX_TRACE_CHARS",
    "_MAX_VISIBLE_TRACES",
    "_MIN_TRACE_CHARS",
    "_NEGOTIATION",
    "_NUMBER_REFERENCE",
    "_OPPONENT",
    "_OPTION",
    "_PLAN",
    "_RECENT_CONTEXT_SECONDS",
    "_ROLE_TITLES",
    "_STOP_WORDS",
    "_STRONG_ROLES",
    "_TARGET_TRACE_CHARS",
    "_UPDATE",
    "_WORD",
    "_boundary_score",
    "_dominant_role",
    "_partition_utterances",
    "_should_share_boundary",
    "_tokens",
    "_trace_payload",
    "_utterance_role",
    "build_semantic_transcript_traces",
]


# The historical private names this module exposed before the split.
_NUMBER_REFERENCE = NUMBER_REFERENCE
_WORD = WORD
_COMMITMENT = COMMITMENT
_PLAN = PLAN
_OPTION = OPTION
_OPPONENT = OPPONENT
_NEGOTIATION = NEGOTIATION
_UPDATE = UPDATE
_BOARD = BOARD
_STOP_WORDS = STOP_WORDS
_ROLE_TITLES = ROLE_TITLES
_STRONG_ROLES = STRONG_ROLES
_MIN_TRACE_CHARS = MIN_TRACE_CHARS
_TARGET_TRACE_CHARS = TARGET_TRACE_CHARS
_MAX_TRACE_CHARS = MAX_TRACE_CHARS
_RECENT_CONTEXT_SECONDS = RECENT_CONTEXT_SECONDS
_MAX_VISIBLE_TRACES = MAX_VISIBLE_TRACES
