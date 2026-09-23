"""Curated replay/transcript pairing and cursor-safe transcript windows."""

from .bounds import guarded_transcript_bounds, select_guarded_transcript_evidence
from .pairing import load_paired_transcript, paired_transcript_fingerprint
from .pairs import (
    _CURATED_PAIRS,
    PROJECT_ROOT,
    TRANSCRIPT_ALIGNMENT_VERSION,
    TRANSCRIPT_SCHEMA,
    CuratedPair,
    curated_pair,
    get_curated_replay_path,
)
from .segments import (
    _MAX_CAPTION_GAP_SECONDS,
    _MAX_UTTERANCE_CHARS,
    _SENTENCE_END,
    _SENTENCE_SPLIT,
    Utterance,
    _event_wall_times,
    parse_transcript_segments,
)
from .window import (
    _rounded_seconds,
    _rounded_transcript_segments,
    build_paired_transcript_window,
)

__all__ = [
    "CuratedPair",
    "PROJECT_ROOT",
    "TRANSCRIPT_ALIGNMENT_VERSION",
    "TRANSCRIPT_SCHEMA",
    "Utterance",
    "_CURATED_PAIRS",
    "_MAX_CAPTION_GAP_SECONDS",
    "_MAX_UTTERANCE_CHARS",
    "_SENTENCE_END",
    "_SENTENCE_SPLIT",
    "_event_wall_times",
    "_rounded_seconds",
    "_rounded_transcript_segments",
    "build_paired_transcript_window",
    "curated_pair",
    "get_curated_replay_path",
    "guarded_transcript_bounds",
    "load_paired_transcript",
    "paired_transcript_fingerprint",
    "parse_transcript_segments",
    "select_guarded_transcript_evidence",
]
