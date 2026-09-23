"""Grouped evidence rendered as the bounded trace list the viewer shows."""

import hashlib
from collections.abc import Sequence

from .patterns import MAX_VISIBLE_TRACES, RECENT_CONTEXT_SECONDS, ROLE_TITLES
from .segmentation import _dominant_role, _partition_utterances
from .utterances import SemanticTrace, Utterance

__all__ = ["_trace_payload", "build_semantic_transcript_traces"]


def _trace_payload(
    utterances: Sequence[Utterance],
    overlaps_previous: bool,
) -> SemanticTrace:
    role = _dominant_role(utterances)
    evidence: list[Utterance] = [
        {
            "start_s": utterance["start_s"],
            "end_s": utterance["end_s"],
            "text": utterance["text"],
            "source_start_index": utterance["source_start_index"],
            "source_end_index": utterance["source_end_index"],
            "source_segment_count": utterance["source_segment_count"],
        }
        for utterance in utterances
    ]
    source_start = min(item["source_start_index"] for item in evidence)
    source_end = max(item["source_end_index"] for item in evidence)
    identity = f"{role}:{source_start}:{source_end}:" + ":".join(
        str(item["source_start_index"]) for item in evidence
    )
    return {
        "trace_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        "role": role,
        "title": ROLE_TITLES[role],
        "start_s": min(float(item["start_s"]) for item in evidence),
        "end_s": max(float(item["end_s"]) for item in evidence),
        "overlaps_previous": overlaps_previous,
        "text": " ".join(str(item["text"]) for item in evidence),
        "evidence": evidence,
    }


def build_semantic_transcript_traces(
    utterances: Sequence[Utterance],
    context_start_s: float | None,
    context_end_s: float | None = None,
) -> list[SemanticTrace]:
    """Return coherent current evidence plus a bounded transition context."""
    selected_utterances = list(utterances)
    if context_start_s is not None:
        context_start = float(context_start_s)
        has_current_evidence = any(
            float(utterance["end_s"]) >= context_start
            for utterance in selected_utterances
        )
        long_empty_interval = (
            context_end_s is not None
            and float(context_end_s) - context_start > RECENT_CONTEXT_SECONDS
            and not has_current_evidence
        )
        if long_empty_interval:
            return []
        recent_cutoff = max(0.0, context_start - RECENT_CONTEXT_SECONDS)
        selected_utterances = [
            utterance
            for utterance in selected_utterances
            if float(utterance["end_s"]) >= recent_cutoff
        ]

    traces = [
        _trace_payload(group, overlaps_previous)
        for group, overlaps_previous in _partition_utterances(selected_utterances)
    ]
    return traces[-MAX_VISIBLE_TRACES:]
