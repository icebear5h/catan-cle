"""Reflowing rolling caption chunks into whole utterances."""

import math
import re
from collections.abc import Mapping, Sequence

__all__ = ["Utterance", "parse_transcript_segments"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_END = re.compile(r"[.!?][\"'”’]?$")
_MAX_UTTERANCE_CHARS = 280
_MAX_CAPTION_GAP_SECONDS = 1.5

Utterance = dict[str, object]


def _event_wall_times(events: Sequence[Mapping[str, object]]) -> list[float | None]:
    """Return cumulative Colonist wall time for every raw replay event."""
    wall_time = 0.0
    times: list[float | None] = []
    clock_valid = True
    for event in events:
        raw_input = event.get("input")
        delta = raw_input.get("deltaS", 0) if isinstance(raw_input, Mapping) else 0
        if (
            clock_valid
            and isinstance(delta, (int, float))
            and not isinstance(delta, bool)
            and math.isfinite(float(delta))
        ):
            wall_time += float(delta)
            clock_valid = math.isfinite(wall_time)
        elif isinstance(delta, (int, float)) and not isinstance(delta, bool):
            clock_valid = False
        times.append(wall_time if clock_valid else None)
    return times


def parse_transcript_segments(
    segments: Sequence[Mapping[str, object]],
) -> list[Utterance]:
    """Reflow rolling caption chunks without inventing text or timestamps."""
    utterances: list[Utterance] = []
    parts: list[str] = []
    source_indices: list[int] = []
    utterance_start: float | None = None
    utterance_end: float | None = None

    def flush() -> None:
        nonlocal parts, source_indices, utterance_start, utterance_end
        if not parts or utterance_start is None or utterance_end is None:
            return
        utterances.append(
            {
                "start_s": utterance_start,
                "end_s": utterance_end,
                "text": " ".join(parts),
                "source_start_index": min(source_indices),
                "source_end_index": max(source_indices),
                "source_segment_count": len(set(source_indices)),
            }
        )
        parts = []
        source_indices = []
        utterance_start = None
        utterance_end = None

    for fallback_index, segment in enumerate(segments):
        start = segment.get("start_s")
        end = segment.get("end_s")
        text = segment.get("text")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or not isinstance(text, str)
        ):
            continue

        if (
            parts
            and utterance_end is not None
            and float(start) > utterance_end + _MAX_CAPTION_GAP_SECONDS
        ):
            flush()

        source_index = segment.get("source_index", fallback_index)
        if not isinstance(source_index, int) or isinstance(source_index, bool):
            source_index = fallback_index

        normalized_text = " ".join(text.split())
        for fragment in _SENTENCE_SPLIT.split(normalized_text):
            fragment = fragment.strip()
            if not fragment:
                continue

            candidate = " ".join([*parts, fragment])
            if parts and len(candidate) > _MAX_UTTERANCE_CHARS:
                flush()

            if utterance_start is None:
                utterance_start = float(start)
            utterance_end = max(utterance_end or float(end), float(end))
            parts.append(fragment)
            source_indices.append(source_index)

            if _SENTENCE_END.search(fragment):
                flush()

    flush()
    return utterances
