"""The guarded cutoff that keeps commentary strictly ahead of its event."""

import math
from collections.abc import Mapping, Sequence
from typing import cast

from .segments import Utterance, parse_transcript_segments

__all__ = ["guarded_transcript_bounds", "select_guarded_transcript_evidence"]


def guarded_transcript_bounds(
    replay_data: Mapping[str, object],
    replay_index: int,
    guard_seconds: float = 3.0,
) -> tuple[float, float] | None:
    """Return the private lower bound and guarded cutoff for one replay cursor."""
    if not math.isfinite(guard_seconds) or guard_seconds < 0:
        raise ValueError("guard_seconds must be finite and nonnegative")

    transcript = cast(Mapping[str, object] | None, replay_data.get("paired_transcript"))
    if not transcript:
        return None
    timings = cast(
        Sequence[Mapping[str, object]], transcript.get("action_timings", [])
    )
    if replay_index < 0 or replay_index >= len(timings):
        return None

    event_time = timings[replay_index].get("wall_time_s")
    prior_time = (
        0.0
        if replay_index == 0
        else timings[replay_index - 1].get("wall_time_s")
    )
    if not isinstance(event_time, (int, float)) or not isinstance(
        prior_time, (int, float)
    ):
        return None
    if not math.isfinite(float(event_time)) or not math.isfinite(float(prior_time)):
        return None

    cutoff = float(event_time) - guard_seconds
    if float(prior_time) > cutoff:
        return None
    return float(prior_time), cutoff


def select_guarded_transcript_evidence(
    replay_data: Mapping[str, object],
    replay_index: int,
    guard_seconds: float = 3.0,
) -> tuple[Utterance, ...]:
    """Select whole commentary segments available before a guarded event cutoff.

    The result intentionally omits the upcoming action, raw event index, and
    event timestamp. It is safe to hand to a provisional contextualization pass.
    """
    bounds = guarded_transcript_bounds(replay_data, replay_index, guard_seconds)
    if bounds is None:
        return ()
    prior_time, cutoff = bounds
    transcript = cast(Mapping[str, object], replay_data["paired_transcript"])

    raw_segments = [
        segment
        for segment in cast(
            Sequence[Mapping[str, float]], transcript.get("segments", [])
        )
        if segment["end_s"] > prior_time and segment["end_s"] <= cutoff
    ]
    return tuple(parse_transcript_segments(raw_segments))
