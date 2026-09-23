"""The causally available transcript window for one replay cursor."""

import math
from collections.abc import Mapping, Sequence
from typing import cast

from .pairs import TRANSCRIPT_ALIGNMENT_VERSION, TRANSCRIPT_SCHEMA
from .segments import parse_transcript_segments

__all__ = ["build_paired_transcript_window"]


def _rounded_seconds(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _rounded_transcript_segments(
    raw_segments: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            **segment,
            "start_s": _rounded_seconds(cast(float, segment["start_s"])),
            "end_s": _rounded_seconds(cast(float, segment["end_s"])),
        }
        for segment in parse_transcript_segments(raw_segments)
    ]


def build_paired_transcript_window(
    replay_data: Mapping[str, object], replay_index: int
) -> dict[str, object] | None:
    """Build current and cumulative causally available transcript text."""
    transcript = cast(Mapping[str, object] | None, replay_data.get("paired_transcript"))
    if not transcript:
        return None

    timings = cast(
        Sequence[Mapping[str, object]], transcript.get("action_timings", [])
    )
    base = {
        "schema": transcript.get("schema", TRANSCRIPT_SCHEMA),
        "alignment_version": transcript.get(
            "alignment_version", TRANSCRIPT_ALIGNMENT_VERSION
        ),
        "video_id": transcript.get("video_id"),
        "video_url": transcript.get("video_url"),
        "pairing_status": transcript.get("pairing_status", "unknown"),
        "verified": bool(transcript.get("verified", False)),
        "narrator": transcript.get("narrator", {}),
        "replay_index": replay_index,
        "window_policy": (
            "prior_boundary <= segment_end < cursor_boundary; "
            "history includes every segment ending before cursor_boundary"
        ),
    }

    if replay_index < 0:
        return {
            **base,
            "status": "unavailable",
            "raw_segment_count": 0,
            "window_start_s": None,
            "window_end_s": None,
            "clock_anomaly": False,
            "segments": [],
            "history_raw_segment_count": 0,
            "history_segments": [],
        }

    if replay_index >= len(timings):
        last_time = timings[-1].get("wall_time_s") if timings else 0.0
        if (
            not isinstance(last_time, (int, float))
            or isinstance(last_time, bool)
            or not math.isfinite(float(last_time))
        ):
            return {
                **base,
                "status": "unavailable",
                "raw_segment_count": 0,
                "window_start_s": _rounded_seconds(cast(float | None, last_time)),
                "window_end_s": None,
                "clock_anomaly": False,
                "segments": [],
                "history_raw_segment_count": 0,
                "history_segments": [],
            }

        history_raw_segments = list(
            cast(Sequence[Mapping[str, float]], transcript.get("segments", []))
        )
        raw_segments = [
            segment
            for segment in history_raw_segments
            if segment["end_s"] >= float(last_time)
        ]
        segments = _rounded_transcript_segments(raw_segments)
        history_segments = _rounded_transcript_segments(history_raw_segments)
        final_time = max(
            (segment["end_s"] for segment in raw_segments),
            default=float(last_time),
        )
        return {
            **base,
            "status": "complete",
            "raw_segment_count": len(raw_segments),
            "window_start_s": _rounded_seconds(float(last_time)),
            "window_end_s": _rounded_seconds(final_time),
            "clock_anomaly": False,
            "segments": segments,
            "history_raw_segment_count": len(history_raw_segments),
            "history_segments": history_segments,
        }

    window_end = timings[replay_index].get("wall_time_s")
    window_start = 0.0 if replay_index == 0 else timings[replay_index - 1].get(
        "wall_time_s"
    )

    if (
        not isinstance(window_start, (int, float))
        or isinstance(window_start, bool)
        or not math.isfinite(float(window_start))
        or not isinstance(window_end, (int, float))
        or isinstance(window_end, bool)
        or not math.isfinite(float(window_end))
    ):
        return {
            **base,
            "status": "unavailable",
            "raw_segment_count": 0,
            "window_start_s": _rounded_seconds(cast(float | None, window_start)),
            "window_end_s": _rounded_seconds(cast(float | None, window_end)),
            "clock_anomaly": False,
            "segments": [],
            "history_raw_segment_count": 0,
            "history_segments": [],
        }

    clock_anomaly = window_start > window_end
    raw_segments = []
    history_raw_segments = []
    if not clock_anomaly:
        # Rolling YouTube chunks often begin before an action and finish after it.
        # Assign each chunk once by its completion time instead of dropping it.
        all_segments = cast(
            Sequence[Mapping[str, float]], transcript.get("segments", [])
        )
        raw_segments = [
            segment
            for segment in all_segments
            if segment["end_s"] >= window_start and segment["end_s"] < window_end
        ]
        history_raw_segments = [
            segment for segment in all_segments if segment["end_s"] < window_end
        ]
    segments = _rounded_transcript_segments(raw_segments)
    history_segments = _rounded_transcript_segments(history_raw_segments)

    return {
        **base,
        "status": "clock_anomaly" if clock_anomaly else ("ready" if segments else "empty"),
        "raw_segment_count": len(raw_segments),
        "window_start_s": _rounded_seconds(window_start),
        "window_end_s": _rounded_seconds(window_end),
        "clock_anomaly": clock_anomaly,
        "segments": segments,
        "history_raw_segment_count": len(history_raw_segments),
        "history_segments": history_segments,
    }
