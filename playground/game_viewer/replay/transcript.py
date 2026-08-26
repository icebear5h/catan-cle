"""Curated replay/transcript pairing and cursor-safe transcript windows."""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRANSCRIPT_SCHEMA = "paired-replay-transcript-v1"
TRANSCRIPT_ALIGNMENT_VERSION = "caption-end-availability-v2"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_END = re.compile(r"[.!?][\"'”’]?$")
_MAX_UTTERANCE_CHARS = 280
_MAX_CAPTION_GAP_SECONDS = 1.5

_CURATED_PAIRS = {
    "242781000": {
        "replay_path": PROJECT_ROOT
        / "data_pipeline/bootstrapping/data/replay_staging/242781000.json",
        "transcript_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/transcript.json",
        "manifest_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/pairing_manifest.json",
    }
}


def get_curated_replay_path(game_id: str) -> Optional[Path]:
    """Return the staged replay path for a curated transcript pair, if any."""
    pair = _CURATED_PAIRS.get(str(game_id))
    return pair["replay_path"] if pair else None


def _event_wall_times(events: List[Dict[str, Any]]) -> List[float]:
    """Return cumulative Colonist wall time for every raw replay event."""
    wall_time = 0.0
    times = []
    clock_valid = True
    for event in events:
        delta = event.get("input", {}).get("deltaS", 0)
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
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Reflow rolling caption chunks without inventing text or timestamps."""
    utterances = []
    parts: List[str] = []
    source_indices: List[int] = []
    utterance_start: Optional[float] = None
    utterance_end: Optional[float] = None

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


def guarded_transcript_bounds(
    replay_data: Dict[str, Any],
    replay_index: int,
    guard_seconds: float = 3.0,
) -> Optional[Tuple[float, float]]:
    """Return the private lower bound and guarded cutoff for one replay cursor."""
    if not math.isfinite(guard_seconds) or guard_seconds < 0:
        raise ValueError("guard_seconds must be finite and nonnegative")

    transcript = replay_data.get("paired_transcript")
    if not transcript:
        return None
    timings = transcript.get("action_timings", [])
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
    replay_data: Dict[str, Any],
    replay_index: int,
    guard_seconds: float = 3.0,
) -> Tuple[Dict[str, Any], ...]:
    """Select whole commentary segments available before a guarded event cutoff.

    The result intentionally omits the upcoming action, raw event index, and
    event timestamp. It is safe to hand to a provisional contextualization pass.
    """
    bounds = guarded_transcript_bounds(replay_data, replay_index, guard_seconds)
    if bounds is None:
        return ()
    prior_time, cutoff = bounds
    transcript = replay_data["paired_transcript"]

    raw_segments = [
        segment
        for segment in transcript.get("segments", [])
        if segment["end_s"] > prior_time and segment["end_s"] <= cutoff
    ]
    return tuple(parse_transcript_segments(raw_segments))


def load_paired_transcript(
    game_id: str,
    events: List[Dict[str, Any]],
    parsed_actions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Load and index the curated transcript associated with a replay."""
    pair = _CURATED_PAIRS.get(str(game_id))
    if pair is None:
        return None

    transcript_data = json.loads(pair["transcript_path"].read_text(encoding="utf-8"))
    manifest = json.loads(pair["manifest_path"].read_text(encoding="utf-8"))
    wall_times = _event_wall_times(events)

    segments = []
    for source_index, segment in enumerate(transcript_data.get("segments", [])):
        start = segment.get("start")
        duration = segment.get("duration")
        text = segment.get("text")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(start))
            or not math.isfinite(float(duration))
            or duration < 0
            or not isinstance(text, str)
        ):
            continue
        segments.append(
            {
                "source_index": source_index,
                "start_s": float(start),
                "end_s": float(start) + float(duration),
                "text": text,
            }
        )

    action_timings = []
    for replay_index, action in enumerate(parsed_actions):
        raw_event_index = action.get("index")
        wall_time = None
        if (
            isinstance(raw_event_index, int)
            and not isinstance(raw_event_index, bool)
            and 0 <= raw_event_index < len(wall_times)
        ):
            wall_time = wall_times[raw_event_index]
        action_timings.append(
            {
                "replay_index": replay_index,
                "raw_event_index": raw_event_index,
                "wall_time_s": wall_time,
                "type": action.get("type"),
                "player": action.get("player"),
            }
        )

    players_by_color = {
        player.get("color"): player.get("username")
        for player in manifest.get("replay_metadata", {}).get("players", [])
        if isinstance(player, dict)
    }
    for action_timing in action_timings:
        action_timing["username"] = players_by_color.get(action_timing["player"])

    video = manifest.get("video", {})
    narrator = manifest.get("narrator_seat_candidate", {})
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "alignment_version": TRANSCRIPT_ALIGNMENT_VERSION,
        "video_id": transcript_data.get("video_id") or video.get("video_id"),
        "video_url": video.get("url"),
        "pairing_status": manifest.get("status", "unknown"),
        "verified": bool(manifest.get("verification", {}).get("verified", False)),
        "narrator": {
            "username": narrator.get("username"),
            "colonist_color": narrator.get("color"),
            "status": narrator.get("status"),
        },
        "segments": segments,
        "action_timings": action_timings,
    }


def paired_transcript_fingerprint(transcript: Dict[str, Any]) -> str:
    """Hash the caption and replay-timing inputs used by derived artifacts."""
    payload = {
        "schema": transcript.get("schema"),
        "alignment_version": transcript.get(
            "alignment_version", TRANSCRIPT_ALIGNMENT_VERSION
        ),
        "video_id": transcript.get("video_id"),
        "narrator": transcript.get("narrator"),
        "segments": transcript.get("segments", []),
        "action_timings": transcript.get("action_timings", []),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rounded_seconds(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def _rounded_transcript_segments(
    raw_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            **segment,
            "start_s": _rounded_seconds(segment["start_s"]),
            "end_s": _rounded_seconds(segment["end_s"]),
        }
        for segment in parse_transcript_segments(raw_segments)
    ]


def build_paired_transcript_window(
    replay_data: Dict[str, Any], replay_index: int
) -> Optional[Dict[str, Any]]:
    """Build current and cumulative causally available transcript text."""
    transcript = replay_data.get("paired_transcript")
    if not transcript:
        return None

    timings = transcript.get("action_timings", [])
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
                "window_start_s": _rounded_seconds(last_time),
                "window_end_s": None,
                "clock_anomaly": False,
                "segments": [],
                "history_raw_segment_count": 0,
                "history_segments": [],
            }

        history_raw_segments = list(transcript.get("segments", []))
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
            "window_start_s": _rounded_seconds(window_start),
            "window_end_s": _rounded_seconds(window_end),
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
        all_segments = transcript.get("segments", [])
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
