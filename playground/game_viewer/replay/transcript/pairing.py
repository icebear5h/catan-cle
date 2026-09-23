"""Loading one curated transcript and indexing it against replay events."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from .pairs import TRANSCRIPT_ALIGNMENT_VERSION, TRANSCRIPT_SCHEMA, curated_pair
from .segments import _event_wall_times

__all__ = ["load_paired_transcript", "paired_transcript_fingerprint"]


def load_paired_transcript(
    game_id: str,
    events: Sequence[Mapping[str, object]],
    parsed_actions: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Load and index the curated transcript associated with a replay."""
    pair = curated_pair(game_id)
    if pair is None:
        return None

    transcript_data = json.loads(pair["transcript_path"].read_text(encoding="utf-8"))
    manifest = json.loads(pair["manifest_path"].read_text(encoding="utf-8"))
    wall_times = _event_wall_times(events)

    segments: list[dict[str, object]] = []
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

    action_timings: list[dict[str, object]] = []
    for replay_index, action in enumerate(parsed_actions):
        raw_event_index = action.get("index")
        wall_time: float | None = None
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


def paired_transcript_fingerprint(transcript: Mapping[str, object]) -> str:
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
