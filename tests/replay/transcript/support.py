"""Curated paired-transcript fixture shared by the replay transcript tests."""
import json
from typing import Any

from playground.game_viewer.replay.transcript import (
    get_curated_replay_path,
    load_paired_transcript,
)


def _paired_fixture() -> dict[str, dict[str, Any]]:
    replay_path = get_curated_replay_path("242781000")
    assert replay_path is not None
    raw_data = json.loads(replay_path.read_text(encoding="utf-8"))
    events = raw_data["data"]["eventHistory"]["events"]
    parsed_actions = [
        {"index": 0, "type": "BUILD_SETTLEMENT", "player": 2},
        {"index": 1, "type": "BUILD_ROAD", "player": 2},
        {"index": 4, "type": "BUILD_SETTLEMENT", "player": 5},
    ]
    transcript = load_paired_transcript("242781000", events, parsed_actions)
    assert transcript is not None
    return {"paired_transcript": transcript}
