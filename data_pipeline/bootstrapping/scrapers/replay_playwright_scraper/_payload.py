"""URL matching, payload validation, and disk writing for captured replays."""

import json
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper._config import (
    BASE_URL,
    REPLAY_DATA_PATH,
)
from data_pipeline.json_types import JsonDict, JsonValue


def replay_page_url(game_id: str, player_color: int) -> str:
    """Build the browser replay URL for a game and player perspective."""
    return f"{BASE_URL}/replay?{urlencode({'gameId': game_id, 'playerColor': player_color})}"


def replay_output_path(output_dir: Path, game_id: str) -> Path:
    """Return the canonical raw replay JSON path for a game."""
    return output_dir / f"{game_id}.json"


def is_matching_replay_response(url: str, game_id: str, player_color: int) -> bool:
    """Return True when a response URL is the replay data endpoint for the target game."""
    parsed = urlparse(url)
    if parsed.path != REPLAY_DATA_PATH:
        return False

    params = parse_qs(parsed.query)
    return (
        params.get("gameId") == [str(game_id)]
        and params.get("playerColor") == [str(player_color)]
    )


def _replay_payload(data: JsonDict) -> JsonDict:
    """Return the replay payload whether the API response is wrapped in data or not."""
    nested = data.get("data")
    return nested if isinstance(nested, dict) else data


def is_valid_replay_payload(data: JsonValue) -> bool:
    """Validate that an API response has the replay shape expected by the decoder."""
    if not isinstance(data, dict):
        return False

    payload = _replay_payload(data)
    event_history = payload.get("eventHistory")
    if not isinstance(event_history, dict):
        return False

    events = event_history.get("events")
    return isinstance(events, list) and len(events) > 0


def replay_event_count(data: JsonDict) -> int:
    """Count the events of a payload already accepted by `is_valid_replay_payload`."""
    payload = _replay_payload(data)
    event_history = payload["eventHistory"]
    if not isinstance(event_history, dict):
        raise TypeError("replay eventHistory must be a JSON object")
    events = event_history["events"]
    if not isinstance(events, list):
        raise TypeError("replay eventHistory.events must be a JSON array")
    return len(events)


def replay_player_count(data: JsonValue) -> int | None:
    """Return the source player count when replay metadata exposes it."""
    if not isinstance(data, dict):
        return None

    payload = _replay_payload(data)
    player_states = payload.get("playerUserStates")
    if isinstance(player_states, list):
        return len(player_states)

    play_order = payload.get("playOrder")
    if isinstance(play_order, list):
        return len(play_order)

    return None


def has_expected_player_count(data: JsonValue, expected_player_count: int | None) -> bool:
    """Return whether a replay matches an optional player-count constraint."""
    if expected_player_count is None:
        return True
    return replay_player_count(data) == expected_player_count


def replay_mode_setting(data: JsonValue) -> int | None:
    """Return Colonist's mode setting when replay metadata exposes it."""
    if not isinstance(data, dict):
        return None

    payload = _replay_payload(data)
    game_settings = payload.get("gameSettings")
    if not isinstance(game_settings, dict):
        return None

    mode_setting = game_settings.get("modeSetting")
    return mode_setting if isinstance(mode_setting, int) else None


def has_expected_mode_setting(data: JsonValue, expected_mode_setting: int | None) -> bool:
    """Return whether a replay matches an optional Colonist mode constraint."""
    if expected_mode_setting is None:
        return True
    return replay_mode_setting(data) == expected_mode_setting


def save_replay_json(output_file: Path, data: JsonDict) -> None:
    """Save replay JSON atomically enough to avoid leaving partial files on interruption."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temp_file.open("w") as f:
        json.dump(data, f)
    temp_file.replace(output_file)


__all__ = [
    "has_expected_mode_setting",
    "has_expected_player_count",
    "is_matching_replay_response",
    "is_valid_replay_payload",
    "replay_event_count",
    "replay_mode_setting",
    "replay_output_path",
    "replay_page_url",
    "replay_player_count",
    "save_replay_json",
]
