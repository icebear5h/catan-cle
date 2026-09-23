import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from data_pipeline.bootstrapping.scrapers import replay_playwright_scraper
from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper import (
    ReplayRateLimitedError,
    has_expected_mode_setting,
    has_expected_player_count,
    is_matching_replay_response,
    is_valid_replay_payload,
    replay_mode_setting,
    replay_output_path,
    replay_page_url,
    replay_player_count,
    scrape_replays_from_index,
)


def test_replay_page_url_contains_game_and_player_color() -> None:
    assert (
        replay_page_url("228953487", 1)
        == "https://colonist.io/replay?gameId=228953487&playerColor=1"
    )


def test_is_matching_replay_response_requires_exact_endpoint_and_params() -> None:
    assert is_matching_replay_response(
        "https://colonist.io/api/replay/data-from-game-id?gameId=228953487&playerColor=1",
        "228953487",
        1,
    )

    assert not is_matching_replay_response(
        "https://colonist.io/api/profile/coincoincoin/history",
        "228953487",
        1,
    )
    assert not is_matching_replay_response(
        "https://colonist.io/api/replay/data-from-game-id?gameId=228953487&playerColor=2",
        "228953487",
        1,
    )


def test_is_valid_replay_payload_accepts_nested_and_top_level_shapes() -> None:
    nested: Any = {"data": {"eventHistory": {"events": [{"stateChange": {}}]}}}
    top_level: Any = {"eventHistory": {"events": [{"stateChange": {}}]}}

    assert is_valid_replay_payload(nested)
    assert is_valid_replay_payload(top_level)


def test_is_valid_replay_payload_rejects_missing_or_empty_events() -> None:
    assert not is_valid_replay_payload({"data": {"eventHistory": {"events": []}}})
    assert not is_valid_replay_payload({"data": {"eventHistory": {}}})
    assert not is_valid_replay_payload({"data": {}})
    assert not is_valid_replay_payload(None)


def test_replay_player_count_supports_nested_and_top_level_shapes() -> None:
    nested: Any = {
        "data": {
            "playerUserStates": [{}, {}, {}, {}],
            "gameSettings": {"modeSetting": 0},
            "eventHistory": {"events": [{}]},
        }
    }
    top_level: Any = {
        "playOrder": [1, 2],
        "eventHistory": {"events": [{}]},
    }

    assert replay_player_count(nested) == 4
    assert replay_player_count(top_level) == 2
    assert replay_player_count({"eventHistory": {"events": [{}]}}) is None
    assert has_expected_player_count(nested, 4)
    assert not has_expected_player_count(nested, 2)
    assert has_expected_player_count(nested, None)
    assert replay_mode_setting(nested) == 0
    assert replay_mode_setting(top_level) is None
    assert has_expected_mode_setting(nested, 0)
    assert not has_expected_mode_setting(nested, 6)
    assert has_expected_mode_setting(nested, None)


def test_batch_stops_immediately_when_rate_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_file = tmp_path / "queue.json"
    index_file.write_text(json.dumps([
        {"game_id": "1", "player_color": 1},
        {"game_id": "2", "player_color": 2},
    ]))

    class RateLimitedScraper:
        calls = []

        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "RateLimitedScraper":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def capture_replay_data(self, game_id: str, player_color: str) -> None:
            self.calls.append((game_id, player_color))
            raise ReplayRateLimitedError(game_id, "600")

    monkeypatch.setattr(
        replay_playwright_scraper,
        "PlaywrightReplayScraper",
        RateLimitedScraper,
    )
    stats = asyncio.run(scrape_replays_from_index(
        index_file=index_file,
        output_dir=tmp_path / "output",
        profile_dir=tmp_path / "profile",
        max_games=2,
        max_attempts=2,
        headless=True,
        skip_existing=True,
        cdp_url=None,
        expected_player_count=4,
    ))

    assert stats["rate_limited"] == 1
    assert stats["success"] == 0
    assert RateLimitedScraper.calls == [("1", 1)]


def test_replay_output_path_uses_game_id_json_filename() -> None:
    assert replay_output_path(Path("/tmp/replays"), "228953487") == Path(
        "/tmp/replays/228953487.json"
    )
