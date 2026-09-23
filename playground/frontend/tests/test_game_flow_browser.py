"""Game-tab browser flows over a recorded session, isolated and inference-free.

Run with uv run --no-sync python -m pytest playground/frontend/tests/test_game_flow_browser.py.
Requires installed frontend dependencies and Playwright Chromium.
"""

from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import playwright.sync_api as pw
import pytest
from playwright.sync_api import expect

from playground.frontend.tests.browser_support import fixture_server, network_sandbox


@pytest.fixture
def saved_session(browser_build: tuple[pw.Browser, Path], viewport: pw.ViewportSize,
                  tmp_path: Path) -> Iterator[pw.Page]:
    browser, build = browser_build
    with (fixture_server(build, "session", tmp_path) as origin,
          network_sandbox(browser, origin, viewport, tmp_path) as sandbox):
        yield sandbox.open_page()


def test_new_game_returns_to_empty_setup_and_keeps_saved_session(saved_session: pw.Page) -> None:
    page = saved_session
    origin = page.url
    assert urlsplit(origin).port != 5001
    before = page.request.get(f"{origin}api/live-traces").json()["games"]
    assert len(before) == 1 and before[0]["step_count"] == 1
    old_id = before[0]["game_id"]
    page.get_by_role("button", name="Game", exact=True).click()
    with page.expect_response(lambda response: response.url.endswith("/api/reset")) as reset:
        page.get_by_role("button", name="New Game", exact=True).click()
    assert reset.value.status == 200
    expect(page.get_by_text("Start a game to view the board", exact=True)).to_be_visible()
    start = page.get_by_role("button", name="Start Game (Random)", exact=True)
    expect(start).to_be_enabled()
    assert page.request.get(f"{origin}api/live-traces/{old_id}").json()["step_count"] == 1
    with page.expect_response(lambda response: response.url.endswith("/api/start-game")) as created:
        start.click()
    assert created.value.status == 200
    assert created.value.json()["trace_game_id"] != old_id
    expect(page.get_by_role("button", name="New Game", exact=True)).to_be_enabled()
    expect(start).to_have_count(0)
    games = page.request.get(f"{origin}api/live-traces").json()["games"]
    assert len(games) == 2
    assert next(game for game in games if game["game_id"] == old_id)["step_count"] == 1
