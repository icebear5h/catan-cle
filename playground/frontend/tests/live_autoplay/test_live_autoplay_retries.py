"""Auto-play retry behaviour after a rejected or failed step."""


import pytest
from playwright.sync_api import Route, expect

from .support import (
    FAILURE,
    GAME_ID,
    MountedLiveApp,
)


@pytest.mark.parametrize("status,payload", [
    (422, {**FAILURE, "trace_game_id": GAME_ID, "checkpoint_saved": True}),
    (500, {"error": "Sandbox step failed", "details": "ValueError. No gameplay action was applied.",
           "trace_game_id": GAME_ID, "checkpoint_saved": True, "action_applied": False, "retryable": False}),
])
def test_failed_step_is_retried_until_it_succeeds(
    mounted_live_app: MountedLiveApp, status: int, payload: dict[str, object]
) -> None:
    page, _, calls, _, _ = mounted_live_app
    rejected: list[str] = []

    def reject_twice(route: Route) -> None:
        if len(rejected) < 2:
            rejected.append(route.request.url)
            route.fulfill(status=status, json=payload, headers={"Access-Control-Allow-Origin": "*"})
        else:
            route.fallback()

    page.route("**/api/step", reject_twice)
    dock = page.get_by_role("region", name="Gameplay controls", exact=True)
    dock.get_by_role("button", name="Auto-play live game", exact=True).click()
    expect(page.get_by_role("alert").first).to_have_text(str(payload["details"]))
    expect(dock.get_by_role("status")).to_contain_text("Auto-play retry 1")
    expect(dock.locator(".playback-status")).to_contain_text("Auto-play retrying")
    expect(dock.get_by_role("status")).to_contain_text("Auto-play retry 2", timeout=5000)
    page.wait_for_function("window.__autoPlayPauses === 1", timeout=10000)
    assert len(rejected) == 2
    assert calls["steps"] == 1, "The first successful step follows the retries"
    expect(dock.get_by_role("status")).to_have_count(0)
    expect(page.get_by_role("alert")).to_have_count(0)
    page.wait_for_function("window.__autoPlayPauses === 2")
    assert calls["steps"] == 2
    dock.get_by_role("button", name="Stop auto-play after the current step", exact=True).click()
    expect(dock.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()


def test_stop_during_retry_wait_cancels_the_retry(
    mounted_live_app: MountedLiveApp,
) -> None:
    page, _, calls, _, _ = mounted_live_app
    page.route("**/api/step", lambda route: route.fulfill(
        status=422, json={**FAILURE, "trace_game_id": GAME_ID, "checkpoint_saved": True},
        headers={"Access-Control-Allow-Origin": "*"},
    ))
    dock = page.get_by_role("region", name="Gameplay controls", exact=True)
    dock.get_by_role("button", name="Auto-play live game", exact=True).click()
    expect(dock.get_by_role("status")).to_contain_text("Auto-play retry 3", timeout=10000)
    dock.get_by_role("button", name="Stop auto-play after the current step", exact=True).click()
    expect(dock.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
    expect(dock.get_by_role("status")).to_have_count(0)
    page.wait_for_timeout(4500)
    expect(dock.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
    assert calls["steps"] == 0
