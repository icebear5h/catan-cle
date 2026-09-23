"""Opening an existing runtime, filtering reasoning, and locked progression."""


import pytest
from playwright.sync_api import expect

from .support import (
    GAME_ID,
    MountedLiveApp,
)


@pytest.mark.parametrize("mounted_live_app", [2], indirect=True)
def test_existing_runtime_opens_at_latest_without_loading_sandbox(
    mounted_live_app: MountedLiveApp,
) -> None:
    page, _, calls, _, _ = mounted_live_app
    expect(page.get_by_role("heading", name="Step 2 reasoning history")).to_be_visible()
    expect(page.get_by_role("button", name="Step", exact=True)).to_be_enabled()
    expect(page.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
    page.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_role("heading", name="Step 1 reasoning history")).to_be_visible()
    expect(page.get_by_role("button", name="Step", exact=True)).to_be_disabled()
    assert calls["steps"] == 2


@pytest.mark.parametrize("mounted_live_app", [4], indirect=True)
def test_reasoning_player_filter_skips_other_players_without_mutating_game(
    mounted_live_app: MountedLiveApp,
) -> None:
    page, _, calls, _, _ = mounted_live_app
    for index, actor in enumerate(["RED", "BLUE", "BLUE", "RED"]):
        calls["model_calls"][index] = [{
            "step_index": index, "call_index": 0, "call_kind": "decision",
            "context_id": f"filter-context-{index}", "actor": actor, "accepted": True,
            "validation_error": None, "choice": None,
            "request": {"messages": []},
            "response": {"native_reasoning": f"{actor} reasoning {index}"},
        }]
    page.reload()
    expect(page.get_by_text("RED reasoning 3", exact=True)).to_be_visible()
    player_filter = page.get_by_label("Reasoning player")
    player_filter.select_option("RED")
    latest_board = page.locator("svg.hex-board").inner_html()
    page.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_text("RED reasoning 0", exact=True)).to_be_visible()
    expect(page.get_by_label("Saved checkpoint step")).to_have_value("0")
    expect(player_filter).to_have_value("RED")
    assert page.locator("svg.hex-board").inner_html() != latest_board
    page.get_by_role("button", name="Next saved step", exact=True).click()
    expect(page.get_by_text("RED reasoning 3", exact=True)).to_be_visible()
    assert page.locator("svg.hex-board").inner_html() == latest_board
    player_filter.select_option("WHITE")
    expect(page.get_by_text("No reasoning for WHITE in this step.", exact=True)).to_be_visible()
    page.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_text("No earlier reasoning for WHITE.", exact=True)).to_be_visible()
    expect(page.get_by_label("Saved checkpoint step")).to_have_value("3")
    player_filter.select_option("RED")
    page.get_by_label("Saved checkpoint step").select_option("1")
    expect(page.get_by_text("No reasoning for RED in this step.", exact=True)).to_be_visible()
    page.get_by_role("button", name="Next saved step", exact=True).click()
    expect(page.get_by_text("RED reasoning 3", exact=True)).to_be_visible()
    page.get_by_role("button", name="Latest", exact=True).click()
    expect(player_filter).to_have_value("RED")
    expect(page.get_by_role("button", name="Step", exact=True)).to_be_enabled()
    assert calls["steps"] == 4


def test_failed_history_get_keeps_progression_locked_until_latest(
    mounted_live_app: MountedLiveApp,
) -> None:
    page, _, calls, _, _ = mounted_live_app
    step = page.get_by_role("button", name="Step", exact=True)
    step.click()
    expect(page.get_by_role("heading", name="Step 1 reasoning history")).to_be_visible()
    expect(step).to_be_enabled()
    step.click()
    expect(page.get_by_role("heading", name="Step 2 reasoning history")).to_be_visible()
    page.route(f"**/api/live-traces/{GAME_ID}/steps/0", lambda route: route.fulfill(
        status=503, json={"error": "Checkpoint temporarily unavailable"},
        headers={"Access-Control-Allow-Origin": "*"},
    ))
    page.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_text("Checkpoint temporarily unavailable", exact=True)).to_be_visible()
    expect(step).to_be_disabled()
    page.get_by_role("button", name="Latest", exact=True).click()
    expect(step).to_be_enabled()
    assert calls["steps"] == 2
