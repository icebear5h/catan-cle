"""Browsing saved checkpoints, and the compact gameplay controls at any width."""

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import FloatRect, expect

from .support import (
    MountedLiveApp,
)


@pytest.mark.parametrize("width", [390, 1280, 1600])
def test_live_history_board_exact_trace_usage_and_nonmutating_latest(
    mounted_live_app: MountedLiveApp, width: int, tmp_path: Path
) -> None:
    page, socketio, calls, snapshots, frames = mounted_live_app
    page.set_viewport_size({"width": width, "height": 900})
    for index, usage in enumerate([
        {"prompt_tokens": 100, "completion_tokens": 40,
         "completion_tokens_details": {"reasoning_tokens": 30}},
        {"input_tokens": 300, "output_tokens": 60},
    ]):
        calls["model_calls"][index] = [{
            "step_index": index, "call_index": 0, "call_kind": "decision",
            "context_id": f"recorded-context-{index}", "actor": "RED", "accepted": True,
            "validation_error": None, "choice": None,
            "request": {"messages": [{"role": "user", "content": f"Exact saved request {index}"}]},
            "response": {"native_reasoning": f"Exact saved reasoning {index}", "usage": usage},
        }]
    dock = page.get_by_role("region", name="Gameplay controls", exact=True)
    step = dock.get_by_role("button", name="Step", exact=True)
    step.click()
    expect(page.get_by_text("Exact saved reasoning 0", exact=True)).to_be_visible()
    first_board = page.locator("svg.hex-board").inner_html()
    expect(step).to_be_enabled()
    step.click()
    expect(page.get_by_text("Exact saved reasoning 1", exact=True)).to_be_visible()
    latest_board = page.locator("svg.hex-board").inner_html()
    assert latest_board != first_board
    expect(dock.get_by_text("Game avg 200 in / 50 out", exact=True)).to_be_visible()
    expect(dock.get_by_text("Game call coverage", exact=False)).not_to_be_visible()
    usage_gets = calls["usage_gets"]

    dock.get_by_role("button", name="Previous saved step", exact=True).click()
    expect(page.get_by_text("Exact saved reasoning 0", exact=True)).to_be_visible()
    assert page.locator("svg.hex-board").inner_html() == first_board
    expect(step).to_be_disabled()
    expect(dock.get_by_role("button", name="Auto-play live game", exact=True)).to_be_disabled()
    expect(dock.get_by_role("button", name="Previous saved step", exact=True)).to_be_disabled()
    expect(dock.get_by_text("Step 100 in / 40 out", exact=True)).to_be_visible()
    dock.get_by_text("Token details", exact=True).click()
    expect(dock.get_by_text("Game call coverage", exact=False)).to_be_visible()
    dock.get_by_text("Token details", exact=True).click()
    if width >= 1280:
        page.get_by_text("Exact request messages (1) · historical context", exact=True).click()
        expect(page.get_by_text("Exact saved request 0", exact=True)).to_be_visible()
    assert calls["usage_gets"] == usage_gets, "Browsing must not refetch game-wide usage"
    bounds = cast(FloatRect, dock.bounding_box())
    assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width
    assert bounds["y"] >= 0 and bounds["y"] + bounds["height"] <= 900
    if width == 1600:
        assert bounds["height"] < 90, "Desktop controls should fit in two compact rows"
    for control in dock.locator("button:enabled, select:enabled").all():
        control.click(trial=True)

    # An authoritative runtime update must not replace a selected historical board.
    runtime = deepcopy(snapshots[2])
    runtime["live_inference"] = {"model": "current/runtime-model", "reasoning": {},
                                 "max_tokens": None, "max_decision_attempts": 2}
    socketio.emit("game_state", runtime)
    page.wait_for_timeout(150)
    assert any("current/runtime-model" in str(frame) for frame in frames)
    assert page.locator("svg.hex-board").inner_html() == first_board

    dock.get_by_label("Saved checkpoint step").select_option("1")
    expect(page.get_by_text("Exact saved reasoning 1", exact=True)).to_be_visible()
    assert page.locator("svg.hex-board").inner_html() == latest_board
    expect(step).to_be_disabled()  # Even the latest saved checkpoint is browse-only.
    dock.get_by_role("button", name="Latest", exact=True).click()
    expect(step).to_be_enabled()
    expect(page.get_by_text("Exact saved reasoning 1", exact=True)).to_be_visible()
    assert page.locator("svg.hex-board").inner_html() == latest_board
    assert calls["steps"] == 2  # No POST /load, rewind, or duplicate step.
    assert calls["history_gets"] == [0, 1, 0, 1, 1]
    page.screenshot(path=str(tmp_path / f"compact-history-{width}.png"))


@pytest.mark.parametrize("width", [390, 1600])
def test_compact_empty_and_busy_controls(
    mounted_live_app: MountedLiveApp, width: int, tmp_path: Path
) -> None:
    page, socketio, _, snapshots, _ = mounted_live_app
    page.set_viewport_size({"width": width, "height": 900})
    dock = page.get_by_role("region", name="Gameplay controls", exact=True)
    expect(dock.get_by_label("Saved checkpoint step")).to_have_count(0)
    expect(dock.get_by_label("Recorded token usage")).to_have_count(0)
    expect(dock).not_to_contain_text("No checkpoints")
    expect(dock).not_to_contain_text("Loading checkpoint")
    runtime = deepcopy(snapshots[0])
    runtime["player_types"] = {"RED": "LLM"}
    runtime["live_inference"] = {"model": "provider/long-model-name-for-responsive-controls",
                                 "reasoning": {"effort": "high"}, "max_tokens": None}
    socketio.emit("game_state", runtime)
    expect(dock.locator(".playback-status small")).to_contain_text("provider/long-model-name")
    pending = []
    page.route("**/api/step", lambda route: pending.append(route))
    dock.get_by_role("button", name="Step", exact=True).click()
    expect(dock.get_by_role("button", name="Step", exact=True)).to_be_disabled()
    expect(dock.locator(".playback-status")).to_contain_text("Waiting for RED")
    assert dock.inner_text().count("Waiting for") == 1
    expect(dock.get_by_label("Recorded token usage")).to_have_count(0)
    assert cast(FloatRect, dock.bounding_box())["height"] < 85
    page.screenshot(path=str(tmp_path / f"compact-controls-{width}.png"))
