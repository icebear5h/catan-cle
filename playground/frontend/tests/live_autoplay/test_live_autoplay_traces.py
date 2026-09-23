"""How one recorded reasoning trace is presented."""


import pytest
from playwright.sync_api import expect

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color

from .support import (
    MountedLiveApp,
)


@pytest.mark.parametrize("immediate_victory", [False, True])
def test_committed_knight_sequence_not_requested_destination(
    mounted_live_app: MountedLiveApp, immediate_victory: bool
) -> None:
    page, _, calls, _, _ = mounted_live_app
    # Presentation fixture: the request has a destination in both cases, but
    # an immediate Knight victory commits no movement.
    sequence = [str(Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None))]
    if not immediate_victory:
        sequence.append(str(Action(Color.RED, ActionType.MOVE_ROBBER, (0, 0, 0))))
    calls["trace_game_id"] = None
    calls["reasoning_traces"] = [{
        "schema": "live-reasoning-trace-v2",
        "context_id": "knight-browser-test",
        "player_color": "RED",
        "action_type": "PLAY_KNIGHT_CARD",
        "action_index": 1,
        "action_sequence": sequence,
        "knight_destination": [0, 0, 0],
        "model": "local/presentation-fixture",
        "game_plan": "Move the robber with the Knight.",
        "native_reasoning_returned": False,
        "native_reasoning_requested": False,
        "native_reasoning_missing": False,
        "reasoning_tokens": None,
        "reasoning_request": {},
        "usage": {},
    }]

    page.get_by_role("button", name="Step", exact=True).click()

    panel = page.get_by_role("region", name="Live model reasoning, private notes, and historical game plans")
    committed = panel.locator(".live-reasoning-section").filter(has_text="Committed actions")
    expect(committed.locator("pre")).to_have_text("\n".join(sequence))
    if immediate_victory:
        expect(committed).not_to_contain_text("MOVE_ROBBER")


@pytest.mark.parametrize("fresh", [False, True], ids=["historical-long", "fresh"])
def test_exact_request_display_keeps_all_recorded_messages(
    mounted_live_app: MountedLiveApp, fresh: bool
) -> None:
    page, _, calls, _, _ = mounted_live_app
    # Deliberately long historical presentation fixture: every recorded message,
    # including the oldest one, must remain inspectable without context filtering.
    messages = [
        {"role": "user" if index % 2 else "assistant", "content": f"Recorded request message {index:03d}"}
        for index in range(2 if fresh else 82)
    ]
    calls["trace_game_id"] = None
    calls["reasoning_traces"] = [{
        "schema": "live-reasoning-trace-v2", "context_id": "exact-request",
        "player_color": "RED", "action_type": "BUILD_SETTLEMENT", "action_index": 0,
        "model": "local/presentation-fixture", "native_reasoning_returned": False,
        "native_reasoning_requested": False, "native_reasoning_missing": False,
        "reasoning_tokens": None, "reasoning_request": {}, "usage": {},
        "request": {
            "decision_id": "exact-request", "session_id": "request-display",
            "context_policy": "fresh_notes" if fresh else None,
            "messages": messages,
            "board_presentation": {"kind": "text", "content": "Recorded complete board attachment"},
        },
    }]
    page.get_by_role("button", name="Step", exact=True).click()
    panel = page.get_by_role("region", name="Live model reasoning, private notes, and historical game plans")
    detail = panel.locator("details").filter(has=page.locator("summary", has_text="Exact request messages"))
    detail.locator("summary").click()
    expect(detail.locator("summary")).to_contain_text(f"({len(messages)})")
    expect(detail.locator("summary")).to_contain_text("fresh context + notes" if fresh else "historical context")
    texts = detail.locator("pre").all_text_contents()
    assert texts[:-1] == [message["content"] for message in messages]
    assert "Recorded complete board attachment" in texts[-1]
