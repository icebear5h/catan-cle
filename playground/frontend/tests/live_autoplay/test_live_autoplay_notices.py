"""What a socket notice does to a paused auto-play run."""

from collections.abc import Mapping
from copy import deepcopy
from typing import cast

import pytest
from playwright.sync_api import expect

from cle.game_engine.public_board import JsonValue

from .support import (
    FAILURE,
    PERSISTENCE_FAILURE,
    PROVIDER_FAILURE,
    WARNING,
    MountedLiveApp,
)


@pytest.mark.parametrize(
    "notice_kind", ["warning", "failure", "provider", "persistence", "null", "checkpoint"],
)
def test_socket_notice_during_autoplay_pause(
    mounted_live_app: MountedLiveApp, notice_kind: str
) -> None:
    page, socketio, calls, snapshots, frames = mounted_live_app
    if notice_kind == "checkpoint":
        calls["checkpoint_notice"] = WARNING
    page.get_by_role("button", name="Auto-play live game", exact=True).click()
    page.wait_for_function("window.__autoPlayPauses === 1")
    assert calls["steps"] == 1

    notice = {
        "warning": WARNING, "failure": FAILURE, "provider": PROVIDER_FAILURE,
        "persistence": PERSISTENCE_FAILURE,
    }.get(notice_kind)
    if notice_kind != "checkpoint":
        runtime = deepcopy(snapshots[1])
        runtime["last_live_step_error"] = cast(JsonValue, notice)
        socketio.emit("game_state", runtime)

    if notice_kind == "persistence":
        details = str(cast(Mapping[str, object], notice)["details"])
        expect(page.get_by_role("alert").first).to_have_text(details)
        page.wait_for_timeout(900)
        assert calls["steps"] == 1, "An un-checkpointed failure must cancel the next automatic step"
        expect(page.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
        assert any(details in str(frame) for frame in frames)
    elif notice is not None:
        # Checkpointed notices from any tab are shown but do not end auto-play:
        # the next Step retries a rejected decision or advances past an applied one.
        expect(page.get_by_role("alert").first).to_have_text(str(notice["details"]))
        page.wait_for_function("window.__autoPlayPauses === 2")
        assert calls["steps"] == 2
        page.get_by_role("button", name="Stop auto-play after the current step", exact=True).click()
        expect(page.get_by_role("button", name="Auto-play live game", exact=True)).to_be_enabled()
        assert any(str(notice["details"]) in str(frame) for frame in frames)
    else:
        page.wait_for_function("window.__autoPlayPauses === 2")
        assert calls["steps"] == 2
        page.get_by_role("button", name="Stop auto-play after the current step", exact=True).click()
        expect(page.get_by_role("alert")).to_have_count(0)
