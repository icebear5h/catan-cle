"""The local replay corpus keeps exact resources and trade lifecycles."""
import contextlib
import io
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from cle.replay.colonist.helpers import validate_resources_match
from playground.game_viewer.app import app
from playground.game_viewer.state import server_state

from .support import (
    RESOURCES,
    assert_replay_ledger_matches_raw,
    raw_active_offers_by_event,
)


@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_REPLAY_CORPUS") != "1",
    reason="set RUN_LOCAL_REPLAY_CORPUS=1 for the local replay audit",
)
def test_all_local_replays_have_exact_resources_and_trade_lifecycle() -> None:
    replay_dir = Path("artifacts/raw/colonist/replays")
    replay_files: Any = sorted(replay_dir.glob("*.json"))
    requested_game_ids = {
        game_id
        for game_id in os.getenv("LOCAL_REPLAY_GAME_IDS", "").split(",")
        if game_id
    }
    requested_game_id = os.getenv("LOCAL_REPLAY_GAME_ID")
    if requested_game_id:
        requested_game_ids.add(requested_game_id)

    if requested_game_ids:
        replay_files: Any = [
            path for path in replay_files if path.stem in requested_game_ids
        ]
        assert {path.stem for path in replay_files} == requested_game_ids
    else:
        expected_count = int(
            os.getenv("EXPECTED_LOCAL_REPLAY_COUNT", "18")
        )
        assert len(replay_files) == expected_count

    total_actions: Any = 0
    trade_counts: Any = Counter()
    trade_types: Any = {
        "OFFER_TRADE",
        "COUNTER_OFFER",
        "ACCEPT_TRADE",
        "REJECT_TRADE",
        "CLEAR_TRADE_RESPONSE",
        "CLOSE_TRADE",
        "CONFIRM_TRADE",
        "MARITIME_TRADE",
    }

    with app.test_client() as client:
        for replay_file in replay_files:
            output: Any = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                load_response: Any = client.post(
                    "/api/load-replay",
                    json={"game_id": replay_file.stem},
                )
            assert load_response.status_code == 200, output.getvalue()[-4_000:]

            parsed_actions: Any = server_state.replay_data["parsed_actions"]
            raw_events: Any = server_state.replay_data["events"]
            raw_trade_snapshots: Any = raw_active_offers_by_event(raw_events)
            total_actions += len(parsed_actions)
            trade_counts.update(
                action["type"]
                for action in parsed_actions
                if action.get("type") in trade_types
            )

            for action_index, action_hint in enumerate(parsed_actions):
                output: Any = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    step_response: Any = client.post("/api/replay-step")
                assert step_response.status_code == 200, (
                    replay_file.stem,
                    action_index,
                    action_hint.get("index"),
                    action_hint.get("type"),
                    output.getvalue()[-4_000:],
                )

                expected_resources: Any = action_hint.get("expected_resources", {})
                if expected_resources:
                    mismatches: Any = validate_resources_match(
                        server_state.current_sandbox.game_engine,
                        expected_resources,
                        server_state.replay_data[
                            "colonist_color_to_engine_idx"
                        ],
                    )
                    assert not mismatches, (
                        replay_file.stem,
                        action_index,
                        action_hint.get("type"),
                        mismatches,
                    )

                game_state: Any = server_state.current_sandbox.game_engine.state
                for player_idx in range(len(game_state.colors)):
                    hand: Any = [
                        game_state.player_state[
                            f"P{player_idx}_{resource}_IN_HAND"
                        ]
                        for resource in RESOURCES
                    ]
                    assert all(count >= 0 for count in hand), (
                        replay_file.stem,
                        action_index,
                        player_idx,
                        hand,
                    )

                for resource_idx, resource in enumerate(RESOURCES):
                    resource_total: Any = game_state.resource_freqdeck[resource_idx]
                    resource_total += sum(
                        game_state.player_state[
                            f"P{player_idx}_{resource}_IN_HAND"
                        ]
                        for player_idx in range(len(game_state.colors))
                    )
                    assert resource_total == 19, (
                        replay_file.stem,
                        action_index,
                        resource,
                        resource_total,
                    )

                raw_event_index: Any = action_hint["index"]
                next_raw_event_index: Any = (
                    parsed_actions[action_index + 1]["index"]
                    if action_index + 1 < len(parsed_actions)
                    else None
                )
                if next_raw_event_index != raw_event_index:
                    assert_replay_ledger_matches_raw(
                        server_state,
                        raw_trade_snapshots[raw_event_index],
                        (
                            replay_file.stem,
                            action_index,
                            raw_event_index,
                            action_hint.get("type"),
                        ),
                    )

            semantic_errors: Any = [
                issue
                for issue in server_state.replay_semantic_issues
                if issue.get("severity") == "error"
            ]
            assert not semantic_errors, (replay_file.stem, semantic_errors)
            assert server_state.replay_index == len(parsed_actions)

    print(
        f"Audited {len(replay_files)} replays, {total_actions} actions, "
        f"and {sum(trade_counts.values())} trade lifecycle actions: "
        f"{dict(sorted(trade_counts.items()))}"
    )
