"""Undo, restep, and failed-step rollback evidence."""

from copy import deepcopy

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.replay.runtime import step_executor
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.state import ServerState

from .support import COLORS, _check_checkpoint


def test_replay_undo_and_restep_preserve_events_and_rng_binding(replay: tuple[ServerState, ReplaySandbox]) -> None:
    _, sandbox = replay
    engine = sandbox.game_engine
    initial_rng = engine.rng.getstate()
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    first_events = tuple(engine.events)
    first_actions = tuple(engine.state.actions)
    first_player_events = sandbox.decision_context()[0].events
    assert len(first_actions) == len(first_events) == 1
    assert first_events[0].event_type == "BUILD_SETTLEMENT"
    assert engine.rng is engine.state.rng
    assert engine.rng.getstate() == initial_rng
    assert sandbox.undo()["status"] == "ok"
    after_undo = {
        "cursor": sandbox.replay_index,
        "actions": tuple(engine.state.actions),
        "events": tuple(engine.events),
        "player_events": sandbox.decision_context()[0].events,
        "rng_bound": engine.rng is engine.state.rng,
        "rng_preserved": engine.state.rng.getstate() == initial_rng,
    }
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    after_restep = {
        "cursor": sandbox.replay_index,
        "actions": tuple(engine.state.actions),
        "events": tuple(engine.events),
        "player_events": sandbox.decision_context()[0].events,
        "rng_bound": engine.rng is engine.state.rng,
        "rng_preserved": engine.state.rng.getstate() == initial_rng,
    }
    assert after_undo["cursor"] == 0
    assert after_undo["actions"] == ()
    assert after_restep["cursor"] == 1
    assert after_restep["actions"] == first_actions
    _check_checkpoint(
        {"undo": after_undo, "restep": after_restep},
        {
            "undo": {
                "cursor": 0, "actions": (), "events": (), "player_events": (),
                "rng_bound": True, "rng_preserved": True,
            },
            "restep": {
                "cursor": 1, "actions": first_actions, "events": first_events,
                "player_events": first_player_events,
                "rng_bound": True, "rng_preserved": True,
            },
        },
    )


def test_replay_confirmation_publishes_shared_player_event(replay: tuple[ServerState, ReplaySandbox]) -> None:
    runtime, sandbox = replay
    engine = sandbox.game_engine
    for _ in range(16):
        engine.step(engine.state.playable_actions[0])
    engine.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
    engine.state.player_state["P0_WOOD_IN_HAND"] += 1
    engine.state.player_state["P1_BRICK_IN_HAND"] += 1
    engine.state.resource_freqdeck[0] -= 1
    engine.state.resource_freqdeck[1] -= 1
    runtime.replay_data["parsed_actions"] = [{
        "index": 0, "type": "CONFIRM_TRADE", "player": 1, "acceptor": 2,
        "trade_id": "audit-trade", "offered": (1, 0, 0, 0, 0),
        "received": (0, 1, 0, 0, 0),
    }]
    before = engine.revision
    assert sandbox.step(allow_lookahead=False)["status"] == "trade_applied"
    assert engine.state.actions[-1].action_type == ActionType.CONFIRM_TRADE
    observed = [
        event.event_type for event in sandbox.decision_context()[0].events
        if event.sequence >= before
    ]
    _check_checkpoint(observed, ["CONFIRM_TRADE"])
    for color in COLORS:
        event = engine.project_game_events(color)[-1]
        assert event.actor == Color.RED
        assert event.payload == {
            "offer_id": "audit-trade", "turn_player": "RED", "counterparty": "BLUE",
            "give": {"WOOD": 1}, "receive": {"BRICK": 1},
        }
    assert sandbox.undo()["status"] == "ok"
    assert engine.revision == before
    assert sandbox.step(allow_lookahead=False)["status"] == "trade_applied"
    assert engine.revision == before + 1
    assert engine.project_game_events(Color.BLUE)[-1] == event


@pytest.mark.parametrize("raises", [False, True])
def test_failed_replay_step_rolls_back_events_rng_commitments_and_metadata(
    replay: tuple[ServerState, ReplaySandbox], monkeypatch: pytest.MonkeyPatch, raises: bool,
) -> None:
    runtime, sandbox = replay
    engine = sandbox.game_engine
    initial = engine.snapshot()
    runtime.replay_trade_ledger = {"prior": {"responses": {"2": "accepted"}}}
    ledger = deepcopy(runtime.replay_trade_ledger)

    def fail_after_mutation(
        state: ServerState, broadcast_fn: object, allow_lookahead: bool
    ) -> dict[str, str]:
        engine.step(engine.state.playable_actions[0])
        engine.rng.random()
        engine.append_message(
            speaker=Color.RED, text="must disappear", audience=COLORS,
            causation_id="failed-action",
            commitment=("offer wood", "return brick", 2),
        )
        state.replay_trade_ledger["prior"]["responses"].clear()
        state.replay_pending_dev_card = {"announcement_type": "PLAY_MONOPOLY"}
        state.replay_final_state_synced = True
        state.game_log.append({"message": "must disappear"})
        state.replay_semantic_issues.append({"kind": "must disappear"})
        state.first_divergence_step["P0"] = 0
        if raises:
            raise RuntimeError("injected replay failure")
        return {"error": "injected replay failure"}, 500

    monkeypatch.setattr(step_executor, "_replay_step_logic", fail_after_mutation)
    if raises:
        with pytest.raises(RuntimeError, match="injected replay failure"):
            sandbox.step(allow_lookahead=False)
    else:
        assert sandbox.step(allow_lookahead=False)[1] == 500
    assert engine.state.actions == initial.state.actions
    assert engine.state.player_state == initial.state.player_state
    assert tuple(engine.events) == initial.events
    assert tuple(engine.commitments) == initial.commitments
    assert engine.rng is engine.state.rng
    assert engine.rng.getstate() == initial.state.rng.getstate()
    assert engine.history == []
    assert runtime.replay_index == sandbox.revision == 0
    assert runtime.game_running is True
    assert runtime.replay_trade_ledger == ledger
    assert runtime.replay_pending_dev_card is None
    assert runtime.replay_final_state_synced is False
    assert runtime.game_log == runtime.replay_semantic_issues == []
    assert runtime.replay_step_checkpoints == runtime.replay_actions_per_step == []
    assert runtime.first_divergence_step == {}
