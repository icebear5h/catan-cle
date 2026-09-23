"""Event publication, projection, and aliasing boundaries."""
import pickle
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.events import GameEvent, event_from_action, project_event
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import (
    Action,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import ensure_trade_window, new_trade_window
from cle.game_engine.trading import TradeWindowStatus

from .support import COLORS, _message


@pytest.mark.parametrize("closed", [False, True], ids=["missing", "closed"])
def test_window_preview_is_pure_and_matches_materialized_identity(engine: GameEngine, offer_action: Action, closed: bool) -> None:
    if closed:
        engine.step(offer_action)
        engine.state.trade_window.close()
    old_window: Any = engine.state.trade_window
    before = pickle.dumps(engine)

    first = new_trade_window(engine.state)
    second = new_trade_window(engine.state)
    assert first is not second
    assert first == second
    assert first.id == f"turn-{engine.state.num_turns}-trade-{len(engine.state.actions)}"
    assert first.turn_player == Color.RED
    assert first.participants == COLORS
    assert engine.is_action_valid(offer_action)
    assert engine.is_action_valid(offer_action)
    assert engine.state.trade_window is old_window
    assert pickle.dumps(engine) == before

    _message(engine)
    assert new_trade_window(engine.state).id == first.id
    first.close()
    materialized: Any = ensure_trade_window(engine.state)
    assert materialized.id == second.id
    assert materialized.status == TradeWindowStatus.OPEN
    assert ensure_trade_window(engine.state) is materialized
    transition = engine.step(offer_action)
    assert transition.resolved_action.value.id == f"{second.id}:o1"
    if closed:
        assert materialized.id != old_window.id


def test_step_and_message_share_the_canonical_publish_seam(engine: GameEngine, offer_action: Action, monkeypatch: pytest.MonkeyPatch) -> None:
    publish = engine.publish_event
    published = []

    def record_publish(*args: object, **kwargs: object) -> GameEvent:
        event = publish(*args, **kwargs)
        published.append(event)
        return event

    monkeypatch.setattr(engine, "publish_event", record_publish)
    external = engine.publish_event("REPLAY_FACT", Color.RED, {"resources": {"WOOD": [1]}})
    message = _message(engine)
    transition = engine.step(offer_action)

    assert [event.event_type for event in published] == [
        "REPLAY_FACT",
        "MESSAGE_SENT",
        "OFFER_TRADE",
    ]
    assert [event.sequence for event in published] == [0, 1, 2]
    assert [event.causation_id for event in published] == ["action:0", "talk:boundary", "action:2"]
    assert (transition.before_revision, transition.after_revision) == (2, 3)
    assert transition.events == (published[-1],)
    assert engine.events == [external, message, transition.events[0]]
    assert len(engine.state.actions) == len(engine.history) == 1
    assert engine.history[0][2] == 2


def test_publish_event_detaches_nested_inputs_returns_and_every_projection(engine: GameEngine) -> None:
    public = {"cards": [{"WOOD": [1, 2]}]}
    private = {"cards": [{"ORE": [3]}]}
    state_before = pickle.dumps(engine.state)
    returned: Any = engine.publish_event(
        "REPLAY_FACT",
        Color.RED,
        public,
        private_overlays=((Color.BLUE, private),),
        causation_id="replay:7",
    )
    canonical = deepcopy(engine.events[0])

    public["cards"][0]["WOOD"].append(9)
    private["cards"][0]["ORE"].append(9)
    returned.public_payload["cards"][0]["WOOD"].append(8)
    returned.private_overlays[0][1]["cards"][0]["ORE"].append(8)
    red: Any = engine.project_events(Color.RED)[0]
    blue: Any = engine.project_events(Color.BLUE)[0]
    white = engine.project_events(Color.WHITE)[0]
    red.payload["cards"][0]["WOOD"].append(7)
    blue.payload["cards"][0]["ORE"].append(7)

    assert engine.events == [canonical]
    assert white.payload == {"cards": [{"WOOD": [1, 2]}]}
    assert engine.project_events(Color.BLUE)[0].payload == {"cards": [{"ORE": [3]}]}
    assert engine.revision == 1
    assert engine.history == []
    assert pickle.dumps(engine.state) == state_before


@pytest.mark.parametrize(
    "event_type, actor", [("", Color.RED), (None, Color.RED), ("FACT", Color.BLACK)]
)
def test_invalid_published_event_is_rejected_without_appending(engine: GameEngine, event_type: str | None, actor: Color) -> None:
    before = pickle.dumps(engine)
    with pytest.raises(ValueError):
        engine.publish_event(event_type, actor, {"cards": []})
    assert pickle.dumps(engine) == before


def test_event_from_action_and_projection_detach_mutable_roll_payload() -> None:
    dice = [2, 3]
    action = Action(Color.RED, ActionType.ROLL, dice)
    event: Any = event_from_action(action, 4)
    projected: Any = project_event(event, Color.RED)
    dice[0] = 6
    projected.payload[1] = 6

    assert event.public_payload == [2, 3]
    assert project_event(event, Color.BLUE).payload == [2, 3]
    assert (event.sequence, event.causation_id) == (4, "action:4")


def test_transition_actions_event_and_request_do_not_alias_live_trade(engine: GameEngine, offer_action: Action) -> None:
    transition: Any = engine.step(offer_action)
    expected_requested = deepcopy(transition.requested_action)
    expected_resolved: Any = deepcopy(transition.resolved_action)
    expected_event = deepcopy(transition.events[0])
    before = pickle.dumps(engine)

    offer_action.value.give = (4, 0, 0, 0, 0)
    assert transition.requested_action == expected_requested
    transition.requested_action.value.willing_by.add(Color.WHITE)
    transition.resolved_action.value.give = (3, 0, 0, 0, 0)
    transition.resolved_action.value.willing_by.add(Color.ORANGE)
    transition.events[0].public_payload["give"]["WOOD"] = 9
    transition.events[0].public_payload["audience"].append("WHITE")

    assert engine.state.actions[-1] == expected_resolved
    assert engine.events[-1] == expected_event
    assert pickle.dumps(engine) == before

    engine.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, expected_resolved.value.id))
    assert engine.state.trade_window.offers[expected_resolved.value.id].willing_by == {Color.BLUE}
    assert engine.state.actions[0] == expected_resolved
    assert engine.events[0] == expected_event


@pytest.mark.parametrize("field", ["requested_action", "resolved_action"])
def test_returned_roll_action_does_not_alias_materialized_state(engine: GameEngine, field: str) -> None:
    transition: Any = engine.step(Action(Color.RED, ActionType.ROLL, [1, 2]), force=True)
    before = pickle.dumps(engine)

    returned_dice = getattr(transition, field).value
    assert tuple(returned_dice) == (1, 2)
    if isinstance(returned_dice, list):
        returned_dice[0] = 6
    else:
        assert isinstance(returned_dice, tuple)

    assert tuple(engine.state.last_dice_roll) == (1, 2)
    assert tuple(engine.state.actions[-1].value) == (1, 2)
    assert tuple(transition.events[0].public_payload) == (1, 2)
    assert pickle.dumps(engine) == before
