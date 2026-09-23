from typing import Any

import pytest

from cle.game_engine.events import event_from_action, project_event
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeCandidate, TradeOffer

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _engine(seed: int = 7) -> GameEngine:
    return GameEngine(COLORS, seed=seed, shuffle_players=False)


def test_engine_step_appends_one_canonical_event_and_projects_it() -> None:
    engine = _engine()
    action = engine.state.playable_actions[0]

    transition = engine.step(action)

    assert transition.before_revision == 0
    assert transition.after_revision == 1
    assert transition.resolved_action == engine.state.actions[-1]
    assert engine.events == list(transition.events)
    assert engine.project_events(Color.RED)[0].sequence == 0
    assert engine.project_events(Color.BLUE)[0].event_type == action.action_type.value


def test_private_event_overlays_never_enter_other_player_projection() -> None:
    bought: Any = event_from_action(
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT"),
        0,
    )
    stolen: Any = event_from_action(
        Action(Color.RED, ActionType.STEAL, (Color.BLUE, "ORE")),
        1,
    )
    discarded: Any = event_from_action(
        Action(Color.RED, ActionType.DISCARD, ("WOOD", "BRICK")),
        2,
    )

    assert project_event(bought, Color.RED).payload == "KNIGHT"
    assert project_event(bought, Color.BLUE).payload is None
    assert project_event(stolen, Color.BLUE).payload == (Color.BLUE, "ORE")
    assert project_event(stolen, Color.WHITE).payload == (Color.BLUE, None)
    assert project_event(discarded, Color.RED).payload == ("WOOD", "BRICK")
    assert project_event(discarded, Color.BLUE).payload == 2


def test_trade_events_expose_named_semantic_payloads() -> None:
    offered_value = TradeOffer(
        id="window:o1",
        offered_by=Color.RED,
        audience=frozenset(COLORS[1:]),
        give=(1, 0, 0, 0, 0),
        receive=(0, 0, 0, 0, 1),
    )
    counter_value = TradeOffer(
        id="window:o2",
        offered_by=Color.BLUE,
        audience=frozenset({Color.RED}),
        give=(0, 0, 0, 0, 1),
        receive=(1, 0, 0, 0, 0),
        parent_offer_id="window:o1",
    )
    offered: Any = event_from_action(
        Action(Color.RED, ActionType.OFFER_TRADE, offered_value),
        0,
    )
    countered: Any = event_from_action(
        Action(Color.BLUE, ActionType.COUNTER_OFFER, counter_value),
        1,
    )

    assert offered.public_payload["give"] == {"WOOD": 1}
    assert offered.public_payload["receive"] == {"ORE": 1}
    confirmed = event_from_action(
        Action(
            Color.RED,
            ActionType.CONFIRM_TRADE,
            TradeCandidate("window:o1", Color.RED, Color.BLUE),
        ),
        2,
    )

    assert countered.public_payload["parent_offer_id"] == "window:o1"
    assert countered.public_payload["offered_by"] == "BLUE"
    assert confirmed.public_payload == {
        "offer_id": "window:o1",
        "turn_player": "RED",
        "counterparty": "BLUE",
    }


def test_engine_observation_exposes_legal_actions_only_to_current_player() -> None:
    engine = _engine()

    red = engine.observe(Color.RED)
    blue = engine.observe(Color.BLUE)

    assert red.valid_actions == engine.state.playable_actions
    assert blue.valid_actions == []
    assert red.my_resources
    assert set(red.opponent_resource_counts) == {
        Color.BLUE,
        Color.WHITE,
        Color.ORANGE,
    }


def test_snapshot_restore_includes_rng_state_and_event_log() -> None:
    engine = _engine(23)
    engine.step(engine.state.playable_actions[0])
    snapshot = engine.snapshot()
    expected_state = snapshot.state.copy()
    expected_events = snapshot.events

    engine.step(engine.state.playable_actions[0])
    engine.restore(snapshot)

    assert engine.revision == 1
    assert tuple(engine.events) == expected_events
    assert engine.state.actions == expected_state.actions
    assert engine.rng.getstate() == expected_state.rng.getstate()


def test_live_step_does_not_copy_full_state(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()

    def fail_copy() -> None:
        raise AssertionError("live step copied the full state")

    monkeypatch.setattr(engine.state, "copy", fail_copy)
    engine.step(engine.state.playable_actions[0])

    assert engine.revision == 1
