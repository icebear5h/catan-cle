"""Exact-discard validation boundaries."""
import pickle

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import (
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import validate_discard
from cle.game_engine.state_functions import (
    get_player_freqdeck,
)

from .support import COLORS


def test_exact_discard_validation_step_and_private_event_agree(engine: GameEngine) -> None:
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    cards = ["WOOD", "BRICK", "WOOD", "BRICK"]
    action = Action(Color.RED, ActionType.DISCARD, cards)
    assert action not in engine.state.playable_actions
    assert engine.state.playable_actions == [Action(Color.RED, ActionType.DISCARD, None)]
    before = pickle.dumps(engine)

    assert validate_discard(engine.state, action) == tuple(cards)
    assert engine.is_action_valid(action)
    assert engine.is_action_valid(action)
    assert pickle.dumps(engine) == before
    rng = engine.rng.getstate()
    transition = engine.step(action)

    assert transition.requested_action.value == cards
    assert transition.resolved_action.value == tuple(cards)
    assert get_player_freqdeck(engine.state, Color.RED) == [3, 2, 0, 0, 0]
    assert engine.state.resource_freqdeck == [16, 17, 19, 19, 16]
    assert engine.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert engine.project_events(Color.RED)[-1].payload == tuple(cards)
    assert all(engine.project_events(color)[-1].payload == 4 for color in COLORS[1:])
    assert engine.rng.getstate() == rng

    cards[0] = "ORE"
    transition.requested_action.value[1] = "ORE"
    assert engine.history[-1][1].value == ["WOOD", "BRICK", "WOOD", "BRICK"]
    assert engine.state.actions[-1].value == ("WOOD", "BRICK", "WOOD", "BRICK")
    assert engine.project_events(Color.RED)[-1].payload == engine.state.actions[-1].value


@pytest.mark.parametrize(
    "cards",
    [
        "WOOD",
        {"WOOD": 4},
        [True] * 4,
        [[]] * 4,
        ["GOLD"] * 4,
        ["WOOD"] * 3,
        ["WOOD"] * 5,
        ["ORE"] * 4,
    ],
)
def test_invalid_exact_discard_is_rejected_before_engine_history(
    engine: GameEngine, cards: object
) -> None:
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    before = pickle.dumps(engine)
    action = Action(Color.RED, ActionType.DISCARD, cards)

    with pytest.raises(ValueError):
        validate_discard(engine.state, action)
    assert not engine.is_action_valid(action)
    with pytest.raises(ValueError, match="not playable"):
        engine.step(action)

    assert pickle.dumps(engine) == before
