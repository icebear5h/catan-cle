import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import WOOD, Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state import apply_action


def make_game():
    return GameEngine(
        [
            Color.RED,
            Color.BLUE,
            Color.WHITE,
            Color.ORANGE,
        ],
        shuffle_players=False,
    )


@pytest.mark.parametrize(
    "action",
    [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.DISCARD, None),
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.RED, ActionType.STEAL, (Color.BLUE, None)),
        Action(Color.RED, ActionType.MOVE_ROBBER, None),
        Action(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, (WOOD, None)),
        Action(Color.RED, ActionType.PLAY_MONOPOLY, None),
    ],
)
def test_forced_actions_reject_random_or_missing_values(action):
    with pytest.raises(ValueError, match="Forced"):
        make_game().step(action, force=True)


def test_apply_action_force_rejects_random_placeholders_directly():
    game = make_game()

    with pytest.raises(ValueError, match="Forced ROLL"):
        apply_action(game.state, Action(Color.RED, ActionType.ROLL, None), force=True)


def test_unforced_random_roll_returns_and_logs_resolved_action():
    game = make_game()

    transition = game.step(
        Action(Color.RED, ActionType.ROLL, None),
        validate_action=False,
    )
    action = transition.resolved_action

    assert action.action_type == ActionType.ROLL
    assert isinstance(action.value, tuple)
    assert len(action.value) == 2
    assert game.state.actions[-1] == action


def test_forced_explicit_roll_is_accepted_without_randomizing():
    game = make_game()

    transition = game.step(
        Action(Color.RED, ActionType.ROLL, (1, 2)),
        force=True,
    )

    assert transition.resolved_action == Action(Color.RED, ActionType.ROLL, (1, 2))
    assert game.state.last_dice_roll == (1, 2)
