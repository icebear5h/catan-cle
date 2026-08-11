import pytest

from engine.game import Game
from engine.models.enums import WOOD, Action, ActionType
from engine.models.player import Color, SimplePlayer
from engine.state import apply_action


def make_game():
    return Game(
        [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
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
        make_game().execute(action, force=True)


def test_apply_action_force_rejects_random_placeholders_directly():
    game = make_game()

    with pytest.raises(ValueError, match="Forced ROLL"):
        apply_action(game.state, Action(Color.RED, ActionType.ROLL, None), force=True)


def test_unforced_random_roll_returns_and_logs_resolved_action():
    game = make_game()

    action = game.execute(
        Action(Color.RED, ActionType.ROLL, None),
        validate_action=False,
    )

    assert action.action_type == ActionType.ROLL
    assert isinstance(action.value, tuple)
    assert len(action.value) == 2
    assert game.state.actions[-1] == action


def test_forced_explicit_roll_is_accepted_without_randomizing():
    game = make_game()

    action = game.execute(
        Action(Color.RED, ActionType.ROLL, (1, 2)),
        force=True,
    )

    assert action == Action(Color.RED, ActionType.ROLL, (1, 2))
    assert game.state.last_dice_roll == (1, 2)
