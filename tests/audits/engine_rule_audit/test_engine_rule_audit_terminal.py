"""Terminal-game menu, context, and rejection evidence."""
import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import (
    Action,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox, TerminalSandboxError

from .support import COLORS, _assert_inventory, _expect_rule


def test_terminal_observation_has_no_legal_menu(terminal_game: GameEngine) -> None:
    sandbox = CatanSandbox(terminal_game, {color: FirstLegalPlayer(color) for color in COLORS})
    views = {color: sandbox.view(color) for color in COLORS}
    assert all(view.winner == Color.RED for view in views.values())
    _assert_inventory(terminal_game)
    observed = {
        color: (tuple(view.legal_actions), tuple(view.observation.valid_actions))
        for color, view in views.items()
    }
    _expect_rule(observed, {color: ((), ()) for color in COLORS})


def test_terminal_game_has_no_decision_context(terminal_game: GameEngine) -> None:
    sandbox = CatanSandbox(terminal_game, {color: FirstLegalPlayer(color) for color in COLORS})
    revision = sandbox.revision
    rejected = False
    context = None
    try:
        context = sandbox.decision_context()
    except (ValueError, TerminalSandboxError):
        rejected = True
    _assert_inventory(terminal_game)
    observed = {
        "rejected": rejected,
        "revision_delta": sandbox.revision - revision,
        "returned_menu": tuple(context.legal_actions) if context is not None else (),
    }
    _expect_rule(observed, {"rejected": True, "revision_delta": 0, "returned_menu": ()})


def test_terminal_engine_rejects_further_actions(terminal_game: GameEngine) -> None:
    revision = terminal_game.revision
    board = terminal_game.state.board
    before_board = (dict(board.buildings), dict(board.roads))
    assert terminal_game.state.current_color() == Color.RED
    rejected = False
    try:
        terminal_game.step(Action(Color.RED, ActionType.END_TURN, None))
    except ValueError:
        rejected = True
    _assert_inventory(terminal_game)
    board = terminal_game.state.board
    observed = {
        "rejected": rejected,
        "revision_delta": terminal_game.revision - revision,
        "board_unchanged": (dict(board.buildings), dict(board.roads)) == before_board,
        "current_color": terminal_game.state.current_color(),
    }
    _expect_rule(
        observed,
        {
            "rejected": True,
            "revision_delta": 0,
            "board_unchanged": True,
            "current_color": Color.RED,
        },
    )


@pytest.mark.asyncio
async def test_terminal_sandbox_rejects_step_without_committing(terminal_game: GameEngine) -> None:
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    sandbox = CatanSandbox(terminal_game, players)
    revision = sandbox.revision
    with pytest.raises(TerminalSandboxError):
        await sandbox.step()
    assert sandbox.revision == revision
    assert all(player.accepted_choices == 0 for player in players.values())
