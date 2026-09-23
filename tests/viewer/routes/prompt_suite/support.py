"""A ready live sandbox for the prompt-suite route tests."""
from cle.game_engine.game import GameEngine
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox

from .conftest import COLORS


def _sandbox() -> CatanSandbox:
    engine = GameEngine(COLORS, seed=5, shuffle_players=False)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    return CatanSandbox(engine, players)
