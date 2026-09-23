"""Shared helpers for openrouter and vllm transport request construction and surfaces."""

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.players.contracts import PlayerContext
from cle.sandbox.decision import build_decision_context

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _board_context() -> PlayerContext:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    return build_decision_context(engine)
