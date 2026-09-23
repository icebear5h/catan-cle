"""Project paths and checkpoint snapshots for replay core boundary tests."""
from copy import deepcopy
from pathlib import Path
from typing import Any

from cle.game_engine.game import GameEngine

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _checkpoint_values(game: GameEngine) -> tuple[Any, ...]:
    return deepcopy((
        game.id, game.seed, game.vps_to_win, game.state.discard_limit,
        game.events, game.commitments, game.state.actions,
        game.state.player_state, game.state.resource_freqdeck,
        game.state.development_listdeck, game.state.playable_actions,
        game.state.board.roads, game.state.board.buildings,
        game.state.current_prompt, game.state.current_player_index,
        game.state.current_turn_index, game.state.num_turns,
        game.rng.getstate(), len(game.history),
    ))
