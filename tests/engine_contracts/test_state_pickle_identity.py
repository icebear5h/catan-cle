"""GameState must keep its historical module path so saved snapshots keep loading."""

import pickle

from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState, core


def test_game_state_pickles_under_historical_module_path() -> None:
    state = GameState([Color.RED, Color.BLUE], shuffle_players=False)
    payload = pickle.dumps(state)

    assert GameState.__module__ == "cle.game_engine.state"
    assert core.GameState is GameState
    assert b"cle.game_engine.state" in payload
    assert b"cle.game_engine.state.core" not in payload

    restored = pickle.loads(payload)
    assert isinstance(restored, GameState)
    assert restored.colors == state.colors
    assert restored.player_state == state.player_state
    assert restored.resource_freqdeck == state.resource_freqdeck
    assert restored.development_listdeck == state.development_listdeck
    assert restored.playable_actions == state.playable_actions
    assert restored.rng.getstate() == state.rng.getstate()
