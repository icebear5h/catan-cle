"""Shared fixtures for harness boundary defects surfaced against a local sandbox."""

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox, RetryPolicy

from .support import COLORS, QuietCommunication


@pytest.fixture
def sandbox() -> CatanSandbox:
    return CatanSandbox(
        GameEngine(COLORS, seed=7, shuffle_players=False),
        {color: FirstLegalPlayer(color) for color in COLORS},
        retry_policy=RetryPolicy(2),
        communication_policy=QuietCommunication(),
    )


@pytest.fixture
def trade_sandbox(sandbox: CatanSandbox) -> CatanSandbox:
    state = sandbox.game_engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_HAS_ROLLED"] = True
    state.player_state["P0_WOOD_IN_HAND"] = 4
    state.resource_freqdeck[0] -= 4
    for index in range(1, 4):
        state.player_state[f"P{index}_ORE_IN_HAND"] = 1
        state.resource_freqdeck[4] -= 1
    state.playable_actions = generate_playable_actions(state)
    return sandbox
