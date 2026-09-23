"""Shared fixtures for harness context suites, prompts, parsing, and discard contracts."""

from dataclasses import replace

import pytest

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_freqdeck_add
from cle.players.contracts import PlayerContext
from cle.sandbox import CatanSandbox

from .support import _sandbox_and_player


@pytest.fixture
def trade_sandbox() -> CatanSandbox:
    sandbox, _, _ = _sandbox_and_player([])
    engine = sandbox.game_engine
    for _ in range(16):
        engine.step(engine.state.playable_actions[0])
    engine.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
    for color, bundle in (
        (Color.RED, (2, 0, 0, 0, 0)),
        (Color.BLUE, (0, 0, 0, 0, 2)),
    ):
        player_freqdeck_add(engine.state, color, bundle)
        for index, count in enumerate(bundle):
            engine.state.resource_freqdeck[index] -= count
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return sandbox


@pytest.fixture
def discard_sandbox() -> CatanSandbox:
    sandbox, _, _ = _sandbox_and_player([])
    engine = sandbox.game_engine
    for _ in range(16):
        engine.step(engine.state.playable_actions[0])
    player_freqdeck_add(engine.state, Color.RED, (4, 0, 0, 0, 4))
    engine.state.resource_freqdeck[0] -= 4
    engine.state.resource_freqdeck[4] -= 4
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    context = sandbox.decision_context()
    assert context.legal_actions[0].action_type == ActionType.DISCARD
    return sandbox


@pytest.fixture
def discard_context(discard_sandbox: CatanSandbox) -> PlayerContext:
    context = discard_sandbox.decision_context()
    # Supply the engine's exact required count independently of context assembly.
    return replace(context, discard_count=sum(context.observation.my_resources.values()) // 2)
