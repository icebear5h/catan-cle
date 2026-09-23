"""Shared fixtures for shared prompt component authoring, compilation, and rendering."""

from pathlib import Path

import pytest

from cle.game_engine.events import PlayerEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.observation import observe_state
from cle.players.contracts import PlayerContext, TalkContext

from .support import COLORS


@pytest.fixture(autouse=True)
def isolated_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def contexts() -> tuple[PlayerContext, TalkContext]:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    observation = observe_state(engine.state, Color.RED)
    event = PlayerEvent(23, "action:23", Color.BLUE, "END_TURN", None)
    talk = PlayerEvent(24, "talk:24", Color.BLUE, "MESSAGE", "WOOD for ORE?")
    decision = PlayerContext(
        context_id="decision:RED", actor=Color.RED, turn_number=observation.current_turn,
        phase=observation.current_phase, observation=observation, events=(event,),
        legal_actions=tuple(observation.valid_actions), prompt_key="initial_settlement_1",
        recent_messages=(talk,),
    )
    speech = TalkContext(
        context_id="speech:RED", player=Color.RED, participants=COLORS, cause=event,
        visible_through_sequence=24, game_events=(event,), recent_messages=(talk,),
        observation=observation,
    )
    return decision, speech
