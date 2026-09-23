"""Checkpoint reuse, backward goto, and branch isolation."""
import random
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.communication import CommitmentStatus
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.replay.runtime import step_executor
from cle.replay.runtime.checkpoint import ReplayStepCheckpoint
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.state import ServerState

from .support import COLORS, _check_checkpoint


def test_backward_goto_restores_original_identity_rng_and_deck(replay: tuple[ServerState, ReplaySandbox], monkeypatch: pytest.MonkeyPatch) -> None:
    _, sandbox = replay
    initial = sandbox.game_engine.snapshot()
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda *args: 99)
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    assert sandbox.goto_sequential(0)["event_index"] == 0
    restored = sandbox.game_engine
    assert sandbox.replay_index == 0
    assert restored.state.actions == []
    assert restored.rng is restored.state.rng
    observed = {
        "identity": restored.id == initial.engine_id,
        "seed": restored.seed == initial.seed,
        "rng": restored.state.rng.getstate() == initial.state.rng.getstate(),
        "deck": restored.state.development_listdeck == initial.state.development_listdeck,
    }
    _check_checkpoint(observed, {"identity": True, "seed": True, "rng": True, "deck": True})


def test_replay_checkpoint_is_reusable_and_restores_private_events_and_commitments(replay: tuple[ServerState, ReplaySandbox]) -> None:
    runtime, sandbox = replay
    engine: Any = sandbox.game_engine
    engine.append_message(
        speaker=Color.RED, text="private promise", audience=(Color.BLUE,),
        causation_id="initial-promise",
        commitment=("offer wood", "return brick", 1),
    )
    checkpoint: Any = ReplayStepCheckpoint.capture(runtime)
    initial: Any = engine.snapshot()
    projected: Any = {color: engine.project_events(color) for color in COLORS}
    for _ in range(2):
        assert sandbox.step(allow_lookahead=False)["status"] == "ok"
        engine.rng.random()
        engine.commitments[0].status = CommitmentStatus.EXPIRED
        engine.append_message(
            speaker=Color.BLUE, text="future promise", audience=(Color.RED,),
            causation_id="future-promise",
            commitment=("offer brick", "return wood", 3),
        )
        checkpoint.restore(runtime)
        runtime.replay_step_checkpoints.clear()
        assert engine.state is not checkpoint.game_state
        assert engine.state.rng is engine.rng
        assert engine.rng.getstate() == initial.state.rng.getstate()
        assert engine.state.actions == initial.state.actions
        assert engine.state.player_state == initial.state.player_state
        assert engine.state.development_listdeck == initial.state.development_listdeck
        assert tuple(engine.events) == initial.events == checkpoint.game_events
        assert tuple(engine.commitments) == initial.commitments == checkpoint.game_commitments
        assert engine.commitments[0] is not checkpoint.game_commitments[0]
        assert engine.history == []
        assert {color: engine.project_events(color) for color in COLORS} == projected
        assert runtime.replay_index == 0
        assert runtime.replay_actions_per_step == []
        assert runtime.game_running is True
        assert runtime.replay_final_state_synced is False


def test_replay_checkpoint_undoes_every_action_in_a_compound_step(replay: tuple[ServerState, ReplaySandbox], monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, sandbox = replay
    engine = sandbox.game_engine

    def compound_step(
        state: ServerState, broadcast_fn: object, allow_lookahead: bool
    ) -> dict[str, str]:
        for _ in range(2):
            engine.step(engine.state.playable_actions[0])
        state.replay_actions_per_step.append(1)
        state.replay_index += 1
        return {"status": "ok"}

    monkeypatch.setattr(step_executor, "_replay_step_logic", compound_step)
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    assert runtime.replay_actions_per_step == [2]
    expected_actions = deepcopy(engine.state.actions)
    expected_events = deepcopy(engine.events)
    assert len(expected_actions) == len(expected_events) == len(engine.history) == 2
    undone: Any = sandbox.undo()
    assert undone["actions_undone"] == len(undone["undone_actions"]) == 2
    assert engine.state.actions == engine.events == engine.history == []
    assert sandbox.step(allow_lookahead=False)["status"] == "ok"
    assert engine.state.actions == expected_actions
    assert engine.events == expected_events


def test_branch_event_mutation_cannot_contaminate_engine_or_snapshot() -> None:
    engine: Any = GameEngine(COLORS, seed=4, shuffle_players=False)
    engine.append_message(
        speaker=Color.RED, text="original", audience=COLORS,
        causation_id="audit",
    )
    snapshot: Any = engine.snapshot()
    branch: Any = engine.copy()
    assert branch.project_events(Color.RED)[0].payload["text"] == "original"
    try:
        branch.project_events(Color.RED)[0].payload["text"] = "branch-only change"
    except TypeError:
        pass  # An immutable projection is also a valid isolation boundary.
    original_text = engine.project_events(Color.RED)[0].payload["text"]
    engine.restore(snapshot)
    observed = (
        original_text,
        snapshot.events[0].public_payload["text"],
        engine.project_events(Color.RED)[0].payload["text"],
    )
    _check_checkpoint(observed, ("original", "original", "original"))
