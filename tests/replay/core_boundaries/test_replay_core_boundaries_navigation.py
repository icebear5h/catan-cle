"""Replay navigation restores engine boundaries and invalidates contexts."""
import asyncio
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.models.player import Color
from cle.harness import ModelResponse
from cle.harness.models import ModelRequest
from cle.harness.prompt_store import resolve_prompt_suites
from cle.sandbox.decision import build_decision_context
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.replay.decision_preview import generate_decision_preview

from .support import _checkpoint_values


@pytest.mark.parametrize("navigation", ["goto_fast", "goto_sequential"])
def test_replay_navigation_restores_original_boundaries_and_invalidates_contexts(
    opening_replay: tuple[SimpleNamespace, ReplaySandbox], monkeypatch: pytest.MonkeyPatch, navigation: str,
) -> None:
    runtime, sandbox = opening_replay
    engine: Any = sandbox.game_engine
    global_rng = random.getstate()
    expected: Any = [_checkpoint_values(engine)]
    for _ in runtime.replay_data["parsed_actions"]:
        assert sandbox.step(allow_lookahead=False)["status"] == "ok"
        expected.append(_checkpoint_values(engine))

    def no_new_seed(*args: object) -> None:
        raise AssertionError("Replay navigation must not initialize a random engine")

    monkeypatch.setattr(random.SystemRandom, "randrange", no_new_seed)
    for target in (10, 0, 18, 3, 18):
        _, identity = sandbox.decision_context()
        revision = sandbox.revision
        result = getattr(sandbox, navigation)(target)
        assert result["event_index"] == target
        assert result["errors"] is None
        assert sandbox.game_engine is engine
        assert _checkpoint_values(engine) == expected[target]
        assert engine.rng is engine.state.rng
        assert sandbox.revision > revision
        assert sandbox.is_stale(identity)
        assert runtime.game_running == (target < 18)
        assert runtime.replay_final_state_synced == (target == 18)
        assert len(runtime.replay_step_checkpoints) == target
        assert runtime.game_log[0] == {"message": "initial log"}
    assert random.getstate() == global_rng


def test_backward_goto_without_initial_checkpoint_does_not_mutate(opening_replay: tuple[SimpleNamespace, ReplaySandbox]) -> None:
    runtime, sandbox = opening_replay
    sandbox.step(allow_lookahead=False)
    runtime.replay_step_checkpoints.clear()
    before = _checkpoint_values(sandbox.game_engine)
    revision = sandbox.revision
    assert sandbox.goto_sequential(0) == (
        {"error": "Initial replay checkpoint is unavailable"}, 409,
    )
    assert _checkpoint_values(sandbox.game_engine) == before
    assert sandbox.revision == revision
    assert sandbox.replay_index == 1


def test_source_completion_overrides_live_threshold_without_score_sync_events(opening_replay: tuple[SimpleNamespace, ReplaySandbox]) -> None:
    runtime, sandbox = opening_replay
    engine = sandbox.game_engine
    engine.vps_to_win = 1
    runtime.replay_data["parsed_actions"] = runtime.replay_data["parsed_actions"][:4]
    runtime.replay_data["end_game_state"] = {
        "players": {"1": {"victoryPoints": {"0": 1, "2": 1}, "winningPlayer": True}},
    }
    first: Any = sandbox.step(allow_lookahead=False)
    assert engine.winning_color() == Color.RED
    assert first["finished"] is False
    assert runtime.game_running is True
    assert runtime.replay_final_state_synced is False
    for _ in range(2):
        assert sandbox.step(allow_lookahead=False)["finished"] is False
    before_final = _checkpoint_values(engine)
    assert sandbox.step(allow_lookahead=False)["finished"] is True
    assert runtime.game_running is False
    assert runtime.replay_final_state_synced is True
    assert engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 2
    assert any(issue["kind"] == "forced_final_state_sync" for issue in runtime.replay_semantic_issues)
    assert [event.event_type for event in engine.events] == [
        "MESSAGE_SENT", "BUILD_SETTLEMENT", "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_ROAD",
    ]
    final = _checkpoint_values(engine)
    assert sandbox.undo()["status"] == "ok"
    assert _checkpoint_values(engine) == before_final
    assert runtime.replay_final_state_synced is False
    assert sandbox.step(allow_lookahead=False)["finished"] is True
    assert _checkpoint_values(engine) == final


def test_ongoing_replay_preview_bypasses_only_terminal_gating(opening_replay: tuple[SimpleNamespace, ReplaySandbox], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "playground.game_viewer.replay.decision_preview.resolve_prompt_suites",
        lambda: resolve_prompt_suites(
            directory=tmp_path, legacy=True, use_environment=False,
        ),
    )
    runtime, sandbox = opening_replay
    engine = sandbox.game_engine
    engine.vps_to_win = 1
    assert sandbox.step(allow_lookahead=False)["finished"] is False
    assert engine.winning_color() == Color.RED
    engine.append_message(
        speaker=Color.BLUE, text="private-to-blue-and-white", audience=(Color.WHITE,),
        causation_id="private-test",
        commitment=("private condition", "private promise", 2),
    )
    before_snapshot = engine.snapshot()
    before = pickle.dumps(before_snapshot)
    context, identity = sandbox.decision_context()
    expected_actions: Any = tuple(engine.state.playable_actions)
    assert context.legal_actions == expected_actions
    assert tuple(context.observation.valid_actions) == expected_actions
    assert all(event.payload["text"] != "private-to-blue-and-white" for event in context.recent_messages)
    assert all(commitment.proposer != Color.BLUE for commitment in context.active_commitments)
    view = sandbox.view(Color.RED)
    assert view.winner is None
    assert view.legal_actions == expected_actions
    assert tuple(view.observation.valid_actions) == expected_actions
    opponent_view: Any = sandbox.view(Color.WHITE)
    assert opponent_view.legal_actions == ()
    assert opponent_view.observation.valid_actions == []
    assert opponent_view.events[-1].payload["text"] == "private-to-blue-and-white"
    assert engine.events[-1].sequence not in {event.sequence for event in view.events}
    requests = []

    class PreviewTransport:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            requests.append(request)
            a, b = sorted(expected_actions[0].value)
            return ModelResponse(content=json.dumps({
                "game_plan": "continue",
                "tool": "build_road",
                "arguments": {"edge": f"<E{a:02d}_{b:02d}>"},
            }))

    preview: Any = asyncio.run(generate_decision_preview(
        sandbox, model="test/model", game_plan="", reasoning_request={"enabled": False},
        transport_factory=lambda **kwargs: PreviewTransport(),
    ))
    assert preview["parse_error"] is None
    assert preview["action"] == str(expected_actions[0])
    assert len(preview["available_actions"]) == len(expected_actions)
    assert preview["stale"] is False
    assert len(requests) == 1
    assert "private-to-blue-and-white" not in repr(requests)
    assert not sandbox.is_stale(identity)
    assert engine.observe(Color.RED).valid_actions == []
    with pytest.raises(ValueError, match="terminal"):
        build_decision_context(engine)
    with pytest.raises(ValueError, match="terminal"):
        engine.step(expected_actions[0])
    runtime.replay_index = len(runtime.replay_data["parsed_actions"])
    runtime.game_running = False
    assert sandbox.view(Color.RED).legal_actions == ()
    assert sandbox.view(Color.RED).observation.valid_actions == []
    with pytest.raises(ValueError, match="terminal"):
        sandbox.decision_context()
    after_snapshot = engine.snapshot()
    assert [
        key for key, value in vars(before_snapshot.state).items()
        if pickle.dumps(value) != pickle.dumps(vars(after_snapshot.state)[key])
    ] == []
    assert pickle.dumps(after_snapshot) == before
