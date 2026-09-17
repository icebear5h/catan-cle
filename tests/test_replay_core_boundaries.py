import asyncio
from copy import deepcopy
from pathlib import Path
import json
import pickle
import random
from types import SimpleNamespace

import pytest

from cle.harness import ContextAssembler, ModelResponse, PlayerSession, load_context_suite
from cle.harness.prompt_store import resolve_prompt_suites
from cle.sandbox.decision import build_decision_context
from cle.sandbox.replay import ReplaySandbox
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from playground.game_viewer.replay.decision_preview import generate_decision_preview


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def opening_replay():
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
    game = GameEngine(
        colors, seed=3, shuffle_players=False, capture_history=True,
        discard_limit=5, vps_to_win=12,
    )
    game.append_message(
        speaker=Color.RED, text="initial promise", audience=(Color.BLUE,),
        causation_id="opening",
        commitment=("offer wood", "return brick", 1),
    )
    runtime = SimpleNamespace(
        replay_revision=0, replay_index=0, replay_mode=True, game_running=True,
        replay_actions_per_step=[], replay_step_checkpoints=[],
        replay_trade_ledger={}, replay_semantic_issues=[], first_divergence_step={},
        replay_final_state_synced=False, replay_pending_dev_card=None,
        game_log=[{"message": "initial log"}],
        corner_to_node_map={}, edge_to_edge_map={},
    )
    sandbox = ReplaySandbox(runtime, game)
    runtime.current_sandbox = sandbox
    trajectory = game.copy()
    actions = []
    for index in range(18):
        if index < 16:
            action = trajectory.state.playable_actions[0]
        elif index == 16:
            action = Action(Color.RED, ActionType.ROLL, (1, 1))
        else:
            action = Action(Color.RED, ActionType.END_TURN, None)
        hint = {
            "index": index, "type": action.action_type.value,
            "player": colors.index(action.color) + 1,
        }
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            hint["colonist_corner"] = str(index)
            runtime.corner_to_node_map[f"_{index}"] = action.value
        elif action.action_type == ActionType.BUILD_ROAD:
            hint["colonist_edge"] = str(index)
            runtime.edge_to_edge_map[f"_{index}"] = tuple(sorted(action.value))
        elif action.action_type == ActionType.ROLL:
            hint["dice"] = action.value
        trajectory.step(action, force=True)
        actions.append(hint)
    runtime.replay_data = {
        "game_id": "opening", "parsed_actions": actions,
        "events": [{} for _ in actions], "end_game_state": {},
        "colonist_color_to_engine_idx": {str(index + 1): index for index in range(4)},
    }
    return runtime, sandbox


def _checkpoint_values(game):
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


@pytest.mark.parametrize("navigation", ["goto_fast", "goto_sequential"])
def test_replay_navigation_restores_original_boundaries_and_invalidates_contexts(
    opening_replay, monkeypatch, navigation,
):
    runtime, sandbox = opening_replay
    engine = sandbox.game_engine
    global_rng = random.getstate()
    expected = [_checkpoint_values(engine)]
    for _ in runtime.replay_data["parsed_actions"]:
        assert sandbox.step(allow_lookahead=False)["status"] == "ok"
        expected.append(_checkpoint_values(engine))

    def no_new_seed(*args):
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


def test_backward_goto_without_initial_checkpoint_does_not_mutate(opening_replay):
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


def test_source_completion_overrides_live_threshold_without_score_sync_events(opening_replay):
    runtime, sandbox = opening_replay
    engine = sandbox.game_engine
    engine.vps_to_win = 1
    runtime.replay_data["parsed_actions"] = runtime.replay_data["parsed_actions"][:4]
    runtime.replay_data["end_game_state"] = {
        "players": {"1": {"victoryPoints": {"0": 1, "2": 1}, "winningPlayer": True}},
    }
    first = sandbox.step(allow_lookahead=False)
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


def test_ongoing_replay_preview_bypasses_only_terminal_gating(opening_replay, tmp_path, monkeypatch):
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
    expected_actions = tuple(engine.state.playable_actions)
    assert context.legal_actions == expected_actions
    assert tuple(context.observation.valid_actions) == expected_actions
    assert all(event.payload["text"] != "private-to-blue-and-white" for event in context.recent_messages)
    assert all(commitment.proposer != Color.BLUE for commitment in context.active_commitments)
    view = sandbox.view(Color.RED)
    assert view.winner is None
    assert view.legal_actions == expected_actions
    assert tuple(view.observation.valid_actions) == expected_actions
    opponent_view = sandbox.view(Color.WHITE)
    assert opponent_view.legal_actions == ()
    assert opponent_view.observation.valid_actions == []
    assert opponent_view.events[-1].payload["text"] == "private-to-blue-and-white"
    assert engine.events[-1].sequence not in {event.sequence for event in view.events}
    requests = []

    class PreviewTransport:
        async def complete(self, request):
            requests.append(request)
            a, b = sorted(expected_actions[0].value)
            return ModelResponse(content=json.dumps({
                "game_plan": "continue",
                "tool": "build_road",
                "arguments": {"edge": f"<E{a:02d}_{b:02d}>"},
            }))

    preview = asyncio.run(generate_decision_preview(
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


def test_replay_context_uses_the_same_general_suite_as_live_games():
    game = GameEngine(
        [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE],
        seed=3,
        shuffle_players=False,
    )
    context = build_decision_context(game)
    suite = load_context_suite()
    session = PlayerSession(
        color=context.actor,
        session_id="replay-shared-context",
        strategic_memory="Build a city.",
    )

    request = ContextAssembler(suite).assemble(context, session)

    assert request.messages[0].content == (
        "You are playing a game of Catan. You are playing as RED."
    )
    assert "YOUR CURRENT GAME PLAN:\nBuild a city." in request.messages[-1].content
    assert suite.version == "11.0.0"
    assert suite.response.format == "json"
    assert "build_settlement" in request.messages[-1].content
    assert "<rationale>" not in request.messages[-1].content
    assert "<action>" not in request.messages[-1].content
    assert "action_index" not in request.messages[-1].content
    assert '"tool"' in request.messages[-1].content
    assert '"arguments"' in request.messages[-1].content
    assert request.components[0].id == "system.identity"
    assert request.components[-1].id == "environment.response_schema"


def test_replay_view_reports_replay_revision_not_engine_action_count():
    game = GameEngine(
        [Color.RED, Color.BLUE],
        seed=3,
        shuffle_players=False,
    )
    runtime = SimpleNamespace(
        current_game=game,
        replay_revision=17,
        replay_index=0,
    )

    assert ReplaySandbox(runtime).view(Color.RED).revision == 17


def test_replay_stale_check_uses_owned_engine_not_removed_state_alias():
    game = GameEngine(
        [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE],
        seed=3,
        shuffle_players=False,
    )
    runtime = SimpleNamespace(
        replay_data={"game_id": "test-game"},
        replay_revision=0,
        replay_index=0,
    )
    sandbox = ReplaySandbox(runtime, game)

    _, identity = sandbox.decision_context()

    assert not sandbox.is_stale(identity)
    sandbox.replace_game_engine(game.copy())
    assert sandbox.is_stale(identity)


def test_legacy_viewer_core_shims_are_removed():
    legacy_paths = (
        "playground/game_viewer/colonist",
        "playground/game_viewer/replay/action_matcher.py",
        "playground/game_viewer/replay/audit.py",
        "playground/game_viewer/replay/checkpoint.py",
        "playground/game_viewer/replay/llm_response.py",
        "playground/game_viewer/replay/navigation.py",
        "playground/game_viewer/replay/step_executor.py",
        "playground/game_viewer/replay/trade_ledger.py",
        "cle/harness/replay.py",
        "cle/harness/replay_suite.py",
        "cle/harness/suites/replay_v2.yaml",
    )

    assert [path for path in legacy_paths if (PROJECT_ROOT / path).exists()] == []


def test_core_runtime_has_no_viewer_or_web_framework_imports():
    forbidden = (
        "playground.game_viewer",
        "from flask",
        "import flask",
        "socketio",
        "playwright",
    )
    paths = [
        *sorted((PROJECT_ROOT / "cle" / "harness").rglob("*.py")),
        *sorted((PROJECT_ROOT / "cle" / "sandbox").rglob("*.py")),
        *sorted((PROJECT_ROOT / "cle" / "replay").rglob("*.py")),
    ]

    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(PROJECT_ROOT)}: {token}")

    assert violations == []
