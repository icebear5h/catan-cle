from pathlib import Path
from types import SimpleNamespace

from cle.harness import ContextAssembler, PlayerSession, load_context_suite
from cle.sandbox.decision import build_decision_context
from cle.sandbox.replay import ReplaySandbox
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    assert "VALID ACTIONS:" in request.messages[-1].content
    assert "<rationale>" not in request.messages[-1].content
    assert "<action>" in request.messages[-1].content
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
