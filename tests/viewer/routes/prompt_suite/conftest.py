"""Flask app and server state for the prompt-suite route tests."""

from pathlib import Path

import pytest
from flask import Flask

from cle.game_engine.models.player import Color
from cle.harness.prompt_store import resolve_prompt_suites
from playground.game_viewer.routes.prompt_suite import prompt_suite_bp
from playground.game_viewer.state import ServerState

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@pytest.fixture
def prompt_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Flask, ServerState]:
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path / "prompt-suites"))
    # Exercise the historical pair editor without changing the shared default.
    monkeypatch.setattr(
        "playground.game_viewer.routes.prompt_suite.resolve_prompt_suites",
        lambda **kwargs: resolve_prompt_suites(legacy=True, use_environment=False),
    )
    app = Flask(__name__)
    state = ServerState()
    app.config["SERVER_STATE"] = state
    app.register_blueprint(prompt_suite_bp)
    return app, state
