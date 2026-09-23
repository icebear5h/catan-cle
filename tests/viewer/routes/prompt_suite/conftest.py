"""Flask app and server state for the pinned legacy prompt-suite route tests."""

from pathlib import Path

import pytest
from flask import Flask

from cle.game_engine.models.player import Color
from cle.harness.communication import default_communication_suite_path
from cle.harness.prompt_store import COMMUNICATION_SUITE_ENV, CONTEXT_SUITE_ENV, SHARED_SUITE_ENV
from cle.harness.suite import default_suite_path
from playground.game_viewer.routes.prompt_suite import prompt_suite_bp
from playground.game_viewer.state import ServerState

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@pytest.fixture
def prompt_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Flask, ServerState]:
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path / "prompt-suites"))
    # Pin the built-in legacy pair the way a runtime would; the Studio shows it read-only.
    monkeypatch.delenv(SHARED_SUITE_ENV, raising=False)
    monkeypatch.setenv(CONTEXT_SUITE_ENV, str(default_suite_path()))
    monkeypatch.setenv(COMMUNICATION_SUITE_ENV, str(default_communication_suite_path()))
    app = Flask(__name__)
    state = ServerState()
    app.config["SERVER_STATE"] = state
    app.register_blueprint(prompt_suite_bp)
    return app, state
