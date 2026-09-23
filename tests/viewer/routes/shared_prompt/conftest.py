"""Prompt studio client, server state, and inference-free transport."""
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest
from flask import Flask
from flask.testing import FlaskClient

from cle.game_engine.models.player import Color
from cle.harness.models import ModelRequest
from playground.game_viewer.routes.prompt_suite import prompt_suite_bp

Studio = tuple[FlaskClient, SimpleNamespace]


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class NeverTransport:
    async def complete(self, request: ModelRequest) -> None:
        raise AssertionError("No inference is permitted in prompt tests")


@pytest.fixture
def studio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FlaskClient, SimpleNamespace]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("CATAN_LIVE_TRACE_DB", str(tmp_path / "unused.sqlite3"))
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path / "prompts"))
    for name in ("CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE"):
        monkeypatch.delenv(name, raising=False)
    app = Flask(__name__)
    state = SimpleNamespace(
        current_sandbox=None, replay_mode=False, replay_data=None, replay_mutation_lock=RLock(),
    )
    app.config["SERVER_STATE"] = state
    app.register_blueprint(prompt_suite_bp)
    return app.test_client(), state
