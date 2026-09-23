"""A loaded curated replay state for the causal commentary tests."""

import contextlib
import io
from collections.abc import Iterator

import pytest

from playground.game_viewer.app import app
from playground.game_viewer.state import ServerState, server_state


@pytest.fixture
def paired_state() -> Iterator[ServerState]:
    app.config["TESTING"] = True
    client = app.test_client()
    server_state.reset()
    with contextlib.redirect_stdout(io.StringIO()):
        response = client.post("/api/load-replay", json={"game_id": "242781000"})
    assert response.status_code == 200
    try:
        yield server_state
    finally:
        server_state.reset()
