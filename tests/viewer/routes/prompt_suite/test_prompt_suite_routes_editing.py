"""A pinned legacy pair is served read-only and every write is refused."""
from pathlib import Path
from typing import Any

from flask import Flask

from cle.harness.communication import default_communication_suite_path
from cle.harness.prompt_store import resolve_prompt_suites, source_sha256
from cle.harness.suite import default_suite_path
from playground.game_viewer.state import ServerState

from .support import _sandbox


def _sha(path: Path) -> str:
    return source_sha256(path.read_text(encoding="utf-8"))


def test_prompt_suite_get_returns_fixed_strings_and_no_store(prompt_app: tuple[Flask, ServerState]) -> None:
    app, _ = prompt_app

    response = app.test_client().get("/api/prompt-suite")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    payload = response.get_json()
    assert set(payload) == {
        "mode", "read_only", "decision", "communication", "saving_locked", "preview",
    }
    assert payload["mode"] == "legacy"
    assert payload["read_only"] is True
    assert payload["saving_locked"] is False
    assert payload["decision"] == {
        "id": "catan-agent", "version": "11.0.0", "status": "legacy",
        "sha256": _sha(default_suite_path()), "overridden": False,
    }
    assert payload["communication"]["sha256"] == _sha(default_communication_suite_path())
    assert payload["communication"]["overridden"] is False
    assert payload["preview"]["decision"]["status"] == "no_game_context"
    communication = payload["preview"]["communication"]
    assert communication["status"] == "no_communication_opportunity_rendered_yet"
    assert [item["id"] for item in communication["components"][:2]] == [
        "system.identity", "environment.communication_policy",
    ]
    assert all(not item["rendered"] for item in communication["components"])


def test_live_editing_is_allowed_but_replay_remains_read_only(prompt_app: tuple[Flask, ServerState]) -> None:
    app, state = prompt_app
    client: Any = app.test_client()
    original = client.get("/api/prompt-suite").get_json()
    shared: Any = resolve_prompt_suites(use_environment=False).shared
    document = shared.bundle.model_dump(mode="json")
    expected = {"shared": shared.sha256}
    pair_edit = {"decision": {}, "communication": {}}

    state.current_sandbox = _sandbox()
    assert client.get("/api/prompt-suite").json["saving_locked"] is False
    assert client.post("/api/prompt-suite/validate", json=pair_edit).status_code == 400
    for response in (
        client.put("/api/prompt-suite", json={"shared": document, "expected": expected}),
        client.delete("/api/prompt-suite", json={"expected": expected}),
    ):
        assert response.status_code == 409
        assert "Explicit environment prompt paths" in response.json["error"]

    state.current_sandbox = None
    state.replay_data = {"events": []}
    saved = client.put("/api/prompt-suite", json={"shared": document, "expected": expected})
    reset = client.delete("/api/prompt-suite", json={"expected": expected})
    assert saved.status_code == reset.status_code == 409
    assert "Clear the loaded replay" in saved.json["error"]
    current = client.get("/api/prompt-suite").get_json()
    assert current["saving_locked"] is True
    assert {kind: current[kind] for kind in ("decision", "communication")} == {
        kind: original[kind] for kind in ("decision", "communication")
    }
