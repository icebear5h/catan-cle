"""Prompt writes use the store lock independently of the inference lock."""
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from cle.harness import prompt_store
from cle.harness.prompt_store import load_active_prompt_suites, save_prompt_suite_overrides
from playground.game_viewer.routes import prompt_suite as prompt_routes

from .conftest import Studio


@pytest.mark.parametrize("guidance", [{}, {"main_game": "Only one phase"}])
def test_incomplete_guidance_is_rejected_without_game_context(
    studio: Studio, guidance: dict[str, str]
) -> None:
    client, _ = studio
    original: Any = client.get("/api/prompt-suite").json
    document: Any = deepcopy(original["shared"]["document"])
    document["phase_guidance"] = guidance
    validated: Any = client.post("/api/prompt-suite/validate", json={"shared": document})
    assert validated.status_code == 400
    assert "phase_guidance" in validated.json["errors"][0]["message"]
    saved = client.put("/api/prompt-suite", json={
        "shared": document, "expected": {"shared": original["shared"]["sha256"]},
    })
    assert saved.status_code == 400
    assert client.get("/api/prompt-suite").json == original


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
@pytest.mark.parametrize("mode", ["shared", "legacy"])
def test_write_uses_store_lock_independently_of_inference_lock(studio: Studio, monkeypatch: pytest.MonkeyPatch, method: str, mode: str) -> None:
    client, state = studio
    if mode == "legacy":
        pair: Any = load_active_prompt_suites()
        save_prompt_suite_overrides(
            decision_source=pair.decision.source, communication_source=pair.communication.source,
            expected_decision_sha256=pair.decision.sha256,
            expected_communication_sha256=pair.communication.sha256,
        )
    original: Any = client.get("/api/prompt-suite").json
    kinds = ("shared",) if mode == "shared" else ("decision", "communication")
    payload: Any = {"expected": {kind: original[kind]["sha256"] for kind in kinds}}
    if method == "PUT":
        if mode == "shared":
            payload["shared"] = original["shared"]["document"]
        else:
            payload["decision"] = {key: original["decision"][key] for key in (
                "system_identity", "components", "phase_guidance", "response_instruction",
            )}
            payload["communication"] = {key: original["communication"][key] for key in (
                "system_identity", "components",
            )}
    original_store_lock = prompt_store._store_lock
    original_check = prompt_routes._saving_locked
    operations = []

    def checked_game_state(value: SimpleNamespace) -> bool:
        operations.append("check")
        return original_check(value)

    @contextmanager
    def checked_store_lock(directory: Path) -> Iterator[None]:
        assert not state.replay_mutation_lock._is_owned()
        operations.append("store")
        with original_store_lock(directory):
            yield
            assert not state.replay_mutation_lock._is_owned()

    monkeypatch.setattr(prompt_routes, "_saving_locked", checked_game_state)
    monkeypatch.setattr(prompt_store, "_store_lock", checked_store_lock)
    result = client.open("/api/prompt-suite", method=method, json=payload)
    assert result.status_code == 200
    assert operations[0] == "check" and "store" in operations


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
def test_source_published_before_preview_waits_for_inference(studio: Studio, method: str) -> None:
    client, state = studio
    original: Any = client.get("/api/prompt-suite").json
    payload: Any = {"expected": {"shared": original["shared"]["sha256"]}}
    if method == "PUT":
        payload["shared"] = original["shared"]["document"]
        payload["shared"]["components"]["board_state"]["template"] = "LIVE EDIT\n{{ board_state }}"
    attempted = Event()
    lock = state.replay_mutation_lock

    class ObservedLock:
        def __enter__(self) -> None:
            attempted.set()
            lock.acquire()

        def __exit__(self, *args: object) -> None:
            lock.release()

    state.replay_mutation_lock = ObservedLock()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with lock:
            pending = pool.submit(client.open, "/api/prompt-suite", method=method, json=payload)
            assert attempted.wait(3), "Preview did not reach the shared state lock"
            assert not pending.done()
            active: Any = prompt_store.resolve_prompt_suites()
            if method == "PUT":
                assert "LIVE EDIT" in active.shared.source
        response: Any = pending.result(timeout=3)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert client.get("/api/prompt-suite").json["shared"] == response.json["shared"]


def test_empty_source_environment_does_not_block_studio(studio: Studio, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = studio
    for name in ("CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE"):
        monkeypatch.setenv(name, "")
    original: Any = client.get("/api/prompt-suite")
    assert original.status_code == 200 and original.json["mode"] == "shared"
    saved: Any = client.put("/api/prompt-suite", json={
        "shared": original.json["shared"]["document"],
        "expected": {"shared": original.json["shared"]["sha256"]},
    })
    assert saved.status_code == 200
    reset: Any = client.delete("/api/prompt-suite", json={"expected": {"shared": saved.json["shared"]["sha256"]}})
    assert reset.status_code == 200
    assert reset.json["shared"] == original.json["shared"]
