"""Environment and runtime prompt sources stay visible and unshadowable."""
from pathlib import Path
from typing import Any

import pytest

from cle.harness import prompt_store
from cle.harness.prompt_store import load_active_prompt_suites, save_prompt_suite_overrides
from cle.harness.shared_suite import default_shared_suite_path
from cle.sandbox.factory import LiveSandboxConfig

from .conftest import Studio


@pytest.mark.parametrize("field,value", [("replay_mode", True), ("replay_data", {})])
def test_loaded_game_save_and_reset_lock(
    studio: Studio, field: str, value: bool | dict[str, Any]
) -> None:
    client, state = studio
    original: Any = client.get("/api/prompt-suite").json
    setattr(state, field, value)
    expected: Any = {"shared": original["shared"]["sha256"]}
    for response in (
        client.put("/api/prompt-suite", json={"expected": expected, "shared": original["shared"]["document"]}),
        client.delete("/api/prompt-suite", json={"expected": expected}),
    ):
        assert response.status_code == 409
        assert response.headers["Cache-Control"] == "no-store"


def test_existing_legacy_pair_remains_editable(studio: Studio) -> None:
    client, _ = studio
    active: Any = load_active_prompt_suites()
    save_prompt_suite_overrides(
        decision_source=active.decision.source, communication_source=active.communication.source,
        expected_decision_sha256=active.decision.sha256,
        expected_communication_sha256=active.communication.sha256,
    )
    original: Any = client.get("/api/prompt-suite").json
    assert original["mode"] == "legacy"
    edits: Any = {
        "decision": {key: original["decision"][key] for key in (
            "system_identity", "components", "phase_guidance", "response_instruction",
        )},
        "communication": {key: original["communication"][key] for key in ("system_identity", "components")},
    }
    edits["decision"]["components"]["board_state"] = "LEGACY EDIT:\n{{ value }}"
    saved: Any = client.put("/api/prompt-suite", json={
        **edits, "expected": {key: original[key]["sha256"] for key in ("decision", "communication")},
    })
    assert saved.status_code == 200
    assert saved.json["mode"] == "legacy"
    assert saved.json["decision"]["components"]["board_state"].startswith("LEGACY EDIT:")
    reset: Any = client.delete("/api/prompt-suite", json={
        "expected": {key: saved.json[key]["sha256"] for key in ("decision", "communication")},
    })
    assert reset.status_code == 200
    assert reset.json["mode"] == "shared"
    assert reset.json["shared"] == client.get("/api/prompt-suite").json["shared"]


def test_environment_source_is_visible_but_cannot_be_shadowed_by_editor(studio: Studio, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = studio
    monkeypatch.setenv("CATAN_SHARED_SUITE", str(default_shared_suite_path()))
    original: Any = client.get("/api/prompt-suite").json
    payload: Any = {"expected": {"shared": original["shared"]["sha256"]}}
    saved: Any = client.put("/api/prompt-suite", json={**payload, "shared": original["shared"]["document"]})
    reset = client.delete("/api/prompt-suite", json=payload)
    assert saved.status_code == reset.status_code == 409
    assert "environment" in saved.json["error"]


def test_runtime_path_selection_is_visible_and_cannot_be_shadowed(studio: Studio, tmp_path: Path) -> None:
    client, state = studio
    path = tmp_path / "explicit-shared.yaml"
    path.write_text(default_shared_suite_path().read_text().replace(
        "{{ board_state }}", "EXPLICIT ACTIVE BOARD {{ board_state }}",
    ))
    state.active_live_config = LiveSandboxConfig(shared_suite_path=str(path))
    original: Any = client.get("/api/prompt-suite").json
    assert "EXPLICIT ACTIVE BOARD" in str(original["shared"]["document"])
    saved: Any = client.put("/api/prompt-suite", json={
        "shared": original["shared"]["document"],
        "expected": {"shared": original["shared"]["sha256"]},
    })
    assert saved.status_code == 409
    assert "Explicit runtime prompt paths" in saved.json["error"]


def test_shared_payload_rejects_legacy_or_unknown_root_keys(studio: Studio) -> None:
    client, _ = studio
    original: Any = client.get("/api/prompt-suite").json
    validated = client.post("/api/prompt-suite/validate", json={
        "shared": original["shared"]["document"], "decision": {},
    })
    assert validated.status_code == 400
    reset = client.delete("/api/prompt-suite", json={
        "expected": {"shared": original["shared"]["sha256"], "decision": "not-allowed"},
    })
    assert reset.status_code == 400


@pytest.mark.parametrize("mode", ["shared", "legacy"])
@pytest.mark.parametrize("failure", ["missing", "invalid"])
def test_reset_endpoint_preserves_override_when_shared_default_cannot_load(studio: Studio, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, failure: str) -> None:
    client, _ = studio
    if mode == "legacy":
        pair: Any = load_active_prompt_suites()
        save_prompt_suite_overrides(
            decision_source=pair.decision.source, communication_source=pair.communication.source,
            expected_decision_sha256=pair.decision.sha256,
            expected_communication_sha256=pair.communication.sha256,
        )
    else:
        initial: Any = client.get("/api/prompt-suite").json
        saved = client.put("/api/prompt-suite", json={
            "shared": initial["shared"]["document"], "expected": {"shared": initial["shared"]["sha256"]},
        })
        assert saved.status_code == 200
    original: Any = client.get("/api/prompt-suite").json
    kinds: Any = ("shared",) if mode == "shared" else ("decision", "communication")
    files = [tmp_path / "prompts" / f"{kind}.yaml" for kind in kinds]
    before = [path.read_bytes() for path in files]
    replacement = tmp_path / "broken-default.yaml"
    if failure == "invalid":
        replacement.write_text("invalid: [", encoding="utf-8")
    monkeypatch.setattr(prompt_store, "default_shared_suite_path", lambda: replacement)
    response = client.delete("/api/prompt-suite", json={
        "expected": {kind: original[kind]["sha256"] for kind in kinds},
    })
    assert response.status_code == 400
    assert response.headers["Cache-Control"] == "no-store"
    assert [path.read_bytes() for path in files] == before
    assert client.get("/api/prompt-suite").json == original
