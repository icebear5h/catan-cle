"""Prompt-suite reads, validation, saves, and replay read-only gating."""
from typing import Any

from flask import Flask

from cle.harness.prompt_store import resolve_prompt_suites
from playground.game_viewer.state import ServerState

from .support import _edits, _sandbox, _save_payload


def test_prompt_suite_get_returns_fixed_strings_and_no_store(prompt_app: tuple[Flask, ServerState]) -> None:
    app, _ = prompt_app

    response = app.test_client().get("/api/prompt-suite")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    payload = response.get_json()
    assert payload["mode"] == "legacy"
    assert payload["saving_locked"] is False
    assert payload["decision"]["version"] == "11.0.0"
    instruction = payload["decision"]["response_instruction"]
    assert '"game_plan"' in instruction
    assert '"tool"' in instruction
    assert '"arguments"' in instruction
    assert "<action>" not in instruction
    assert "action_index" not in instruction
    assert "Maximizing raw pip count is not the objective" in (
        payload["decision"]["phase_guidance"]["initial_settlement_1"]
    )
    assert payload["decision"]["component_order"] == [
        "environment.strategic_memory",
        "environment.visible_events",
        "environment.recent_table_talk",
        "environment.commitments",
        "environment.phase_info",
        "environment.board_state",
        "environment.resources",
        "environment.opponents",
        "environment.trade_window",
        "environment.phase_guidance",
        "environment.legal_actions",
        "environment.decision_request",
        "environment.response_schema",
    ]
    assert "{{ value }}" in payload["decision"]["components"]["board_state"]
    assert "initial_road" not in payload["decision"]["phase_guidance"]
    assert {
        "initial_road_1",
        "initial_road_2",
    } <= set(payload["decision"]["phase_guidance"])
    assert payload["communication"]["component_order"][0] == (
        "environment.communication_policy"
    )
    assert payload["variables"] == {
        "system.identity": ["color"],
        "environment.*": ["value"],
    }
    assert payload["preview"]["decision"]["status"] == "no_game_context"
    assert payload["preview"]["communication"]["status"] == (
        "no_communication_opportunity_rendered_yet"
    )


def test_validate_save_stale_and_reset_prompt_strings(prompt_app: tuple[Flask, ServerState]) -> None:
    app, _ = prompt_app
    client: Any = app.test_client()
    original: Any = client.get("/api/prompt-suite").get_json()
    edits: Any = _edits(original)
    edits["decision"]["components"]["board_state"] = (
        "CUSTOM BOARD COMPONENT:\n{{ value }}"
    )
    edits["decision"]["phase_guidance"]["initial_road_1"] = (
        "The second settlement remains independent. This road only seeds later expansion."
    )

    validated: Any = client.post("/api/prompt-suite/validate", json=edits)
    unchanged = client.get("/api/prompt-suite").get_json()
    saved: Any = client.put(
        "/api/prompt-suite",
        json=_save_payload(original, edits),
    )

    assert validated.status_code == 200
    assert validated.json["valid"] is True
    assert validated.json["candidate"]["decision"]["components"]["board_state"] == (
        "CUSTOM BOARD COMPONENT:\n{{ value }}"
    )
    assert unchanged["decision"]["sha256"] == original["decision"]["sha256"]
    assert saved.status_code == 200
    assert saved.json["status"] == "saved"
    assert saved.json["decision"]["overridden"] is True
    assert saved.json["decision"]["sha256"] != original["decision"]["sha256"]

    stale = client.put(
        "/api/prompt-suite",
        json=_save_payload(original, edits),
    )
    after_stale: Any = client.get("/api/prompt-suite").get_json()

    assert stale.status_code == 409
    assert after_stale["decision"]["sha256"] == saved.json["decision"]["sha256"]
    assert after_stale["communication"]["sha256"] == (
        saved.json["communication"]["sha256"]
    )

    reset: Any = client.delete(
        "/api/prompt-suite",
        json={
            "expected": {
                "decision": saved.json["decision"]["sha256"],
                "communication": saved.json["communication"]["sha256"],
            }
        },
    )

    assert reset.status_code == 200
    assert reset.json["status"] == "reset"
    assert reset.json["mode"] == "shared"
    assert reset.json["shared"]["overridden"] is False
    assert reset.json["shared"]["sha256"] == resolve_prompt_suites(use_environment=False).shared.sha256


def test_invalid_component_edit_leaves_both_suites_unchanged(prompt_app: tuple[Flask, ServerState]) -> None:
    app, _ = prompt_app
    client = app.test_client()
    original = client.get("/api/prompt-suite").get_json()
    edits: Any = _edits(original)
    edits["decision"]["components"]["resources"] = "{{ another_player_hand }}"

    validation: Any = client.post("/api/prompt-suite/validate", json=edits)
    saved = client.put(
        "/api/prompt-suite",
        json=_save_payload(original, edits),
    )
    current = client.get("/api/prompt-suite").get_json()

    assert validation.status_code == 400
    assert validation.json["valid"] is False
    assert "unknown variables" in validation.json["errors"][0]["message"]
    assert saved.status_code == 400
    assert current["decision"]["sha256"] == original["decision"]["sha256"]
    assert current["communication"]["sha256"] == original["communication"]["sha256"]


def test_live_editing_is_allowed_but_replay_remains_read_only(prompt_app: tuple[Flask, ServerState]) -> None:
    app, state = prompt_app
    client: Any = app.test_client()
    original = client.get("/api/prompt-suite").get_json()
    edits = _edits(original)
    save_payload = _save_payload(original, edits)
    reset_payload = {
        "expected": {
            "decision": original["decision"]["sha256"],
            "communication": original["communication"]["sha256"],
        }
    }

    state.current_sandbox = _sandbox()
    assert client.get("/api/prompt-suite").json["saving_locked"] is False
    saved: Any = client.put("/api/prompt-suite", json=save_payload)
    assert saved.status_code == 200
    reset_payload["expected"] = {
        kind: saved.json[kind]["sha256"] for kind in ("decision", "communication")
    }
    assert client.delete("/api/prompt-suite", json=reset_payload).status_code == 200

    state.current_sandbox = None
    state.replay_data = {"events": []}
    assert client.put("/api/prompt-suite", json=save_payload).status_code == 409
    assert client.delete("/api/prompt-suite", json=reset_payload).status_code == 409
