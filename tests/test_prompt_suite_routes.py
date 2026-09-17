import pytest
from flask import Flask

from cle.harness import ModelMessage, ModelRequest, PromptComponent
from cle.harness.prompt_store import resolve_prompt_suites
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import CommunicationChoice
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import CommunicationAdmission, CommunicationOpportunity, ReactionReason
from cle.game_engine.events import PlayerEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from playground.game_viewer.routes.prompt_suite import prompt_suite_bp
from playground.game_viewer.state import ServerState


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@pytest.fixture
def prompt_app(tmp_path, monkeypatch):
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


def _edits(payload):
    return {
        "decision": {
            "system_identity": payload["decision"]["system_identity"],
            "components": payload["decision"]["components"],
            "phase_guidance": payload["decision"]["phase_guidance"],
            "response_instruction": payload["decision"]["response_instruction"],
        },
        "communication": {
            "system_identity": payload["communication"]["system_identity"],
            "components": payload["communication"]["components"],
        },
    }


def _save_payload(payload, edits):
    return {
        "expected": {
            "decision": payload["decision"]["sha256"],
            "communication": payload["communication"]["sha256"],
        },
        **edits,
    }


def _sandbox():
    engine = GameEngine(COLORS, seed=5, shuffle_players=False)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    return CatanSandbox(engine, players)


def test_prompt_suite_get_returns_fixed_strings_and_no_store(prompt_app):
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


def test_validate_save_stale_and_reset_prompt_strings(prompt_app):
    app, _ = prompt_app
    client = app.test_client()
    original = client.get("/api/prompt-suite").get_json()
    edits = _edits(original)
    edits["decision"]["components"]["board_state"] = (
        "CUSTOM BOARD COMPONENT:\n{{ value }}"
    )
    edits["decision"]["phase_guidance"]["initial_road_1"] = (
        "The second settlement remains independent. This road only seeds later expansion."
    )

    validated = client.post("/api/prompt-suite/validate", json=edits)
    unchanged = client.get("/api/prompt-suite").get_json()
    saved = client.put(
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
    after_stale = client.get("/api/prompt-suite").get_json()

    assert stale.status_code == 409
    assert after_stale["decision"]["sha256"] == saved.json["decision"]["sha256"]
    assert after_stale["communication"]["sha256"] == (
        saved.json["communication"]["sha256"]
    )

    reset = client.delete(
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


def test_invalid_component_edit_leaves_both_suites_unchanged(prompt_app):
    app, _ = prompt_app
    client = app.test_client()
    original = client.get("/api/prompt-suite").get_json()
    edits = _edits(original)
    edits["decision"]["components"]["resources"] = "{{ another_player_hand }}"

    validation = client.post("/api/prompt-suite/validate", json=edits)
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


def test_live_editing_is_allowed_but_replay_remains_read_only(prompt_app):
    app, state = prompt_app
    client = app.test_client()
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
    saved = client.put("/api/prompt-suite", json=save_payload)
    assert saved.status_code == 200
    reset_payload["expected"] = {
        kind: saved.json[kind]["sha256"] for kind in ("decision", "communication")
    }
    assert client.delete("/api/prompt-suite", json=reset_payload).status_code == 200

    state.current_sandbox = None
    state.replay_data = {"events": []}
    assert client.put("/api/prompt-suite", json=save_payload).status_code == 409
    assert client.delete("/api/prompt-suite", json=reset_payload).status_code == 409


def test_current_decision_preview_is_componentized_and_perspective_safe(prompt_app):
    app, state = prompt_app
    state.current_sandbox = _sandbox()
    engine_state = state.current_sandbox.game_engine.state
    engine_state.player_state["P0_WOOD_IN_HAND"] = 2
    engine_state.player_state["P1_ORE_IN_HAND"] = 4

    payload = app.test_client().get("/api/prompt-suite").get_json()
    preview = payload["preview"]["decision"]
    components = {component["id"]: component for component in preview["components"]}

    assert preview["status"] == "rendered"
    assert preview["actor"] == "RED"
    assert components["environment.resources"]["value"].startswith(
        "YOUR RESOURCES:"
    )
    assert "WOOD: 2" in components["environment.resources"]["value"]
    assert "ORE: 4" not in str(preview)
    assert components["environment.board_state"]["rendered"].startswith(
        "BOARD STATE:"
    )
    assert "build_settlement" in components["environment.legal_actions"]["value"]
    assert "<N00>" in components["environment.legal_actions"]["value"]
    assert "action_index" not in components["environment.decision_request"]["value"]
    board = preview["board_presentation"]
    assert board["kind"] == "text"
    assert board["format"] == "indexed_tile_rows/v3"
    assert board["identity_space"] == "canonical_engine_ids"
    assert "CATAN FULL PUBLIC GRAPH V1" in board["content"]
    assert "ORE: 4" not in board["content"]


@pytest.mark.parametrize("accepted", [False, True])
def test_latest_communication_preview_uses_traced_component_metadata(prompt_app, accepted):
    app, state = prompt_app
    sandbox = _sandbox()
    state.current_sandbox = sandbox
    request = ModelRequest(
        decision_id="talk:preview:BLUE",
        session_id="preview",
        messages=(ModelMessage("user", "TRIGGER:\nA road was built."),),
        components=(
            PromptComponent(
                id="environment.trigger",
                channel="environment",
                template="TRIGGER:\n{{ value }}",
                value="A road was built.",
                rendered="TRIGGER:\nA road was built.",
                variables=(("value", "A road was built."),),
            ),
        ),
    )
    opportunity = CommunicationOpportunity(
        player=Color.BLUE,
        cause=PlayerEvent(0, "action:0", Color.RED, "BUILD_ROAD", None),
        visible_through_sequence=0,
        reason=ReactionReason.MAJOR_BUILD,
        round=0,
    )
    sandbox.communication_trace.append(
        CommunicationAdmission(
            opportunity, CommunicationChoice(model_request=request), accepted=accepted,
            validation_error=None if accepted else "message rejected",
        )
    )

    preview = app.test_client().get("/api/prompt-suite").json["preview"][
        "communication"
    ]

    assert preview["status"] == "rendered"
    assert preview["components"] == [
        {
            "id": "environment.trigger",
            "channel": "environment",
            "template": "TRIGGER:\n{{ value }}",
            "value": "A road was built.",
            "rendered": "TRIGGER:\nA road was built.",
            "variables": {"value": "A road was built."},
        }
    ]
