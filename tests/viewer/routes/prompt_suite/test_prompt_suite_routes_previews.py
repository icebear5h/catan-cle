"""Decision and communication previews stay componentized and safe."""

import pytest
from flask import Flask

from cle.game_engine.events import PlayerEvent
from cle.game_engine.models.player import Color
from cle.harness import ModelMessage, ModelRequest, PromptComponent
from cle.players.contracts import CommunicationChoice
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    ReactionReason,
)
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.state import ServerState

from .support import _sandbox


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



def test_current_decision_preview_is_componentized_and_perspective_safe(prompt_app: tuple[Flask, ServerState]) -> None:
    app, state = prompt_app
    state.current_sandbox = _sandbox()
    engine_state = live_sandbox(state).game_engine.state
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
def test_latest_communication_preview_uses_traced_component_metadata(prompt_app: tuple[Flask, ServerState], accepted: bool) -> None:
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
