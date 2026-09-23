"""Shared prompt edits round-trip and strict schema rejects invalid writes."""
import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.harness.models import ModelMessage, ModelRequest
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import CommunicationChoice, PlayerContext
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.communication import CommunicationAdmission
from cle.sandbox.factory import LiveSandboxConfig, create_live_sandbox
from playground.game_viewer.routes import prompt_suite as prompt_routes

from .conftest import COLORS, NeverTransport, Studio


def test_shared_roundtrip_reorder_conflict_and_reset(studio: Studio, tmp_path: Path) -> None:
    client, _ = studio
    response = client.get("/api/prompt-suite")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    original: Any = response.json
    assert original["mode"] == "shared"
    assert "decision" not in original and "communication" not in original
    document: Any = deepcopy(original["shared"]["document"])
    assert set(document) == {
        "id", "version", "status", "components", "compositions", "phase_guidance", "memory_mode",
        "max_notes_chars", "initial_placement_order", "reactive_speech", "deterministic_batches",
    }
    assert document["memory_mode"] == "fresh_notes"
    assert document["status"] == "active"
    document["components"]["board_state"]["template"] = "SHARED BOARD EDIT:\n{{ board_state }}"
    document["compositions"]["decision"]["order"].reverse()
    speech = document["compositions"]["speech"]["order"]
    speech.append(speech.pop(0))
    validated: Any = client.post("/api/prompt-suite/validate", json={"shared": document})
    assert validated.status_code == 200
    assert validated.json["candidate"]["shared"]["document"] == document
    assert client.get("/api/prompt-suite").json["shared"] == original["shared"]
    payload = {"shared": document, "expected": {"shared": original["shared"]["sha256"]}}
    saved: Any = client.put("/api/prompt-suite", json=payload)
    assert saved.status_code == 200
    assert saved.json["shared"]["document"] == document
    assert saved.json["shared"]["overridden"] is True
    assert client.get("/api/prompt-suite").json["shared"] == saved.json["shared"]
    assert client.put("/api/prompt-suite", json=payload).status_code == 409
    stale = client.delete("/api/prompt-suite", json={"expected": payload["expected"]})
    assert stale.status_code == 409
    assert stale.headers["Cache-Control"] == "no-store"
    reset: Any = client.delete("/api/prompt-suite", json={"expected": {"shared": saved.json["shared"]["sha256"]}})
    assert reset.status_code == 200
    assert reset.json["shared"] == original["shared"]
    assert not (tmp_path / "prompts" / "decision.yaml").exists()
    assert not (tmp_path / "prompts" / "communication.yaml").exists()


@pytest.mark.parametrize("problem", ["extra", "boolean_version", "unknown_input", "channel", "missing", "duplicate", "unknown", "response"])
def test_strict_schema_rejects_invalid_shared_edits_without_writes(studio: Studio, problem: str) -> None:
    client, _ = studio
    original: Any = client.get("/api/prompt-suite").json
    document: Any = deepcopy(original["shared"]["document"])
    order = document["compositions"]["speech"]["order"]
    if problem == "extra":
        document["include"] = "another.yaml"
    elif problem == "boolean_version":
        document["version"] = True
    elif problem == "unknown_input":
        document["components"]["notes"]["inputs"] = ["secret_hand"]
    elif problem == "channel":
        document["components"]["notes"]["channel"] = "assistant"
    elif problem == "missing":
        order.remove("notes")
    elif problem == "duplicate":
        order.append(order[0])
    elif problem == "unknown":
        order.append("not_defined")
    else:
        document["compositions"]["speech"]["response"] = "not_in_order"
    validated: Any = client.post("/api/prompt-suite/validate", json={"shared": document})
    assert validated.status_code == 400
    assert validated.json["valid"] is False
    assert validated.headers["Cache-Control"] == "no-store"
    saved = client.put("/api/prompt-suite", json={
        "shared": document, "expected": {"shared": original["shared"]["sha256"]},
    })
    assert saved.status_code == 400
    assert client.get("/api/prompt-suite").json["shared"] == original["shared"]


def test_no_context_never_invents_rendered_game_values(studio: Studio) -> None:
    client, _ = studio
    payload: Any = client.get("/api/prompt-suite").json
    assert payload["preview"]["decision"] == {"status": "no_game_context", "components": []}
    communication = payload["preview"]["communication"]
    assert communication["status"] == "no_current_communication_context"
    assert all(not item["rendered"] and not item["variables"] for item in communication["components"])


def test_shared_candidate_renders_current_typed_context_not_recorded_text(studio: Studio, monkeypatch: pytest.MonkeyPatch) -> None:
    client, state = studio
    sandbox = create_live_sandbox(LiveSandboxConfig(
        mode="llm", seed=3, palette="canonical_four", shuffle_players=False,
    ), transport=NeverTransport())
    state.current_sandbox = sandbox
    engine = sandbox.game_engine
    actor = sandbox.current_actor()
    # These are actual private engine holdings, not stand-in preview strings.
    engine.state.player_state["P0_WOOD_IN_HAND"] = 2
    engine.state.player_state["P1_ORE_IN_HAND"] = 4
    player: Any = sandbox.players[actor]
    player.session.strategic_memory = "Current accepted private notes"
    action = engine.state.playable_actions[0]
    transition = engine.step(action)
    opportunities = sandbox.communication_policy.after_events(engine, transition.events, round_number=0)
    opportunity = next(item for item in opportunities if item.player != actor)
    sandbox.communication_trace.append(CommunicationAdmission(
        opportunity, CommunicationChoice(model_request=ModelRequest(
            decision_id="old", session_id="old", messages=(ModelMessage("user", "RECORDED TEXT MUST NOT BE REUSED"),),
        )), accepted=True,
    ))
    original_snapshot = pickle.dumps(sandbox.snapshot())
    original_decision_preview = prompt_routes._decision_preview
    original_speech_preview = prompt_routes._communication_preview
    original_saving_locked = prompt_routes._saving_locked

    def checked_decision(*args: object, **kwargs: object) -> dict[str, Any]:
        assert state.replay_mutation_lock._is_owned()
        return original_decision_preview(*args, **kwargs)

    def checked_speech(*args: object, **kwargs: object) -> dict[str, Any]:
        assert state.replay_mutation_lock._is_owned()
        return original_speech_preview(*args, **kwargs)

    def checked_saving_locked(*args: object, **kwargs: object) -> bool:
        assert state.replay_mutation_lock._is_owned()
        return original_saving_locked(*args, **kwargs)

    monkeypatch.setattr(prompt_routes, "_decision_preview", checked_decision)
    monkeypatch.setattr(prompt_routes, "_communication_preview", checked_speech)
    monkeypatch.setattr(prompt_routes, "_saving_locked", checked_saving_locked)
    document = client.get("/api/prompt-suite").json["shared"]["document"]
    document["components"]["board_state"]["template"] = "EDITED CURRENT BOARD:\n{{ board_state }}"
    validated: Any = client.post("/api/prompt-suite/validate", json={"shared": document})
    assert validated.status_code == 200
    preview = validated.json["candidate"]["preview"]
    for consumer in ("decision", "communication"):
        group = preview[consumer]
        assert group["status"] == "rendered"
        board = next(item for item in group["components"] if item["id"] == "environment.board_state")
        assert board["rendered"].startswith("EDITED CURRENT BOARD:")
        assert board["variables"]["board_state"]
        assert "RECORDED TEXT MUST NOT BE REUSED" not in str(group)
    assert "ORE: 4" not in str(preview["decision"])
    assert pickle.dumps(sandbox.snapshot()) == original_snapshot


def test_replay_tuple_context_without_players_or_communication_trace(studio: Studio) -> None:
    client, state = studio
    engine = GameEngine(COLORS, seed=5, shuffle_players=False)
    sandbox = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    context = sandbox.decision_context()

    def replay_context() -> tuple[PlayerContext, dict[str, Any]]:
        assert state.replay_mutation_lock._is_owned()
        with state.replay_mutation_lock:
            return context, {"replay_index": 0}

    state.current_sandbox = SimpleNamespace(
        game_engine=engine, decision_context=replay_context,
    )
    state.replay_mode = True
    payload: Any = client.get("/api/prompt-suite")
    assert payload.status_code == 200
    assert payload.json["preview"]["decision"]["status"] == "rendered"
    assert payload.json["preview"]["communication"]["status"] == "no_current_communication_context"
    assert payload.json["saving_locked"] is True
    document = payload.json["shared"]["document"]
    assert client.post("/api/prompt-suite/validate", json={"shared": document}).status_code == 200
