"""Prompt Studio roundtrips use only temporary storage and offline contexts."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import pickle
from threading import Event, RLock
from types import SimpleNamespace

import pytest
from flask import Flask

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness.models import ModelMessage, ModelRequest
from cle.harness import prompt_store
from cle.harness.prompt_store import load_active_prompt_suites, save_prompt_suite_overrides
from cle.harness.shared_suite import default_shared_suite_path
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import CommunicationChoice
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.communication import CommunicationAdmission
from cle.sandbox.factory import LiveSandboxConfig, create_live_sandbox
from playground.game_viewer.routes.prompt_suite import prompt_suite_bp
from playground.game_viewer.routes import prompt_suite as prompt_routes


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class NeverTransport:
    async def complete(self, request):
        raise AssertionError("No inference is permitted in prompt tests")


@pytest.fixture
def studio(tmp_path, monkeypatch):
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


def test_shared_roundtrip_reorder_conflict_and_reset(studio, tmp_path):
    client, _ = studio
    response = client.get("/api/prompt-suite")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    original = response.json
    assert original["mode"] == "shared"
    assert "decision" not in original and "communication" not in original
    document = deepcopy(original["shared"]["document"])
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
    validated = client.post("/api/prompt-suite/validate", json={"shared": document})
    assert validated.status_code == 200
    assert validated.json["candidate"]["shared"]["document"] == document
    assert client.get("/api/prompt-suite").json["shared"] == original["shared"]
    payload = {"shared": document, "expected": {"shared": original["shared"]["sha256"]}}
    saved = client.put("/api/prompt-suite", json=payload)
    assert saved.status_code == 200
    assert saved.json["shared"]["document"] == document
    assert saved.json["shared"]["overridden"] is True
    assert client.get("/api/prompt-suite").json["shared"] == saved.json["shared"]
    assert client.put("/api/prompt-suite", json=payload).status_code == 409
    stale = client.delete("/api/prompt-suite", json={"expected": payload["expected"]})
    assert stale.status_code == 409
    assert stale.headers["Cache-Control"] == "no-store"
    reset = client.delete("/api/prompt-suite", json={"expected": {"shared": saved.json["shared"]["sha256"]}})
    assert reset.status_code == 200
    assert reset.json["shared"] == original["shared"]
    assert not (tmp_path / "prompts" / "decision.yaml").exists()
    assert not (tmp_path / "prompts" / "communication.yaml").exists()


@pytest.mark.parametrize("problem", ["extra", "boolean_version", "unknown_input", "channel", "missing", "duplicate", "unknown", "response"])
def test_strict_schema_rejects_invalid_shared_edits_without_writes(studio, problem):
    client, _ = studio
    original = client.get("/api/prompt-suite").json
    document = deepcopy(original["shared"]["document"])
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
    validated = client.post("/api/prompt-suite/validate", json={"shared": document})
    assert validated.status_code == 400
    assert validated.json["valid"] is False
    assert validated.headers["Cache-Control"] == "no-store"
    saved = client.put("/api/prompt-suite", json={
        "shared": document, "expected": {"shared": original["shared"]["sha256"]},
    })
    assert saved.status_code == 400
    assert client.get("/api/prompt-suite").json["shared"] == original["shared"]


def test_no_context_never_invents_rendered_game_values(studio):
    client, _ = studio
    payload = client.get("/api/prompt-suite").json
    assert payload["preview"]["decision"] == {"status": "no_game_context", "components": []}
    communication = payload["preview"]["communication"]
    assert communication["status"] == "no_current_communication_context"
    assert all(not item["rendered"] and not item["variables"] for item in communication["components"])


def test_shared_candidate_renders_current_typed_context_not_recorded_text(studio, monkeypatch):
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
    player = sandbox.players[actor]
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

    def checked_decision(*args, **kwargs):
        assert state.replay_mutation_lock._is_owned()
        return original_decision_preview(*args, **kwargs)

    def checked_speech(*args, **kwargs):
        assert state.replay_mutation_lock._is_owned()
        return original_speech_preview(*args, **kwargs)

    def checked_saving_locked(*args, **kwargs):
        assert state.replay_mutation_lock._is_owned()
        return original_saving_locked(*args, **kwargs)

    monkeypatch.setattr(prompt_routes, "_decision_preview", checked_decision)
    monkeypatch.setattr(prompt_routes, "_communication_preview", checked_speech)
    monkeypatch.setattr(prompt_routes, "_saving_locked", checked_saving_locked)
    document = client.get("/api/prompt-suite").json["shared"]["document"]
    document["components"]["board_state"]["template"] = "EDITED CURRENT BOARD:\n{{ board_state }}"
    validated = client.post("/api/prompt-suite/validate", json={"shared": document})
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


def test_replay_tuple_context_without_players_or_communication_trace(studio):
    client, state = studio
    engine = GameEngine(COLORS, seed=5, shuffle_players=False)
    sandbox = CatanSandbox(engine, {color: FirstLegalPlayer(color) for color in COLORS})
    context = sandbox.decision_context()

    def replay_context():
        assert state.replay_mutation_lock._is_owned()
        with state.replay_mutation_lock:
            return context, {"replay_index": 0}

    state.current_sandbox = SimpleNamespace(
        game_engine=engine, decision_context=replay_context,
    )
    state.replay_mode = True
    payload = client.get("/api/prompt-suite")
    assert payload.status_code == 200
    assert payload.json["preview"]["decision"]["status"] == "rendered"
    assert payload.json["preview"]["communication"]["status"] == "no_current_communication_context"
    assert payload.json["saving_locked"] is True
    document = payload.json["shared"]["document"]
    assert client.post("/api/prompt-suite/validate", json={"shared": document}).status_code == 200


@pytest.mark.parametrize("field,value", [("replay_mode", True), ("replay_data", {})])
def test_loaded_game_save_and_reset_lock(studio, field, value):
    client, state = studio
    original = client.get("/api/prompt-suite").json
    setattr(state, field, value)
    expected = {"shared": original["shared"]["sha256"]}
    for response in (
        client.put("/api/prompt-suite", json={"expected": expected, "shared": original["shared"]["document"]}),
        client.delete("/api/prompt-suite", json={"expected": expected}),
    ):
        assert response.status_code == 409
        assert response.headers["Cache-Control"] == "no-store"


def test_existing_legacy_pair_remains_editable(studio):
    client, _ = studio
    active = load_active_prompt_suites()
    save_prompt_suite_overrides(
        decision_source=active.decision.source, communication_source=active.communication.source,
        expected_decision_sha256=active.decision.sha256,
        expected_communication_sha256=active.communication.sha256,
    )
    original = client.get("/api/prompt-suite").json
    assert original["mode"] == "legacy"
    edits = {
        "decision": {key: original["decision"][key] for key in (
            "system_identity", "components", "phase_guidance", "response_instruction",
        )},
        "communication": {key: original["communication"][key] for key in ("system_identity", "components")},
    }
    edits["decision"]["components"]["board_state"] = "LEGACY EDIT:\n{{ value }}"
    saved = client.put("/api/prompt-suite", json={
        **edits, "expected": {key: original[key]["sha256"] for key in ("decision", "communication")},
    })
    assert saved.status_code == 200
    assert saved.json["mode"] == "legacy"
    assert saved.json["decision"]["components"]["board_state"].startswith("LEGACY EDIT:")
    reset = client.delete("/api/prompt-suite", json={
        "expected": {key: saved.json[key]["sha256"] for key in ("decision", "communication")},
    })
    assert reset.status_code == 200
    assert reset.json["mode"] == "shared"
    assert reset.json["shared"] == client.get("/api/prompt-suite").json["shared"]


def test_environment_source_is_visible_but_cannot_be_shadowed_by_editor(studio, monkeypatch):
    client, _ = studio
    monkeypatch.setenv("CATAN_SHARED_SUITE", str(default_shared_suite_path()))
    original = client.get("/api/prompt-suite").json
    payload = {"expected": {"shared": original["shared"]["sha256"]}}
    saved = client.put("/api/prompt-suite", json={**payload, "shared": original["shared"]["document"]})
    reset = client.delete("/api/prompt-suite", json=payload)
    assert saved.status_code == reset.status_code == 409
    assert "environment" in saved.json["error"]


def test_runtime_path_selection_is_visible_and_cannot_be_shadowed(studio, tmp_path):
    client, state = studio
    path = tmp_path / "explicit-shared.yaml"
    path.write_text(default_shared_suite_path().read_text().replace(
        "{{ board_state }}", "EXPLICIT ACTIVE BOARD {{ board_state }}",
    ))
    state.active_live_config = LiveSandboxConfig(shared_suite_path=str(path))
    original = client.get("/api/prompt-suite").json
    assert "EXPLICIT ACTIVE BOARD" in str(original["shared"]["document"])
    saved = client.put("/api/prompt-suite", json={
        "shared": original["shared"]["document"],
        "expected": {"shared": original["shared"]["sha256"]},
    })
    assert saved.status_code == 409
    assert "Explicit runtime prompt paths" in saved.json["error"]


def test_shared_payload_rejects_legacy_or_unknown_root_keys(studio):
    client, _ = studio
    original = client.get("/api/prompt-suite").json
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
def test_reset_endpoint_preserves_override_when_shared_default_cannot_load(studio, tmp_path, monkeypatch, mode, failure):
    client, _ = studio
    if mode == "legacy":
        pair = load_active_prompt_suites()
        save_prompt_suite_overrides(
            decision_source=pair.decision.source, communication_source=pair.communication.source,
            expected_decision_sha256=pair.decision.sha256,
            expected_communication_sha256=pair.communication.sha256,
        )
    else:
        initial = client.get("/api/prompt-suite").json
        saved = client.put("/api/prompt-suite", json={
            "shared": initial["shared"]["document"], "expected": {"shared": initial["shared"]["sha256"]},
        })
        assert saved.status_code == 200
    original = client.get("/api/prompt-suite").json
    kinds = ("shared",) if mode == "shared" else ("decision", "communication")
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


@pytest.mark.parametrize("guidance", [{}, {"main_game": "Only one phase"}])
def test_incomplete_guidance_is_rejected_without_game_context(studio, guidance):
    client, _ = studio
    original = client.get("/api/prompt-suite").json
    document = deepcopy(original["shared"]["document"])
    document["phase_guidance"] = guidance
    validated = client.post("/api/prompt-suite/validate", json={"shared": document})
    assert validated.status_code == 400
    assert "phase_guidance" in validated.json["errors"][0]["message"]
    saved = client.put("/api/prompt-suite", json={
        "shared": document, "expected": {"shared": original["shared"]["sha256"]},
    })
    assert saved.status_code == 400
    assert client.get("/api/prompt-suite").json == original


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
@pytest.mark.parametrize("mode", ["shared", "legacy"])
def test_write_uses_store_lock_independently_of_inference_lock(studio, monkeypatch, method, mode):
    client, state = studio
    if mode == "legacy":
        pair = load_active_prompt_suites()
        save_prompt_suite_overrides(
            decision_source=pair.decision.source, communication_source=pair.communication.source,
            expected_decision_sha256=pair.decision.sha256,
            expected_communication_sha256=pair.communication.sha256,
        )
    original = client.get("/api/prompt-suite").json
    kinds = ("shared",) if mode == "shared" else ("decision", "communication")
    payload = {"expected": {kind: original[kind]["sha256"] for kind in kinds}}
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

    def checked_game_state(value):
        operations.append("check")
        return original_check(value)

    @contextmanager
    def checked_store_lock(directory):
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
def test_source_published_before_preview_waits_for_inference(studio, method):
    client, state = studio
    original = client.get("/api/prompt-suite").json
    payload = {"expected": {"shared": original["shared"]["sha256"]}}
    if method == "PUT":
        payload["shared"] = original["shared"]["document"]
        payload["shared"]["components"]["board_state"]["template"] = "LIVE EDIT\n{{ board_state }}"
    attempted = Event()
    lock = state.replay_mutation_lock

    class ObservedLock:
        def __enter__(self):
            attempted.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    state.replay_mutation_lock = ObservedLock()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with lock:
            pending = pool.submit(client.open, "/api/prompt-suite", method=method, json=payload)
            assert attempted.wait(3), "Preview did not reach the shared state lock"
            assert not pending.done()
            active = prompt_store.resolve_prompt_suites()
            if method == "PUT":
                assert "LIVE EDIT" in active.shared.source
        response = pending.result(timeout=3)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert client.get("/api/prompt-suite").json["shared"] == response.json["shared"]


def test_empty_source_environment_does_not_block_studio(studio, monkeypatch):
    client, _ = studio
    for name in ("CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE"):
        monkeypatch.setenv(name, "")
    original = client.get("/api/prompt-suite")
    assert original.status_code == 200 and original.json["mode"] == "shared"
    saved = client.put("/api/prompt-suite", json={
        "shared": original.json["shared"]["document"],
        "expected": {"shared": original.json["shared"]["sha256"]},
    })
    assert saved.status_code == 200
    reset = client.delete("/api/prompt-suite", json={"expected": {"shared": saved.json["shared"]["sha256"]}})
    assert reset.status_code == 200
    assert reset.json["shared"] == original.json["shared"]
