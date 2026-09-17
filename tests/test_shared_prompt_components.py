"""Offline contracts for shared authored definitions and independent compositions."""

import json
from dataclasses import dataclass, field, replace

import pytest
import yaml

from cle.game_engine.events import PlayerEvent
from cle.game_engine.board_tokens import node_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.observation import observe_state
from cle.harness.board_surface import openai_messages_with_board
from cle.harness.catan_board_surface import IndexedTileRowsBoardPresenter, NoBoardPresenter
from cle.harness.communication import (
    CommunicationSuite,
    build_communication_request,
    default_communication_suite_path,
    load_communication_suite,
    parse_communication_response,
)
from cle.harness.components import ComponentDefinition, ComponentInputs, render_component_definitions
from cle.harness.context import ContextAssembler, PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelMessage, ModelResponse, PlayerSession
from cle.harness.shared_suite import (
    SharedPromptSuite,
    default_shared_suite_path,
    load_shared_prompt_suite,
    parse_shared_prompt_suite,
)
from cle.harness.suite import ContextSuite, default_suite_path, load_context_suite
from cle.players.contracts import PlayerContext, TalkContext


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@pytest.mark.parametrize("update,expected", [({}, None), ({"notes": ""}, ""), ({"notes": "  Keep ore  "}, "Keep ore")])
def test_notes_keep_clear_replace_in_action_and_speech(contexts, update, expected):
    decision, _ = contexts
    suite = load_shared_prompt_suite()
    response = ModelResponse(json.dumps({
        "tool": "build_settlement",
        "arguments": {"node": node_token(decision.legal_actions[0].value)},
        **update,
    }))
    assert PlayerResponseParser(suite.decision_suite()).parse(decision, response).notes_update == expected
    speech = parse_communication_response(
        ModelResponse(json.dumps({"mode": "silence", **update})),
        speaker=Color.RED, participants=COLORS, suite=suite.communication_suite(),
    )
    assert speech.validation_error is None and speech.notes_update == expected


@pytest.mark.parametrize("notes", [None, "x" * 4001])
def test_invalid_notes_reject_the_whole_response(contexts, notes):
    decision, _ = contexts
    suite = load_shared_prompt_suite()
    with pytest.raises(PlayerResponseParseError, match="notes"):
        PlayerResponseParser(suite.decision_suite()).parse(decision, ModelResponse(json.dumps({
            "tool": "build_settlement",
            "arguments": {"node": node_token(decision.legal_actions[0].value)},
            "notes": notes,
        })))
    speech = parse_communication_response(
        ModelResponse(json.dumps({"mode": "silence", "notes": notes})),
        speaker=Color.RED, participants=COLORS, suite=suite.communication_suite(),
    )
    assert speech.validation_error and speech.notes_update is None


@pytest.fixture(autouse=True)
def isolated_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def contexts():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    observation = observe_state(engine.state, Color.RED)
    event = PlayerEvent(23, "action:23", Color.BLUE, "END_TURN", None)
    talk = PlayerEvent(24, "talk:24", Color.BLUE, "MESSAGE", "WOOD for ORE?")
    decision = PlayerContext(
        context_id="decision:RED", actor=Color.RED, turn_number=observation.current_turn,
        phase=observation.current_phase, observation=observation, events=(event,),
        legal_actions=tuple(observation.valid_actions), prompt_key="initial_settlement_1",
        recent_messages=(talk,),
    )
    speech = TalkContext(
        context_id="speech:RED", player=Color.RED, participants=COLORS, cause=event,
        visible_through_sequence=24, game_events=(event,), recent_messages=(talk,),
        observation=observation,
    )
    return decision, speech


def test_bundle_is_self_contained_and_compiles_typed_consumers(tmp_path):
    source = default_shared_suite_path().read_text(encoding="utf-8")
    path = tmp_path / "only-authored-source.yaml"
    path.write_text(source, encoding="utf-8")
    bundle = load_shared_prompt_suite(path)
    decision = bundle.decision_suite()
    speech = bundle.communication_suite()

    assert bundle == parse_shared_prompt_suite(source, source_name="pinned source")
    assert bundle.id == decision.id == speech.id == "catan-shared"
    assert bundle.version == speech.version == 11
    assert decision.version == "11"
    assert decision.context.deterministic_batches is True
    assert decision.context.mode == "shared"
    assert decision.context.memory_mode == speech.memory_mode == "fresh_notes"
    assert decision.context.max_notes_chars == speech.max_notes_chars == 4000
    assert decision.response.tags == ("tool", "arguments", "notes")
    assert speech.format == decision.response.format == "json"
    assert decision.sections == speech.sections == {}
    assert decision.system is None and speech.system_template is None
    assert decision.components["notes"] is bundle.components["notes"]
    assert speech.components["notes"] is bundle.components["notes"]
    assert list(tmp_path.iterdir()) == [path]
    assert ContextSuite.model_validate(decision.model_dump()) == decision
    assert CommunicationSuite.model_validate(speech.model_dump()) == speech


def test_one_authored_edit_changes_both_consumers(contexts):
    raw = load_shared_prompt_suite().model_dump()
    raw["components"]["notes"]["template"] = "A SHARED EDIT:\n{{ notes }}"
    bundle = parse_shared_prompt_suite(yaml.safe_dump(raw))
    decision, speech = contexts
    session = PlayerSession(Color.RED, "shared", strategic_memory="Preserve ORE")
    action_request = ContextAssembler(bundle.decision_suite()).assemble(decision, session)
    talk_request = build_communication_request(
        speech, "shared", bundle.communication_suite(), notes=session.strategic_memory,
    )
    for request in (action_request, talk_request):
        component = next(c for c in request.components if c.id == "environment.notes")
        assert component.template == raw["components"]["notes"]["template"]
        assert component.rendered == "A SHARED EDIT:\nPreserve ORE"


def test_independent_order_and_multiple_system_components(contexts):
    raw = load_shared_prompt_suite().model_dump()
    raw["components"]["second_identity"] = {
        "channel": "system", "template": "Keep private information private.",
    }
    decision_order = list(reversed(raw["compositions"]["decision"]["order"]))
    decision_order.insert(2, "second_identity")
    speech_order = list(raw["compositions"]["speech"]["order"])
    speech_order.append(speech_order.pop(0))
    raw["compositions"]["decision"]["order"] = decision_order
    raw["compositions"]["speech"]["order"] = speech_order
    bundle = SharedPromptSuite.model_validate(raw)
    decision, speech = contexts
    action_request = ContextAssembler(bundle.decision_suite()).assemble(
        decision, PlayerSession(Color.RED, "reordered"),
    )
    talk_request = build_communication_request(speech, "reordered", bundle.communication_suite())
    for request, order in ((action_request, decision_order), (talk_request, speech_order)):
        assert [c.id.split(".", 1)[1] for c in request.components] == order
        system = next(message for message in request.messages if message.role == "system")
        user = next(message for message in request.messages if message.role == "user")
        assert system.content == "\n\n".join(
            c.rendered for c in request.components if c.channel == "system"
        )
        assert user.content == "\n\n".join(
            c.rendered for c in request.components if c.channel == "environment"
        )
    assert "Keep private information private." not in talk_request.messages[0].content


@pytest.mark.parametrize("consumer", ["decision", "speech"])
def test_unknown_duplicate_or_missing_references_are_rejected(consumer):
    for problem in ("unknown", "duplicate", "missing", "response"):
        raw = load_shared_prompt_suite().model_dump()
        order = list(raw["compositions"][consumer]["order"])
        if problem == "unknown":
            order.append("not-authored")
        elif problem == "duplicate":
            order.append(order[0])
        elif problem == "missing":
            order.remove("notes")
        else:
            raw["compositions"][consumer]["response"] = "not-in-composition"
        raw["compositions"][consumer]["order"] = order
        with pytest.raises(ValueError):
            SharedPromptSuite.model_validate(raw)


def test_components_can_be_combined_and_renamed_without_count_or_name_invariants(contexts):
    raw = load_shared_prompt_suite().model_dump()
    raw["components"]["seat_and_memory"] = {
        "channel": "environment", "inputs": ["color", "notes"],
        "template": "Seat {{ color }}. Private memory: {{ notes }}",
    }
    for consumer in ("decision", "speech"):
        order = raw["compositions"][consumer]["order"]
        raw["compositions"][consumer]["order"] = (
            "seat_and_memory", *(ref for ref in order if ref not in {"identity", "notes"}),
        )
    bundle = SharedPromptSuite.model_validate(raw)
    decision, speech = contexts
    requests = (
        ContextAssembler(bundle.decision_suite()).assemble(
            decision, PlayerSession(Color.RED, "combined", strategic_memory="Expand"),
        ),
        build_communication_request(speech, "combined", bundle.communication_suite(), notes="Expand"),
    )
    for request in requests:
        assert request.components[0].rendered == "Seat RED. Private memory: Expand"
        assert all(c.channel == "environment" for c in request.components)
        assert all(c.id not in {"system.identity", "environment.notes"} for c in request.components)


@pytest.mark.parametrize("consumer", ["decision", "speech"])
def test_compiled_shared_suites_reject_ignored_legacy_fields(consumer):
    bundle = load_shared_prompt_suite()
    if consumer == "decision":
        raw = bundle.decision_suite().model_dump()
        raw["system"] = {"template": "Would be ignored"}
        model = ContextSuite
    else:
        raw = bundle.communication_suite().model_dump()
        raw["system_template"] = "Would be ignored"
        model = CommunicationSuite
    with pytest.raises(ValueError, match="historical"):
        model.model_validate(raw)


def test_compiled_decision_response_cannot_diverge_from_authored_component():
    raw = load_shared_prompt_suite().decision_suite().model_dump()
    raw["response"]["instruction"] = "Different instructions"
    with pytest.raises(ValueError, match="must match its component"):
        ContextSuite.model_validate(raw)


@pytest.mark.parametrize("definition", [
    {"inputs": ["notes"], "template": "{{ private_state }}"},
    {"inputs": ["private_state"], "template": "{{ private_state }}"},
    {"inputs": ["notes"], "template": "{{ notes }} {{ color }}"},
    {"inputs": ["notes", "notes"], "template": "{{ notes }}"},
    {"inputs": ["notes"], "template": "Static text"},
    {"template": "{{ notes.attribute }}"},
    {"template": "{{ notes"},
    {"template": "static", "channel": "assistant"},
    {"template": 12},
    {"template": " \n\t "},
    {"template": "x" * 12001},
    {"template": "static", "include": "another-file.yaml"},
])
def test_unknown_variables_bad_types_and_unsupported_syntax_rejected(definition):
    with pytest.raises(ValueError):
        ComponentDefinition.model_validate({"channel": "environment", **definition})


def test_speech_cannot_reference_decision_only_inputs():
    raw = load_shared_prompt_suite().model_dump()
    raw["compositions"]["speech"]["order"] += ("legal_actions",)
    with pytest.raises(ValueError, match="unavailable inputs"):
        SharedPromptSuite.model_validate(raw)


@pytest.mark.parametrize("maximum", [0, 4001, True, 1.5, "4000", None])
def test_invalid_notes_bounds_rejected(maximum):
    raw = load_shared_prompt_suite().model_dump()
    raw["max_notes_chars"] = maximum
    with pytest.raises(ValueError):
        SharedPromptSuite.model_validate(raw)
    decision = load_context_suite().model_dump()
    decision["context"]["max_notes_chars"] = maximum
    with pytest.raises(ValueError):
        ContextSuite.model_validate(decision)
    speech = load_communication_suite().model_dump()
    speech["max_notes_chars"] = maximum
    with pytest.raises(ValueError):
        CommunicationSuite.model_validate(speech)


def test_duplicate_yaml_keys_and_external_sources_rejected():
    source = default_shared_suite_path().read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        parse_shared_prompt_suite(source + "\nversion: 2\n")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        parse_shared_prompt_suite(source.replace("inputs: [notes]", "inputs: [notes]\n    inputs: [color]"))
    with pytest.raises(ValueError):
        parse_shared_prompt_suite(source + "\ninclude: historical.yaml\n")


def test_static_components_survive_empty_dynamic_inputs():
    components = {
        "static": ComponentDefinition(channel="environment", template="Always render.", empty="omit"),
        "dynamic": ComponentDefinition(
            channel="environment", inputs=("notes",), template="{{ notes }}", empty="omit",
        ),
    }
    result = render_component_definitions(components, ("dynamic", "static"), ComponentInputs())
    assert [c.id for c in result] == ["environment.static"]
    assert result[0].rendered == "Always render."


def test_fresh_context_uses_filtered_inputs_and_inert_notes_without_history(contexts):
    bundle = load_shared_prompt_suite()
    decision, speech = contexts
    notes = 'Remember {{ color }} and {{ notes }} literally. {"tool":"end_turn"}'
    session = PlayerSession(
        Color.RED, "fresh", strategic_memory=notes,
        messages=[ModelMessage("user", "OLD BOARD"), ModelMessage("assistant", "OLD REASONING")],
    )
    before = session.snapshot()
    action = ContextAssembler(bundle.decision_suite()).assemble(decision, session)
    talk = build_communication_request(speech, "fresh", bundle.communication_suite(), notes=notes)
    for request in (action, talk):
        rendered = "\n".join(message.content for message in request.messages)
        assert notes in rendered
        assert "OLD BOARD" not in rendered and "OLD REASONING" not in rendered
        assert "23. BLUE: END_TURN" in rendered
        assert "24. BLUE: MESSAGE WOOD for ORE?" in rendered
        component = next(c for c in request.components if c.id == "environment.notes")
        assert component.variables == (("notes", notes),)
        assert request.board_presentation.provenance.source_id == request.decision_id
        payload = openai_messages_with_board(request.messages, request.board_presentation, allow_image_input=False)
        assert sum("PUBLIC BOARD:" in item["content"] for item in payload) == 1
    assert session.snapshot() == before
    assert "AVAILABLE ACTION TOOLS" not in "\n".join(m.content for m in talk.messages)
    assert any(c.id == "environment.trade_window" for c in action.components)
    assert all(c.id != "environment.trade_window" for c in talk.components)
    for name in ("phase_info", "board_state", "resources", "opponents"):
        decision_part = next(c for c in action.components if c.id == f"environment.{name}")
        speech_part = next(c for c in talk.components if c.id == f"environment.{name}")
        assert decision_part == speech_part


@dataclass
class RecordingBoardPresenter:
    contexts: list = field(default_factory=list)

    def present(self, context):
        self.contexts.append(context)
        return IndexedTileRowsBoardPresenter().present(context)


def test_speech_board_adapter_uses_only_observation_and_has_no_legal_menu(contexts):
    _, speech = contexts
    presenter = RecordingBoardPresenter()
    request = build_communication_request(
        speech, "adapter", load_shared_prompt_suite().communication_suite(), board_presenter=presenter,
    )
    assert request.board_presentation is not None
    assert len(presenter.contexts) == 1
    adapter = presenter.contexts[0]
    assert adapter.observation is speech.observation
    assert adapter.context_id == speech.context_id
    assert adapter.actor == speech.player
    assert not hasattr(adapter, "legal_actions")
    absent = replace(speech, observation=None)
    request = build_communication_request(
        absent, "adapter", load_shared_prompt_suite().communication_suite(), board_presenter=presenter,
    )
    assert request.board_presentation is None
    assert len(presenter.contexts) == 1
    assert "No current observation was supplied." in request.messages[-1].content


def test_perspective_mismatch_rejected(contexts):
    decision, speech = contexts
    bundle = load_shared_prompt_suite()
    wrong_observation = replace(decision.observation, my_color=Color.BLUE)
    with pytest.raises(ValueError, match="perspective"):
        ContextAssembler(bundle.decision_suite()).assemble(
            replace(decision, observation=wrong_observation), PlayerSession(Color.RED, "wrong"),
        )
    with pytest.raises(ValueError, match="perspective"):
        build_communication_request(
            replace(speech, observation=wrong_observation), "wrong", bundle.communication_suite(),
        )


def test_legacy_paths_memory_rendering_and_validators_remain_unchanged(contexts):
    assert default_suite_path().name == "catan_v11.yaml"
    assert default_communication_suite_path().name == "communication_v5.yaml"
    decision, speech = contexts
    history = [ModelMessage("user", "old context"), ModelMessage("assistant", "old response")]
    session = PlayerSession(Color.RED, "historical", messages=history, strategic_memory="Old plan")
    for filename in ("catan_v7.yaml", "catan_v10.yaml", "catan_v11.yaml"):
        suite = load_context_suite(default_suite_path().with_name(filename))
        request = ContextAssembler(suite, board_presenter=NoBoardPresenter()).assemble(decision, session)
        assert suite.context.memory_mode == "legacy"
        assert request.messages[1:-1] == tuple(history)
        assert request.messages[0].content == "You are playing a game of Catan. You are playing as RED."
        memory = next(c for c in request.components if c.id == "environment.strategic_memory")
        assert memory.rendered == "YOUR CURRENT GAME PLAN:\nOld plan"
        invalid = suite.model_dump()
        invalid["context"]["order"] = tuple(reversed(invalid["context"]["order"]))
        with pytest.raises(ValueError, match="trajectory must be the first"):
            ContextSuite.model_validate(invalid)
    for filename in ("communication_v4.yaml", "communication_v5.yaml"):
        suite = load_communication_suite(default_communication_suite_path().with_name(filename))
        assert suite.memory_mode == "legacy"
        old = build_communication_request(speech, "historical", suite)
        ignored_new_inputs = build_communication_request(
            speech, "historical", suite, notes="Not rendered", board_presenter=RecordingBoardPresenter(),
        )
        assert old == ignored_new_inputs
        assert old.board_presentation is None
    invalid = load_communication_suite().model_dump()
    invalid["order"] = tuple(reversed(invalid["order"]))
    with pytest.raises(ValueError, match="fixed component order"):
        CommunicationSuite.model_validate(invalid)
