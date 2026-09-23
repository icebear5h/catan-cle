"""Bundle self-containment, ordering, and reference checks."""

from pathlib import Path

import pytest
import yaml

from cle.game_engine.models.player import Color
from cle.harness.communication import (
    CommunicationSuite,
    build_communication_request,
)
from cle.harness.context import ContextAssembler
from cle.harness.models import PlayerSession
from cle.harness.shared_suite import (
    SharedPromptSuite,
    default_shared_suite_path,
    load_shared_prompt_suite,
    parse_shared_prompt_suite,
)
from cle.harness.suite import ContextSuite
from cle.players.contracts import PlayerContext, TalkContext


def test_bundle_is_self_contained_and_compiles_typed_consumers(tmp_path: Path) -> None:
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


def test_one_authored_edit_changes_both_consumers(contexts: tuple[PlayerContext, TalkContext]) -> None:
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


def test_independent_order_and_multiple_system_components(contexts: tuple[PlayerContext, TalkContext]) -> None:
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
def test_unknown_duplicate_or_missing_references_are_rejected(consumer: str) -> None:
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


def test_components_can_be_combined_and_renamed_without_count_or_name_invariants(contexts: tuple[PlayerContext, TalkContext]) -> None:
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
