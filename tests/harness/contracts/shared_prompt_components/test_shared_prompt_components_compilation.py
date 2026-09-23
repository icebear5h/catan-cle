"""Compiled suite strictness and variable validation."""
from typing import Any

import pytest

from cle.harness.communication import (
    CommunicationSuite,
)
from cle.harness.components import (
    ComponentDefinition,
    ComponentInputs,
    render_component_definitions,
)
from cle.harness.shared_suite import (
    SharedPromptSuite,
    default_shared_suite_path,
    load_shared_prompt_suite,
    parse_shared_prompt_suite,
)
from cle.harness.suite import ContextSuite


@pytest.mark.parametrize("consumer", ["decision", "speech"])
def test_compiled_shared_suites_reject_ignored_legacy_fields(consumer: str) -> None:
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


def test_compiled_decision_response_cannot_diverge_from_authored_component() -> None:
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
def test_unknown_variables_bad_types_and_unsupported_syntax_rejected(
    definition: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        ComponentDefinition.model_validate({"channel": "environment", **definition})


def test_speech_cannot_reference_decision_only_inputs() -> None:
    raw = load_shared_prompt_suite().model_dump()
    raw["compositions"]["speech"]["order"] += ("legal_actions",)
    with pytest.raises(ValueError, match="unavailable inputs"):
        SharedPromptSuite.model_validate(raw)


def test_duplicate_yaml_keys_and_external_sources_rejected() -> None:
    source = default_shared_suite_path().read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        parse_shared_prompt_suite(source + "\nversion: 2\n")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        parse_shared_prompt_suite(source.replace("inputs: [notes]", "inputs: [notes]\n    inputs: [color]"))
    with pytest.raises(ValueError):
        parse_shared_prompt_suite(source + "\ninclude: historical.yaml\n")


def test_static_components_survive_empty_dynamic_inputs() -> None:
    components = {
        "static": ComponentDefinition(channel="environment", template="Always render.", empty="omit"),
        "dynamic": ComponentDefinition(
            channel="environment", inputs=("notes",), template="{{ notes }}", empty="omit",
        ),
    }
    result = render_component_definitions(components, ("dynamic", "static"), ComponentInputs())
    assert [c.id for c in result] == ["environment.static"]
    assert result[0].rendered == "Always render."
