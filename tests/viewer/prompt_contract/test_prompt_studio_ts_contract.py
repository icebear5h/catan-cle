"""The Prompt Studio's TypeScript types must match what the Python API serializes."""

import re
from pathlib import Path

from cle.harness.prompt_store import ActivePromptSuites, validate_shared_prompt_source
from cle.harness.shared_suite import default_shared_suite_path
from playground.game_viewer.routes.prompt_studio.editor import editor_payload, suite_metadata
from scripts.gen_prompt_studio_types import OUTPUT, render

PROMPTS = Path(__file__).resolve().parents[3] / "playground/frontend/src/components/prompts"


def interface_members(source: str, name: str) -> dict[str, str]:
    """Map `key: type;` members of one flat exported interface in a .ts source."""
    match = re.search(rf"export interface {name} \{{\n(.*?)\n\}}", source, re.S)
    assert match, f"interface {name} not found"
    members = [line.strip().rstrip(";") for line in match.group(1).splitlines()]
    return dict(member.split(": ", 1) for member in members if member)


def default_active() -> ActivePromptSuites:
    source = default_shared_suite_path().read_text(encoding="utf-8")
    return validate_shared_prompt_source(source, overridden=False)


def test_generated_shared_suite_types_are_current() -> None:
    assert OUTPUT.read_text(encoding="utf-8") == render(), (
        "Stale TS types: uv run --no-sync python -m scripts.gen_prompt_studio_types"
    )


def test_generated_document_keys_match_the_editor_payload() -> None:
    shared = editor_payload(default_active())["shared"]
    assert isinstance(shared, dict)
    document = shared["document"]
    generated = OUTPUT.read_text(encoding="utf-8")
    assert list(interface_members(generated, "SharedPromptDocument")) == list(document)
    for component in document["components"].values():
        assert list(interface_members(generated, "SharedComponentDefinition")) == list(component)


def test_suite_metadata_matches_the_serialized_metadata() -> None:
    shared = default_active().shared
    assert shared is not None
    metadata = suite_metadata(shared)
    members = interface_members(
        (PROMPTS / "promptStudioTypes.ts").read_text(encoding="utf-8"), "SuiteMetadata",
    )
    scalar = {str: "string", bool: "boolean", int: "number"}
    expected = {
        key: "SuiteStatus" if key == "status" else scalar[type(value)]
        for key, value in metadata.items()
    }
    assert members == expected
