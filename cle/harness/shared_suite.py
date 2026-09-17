"""One self-contained authored bundle compiled into decision and speech suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cle.harness.communication import CommunicationSuite
from cle.harness.components import (
    SuiteStatus,
    MAX_COMPONENT_TEMPLATE_CHARS,
    MAX_SUITE_AUTHORED_CHARS,
    ComponentComposition,
    ComponentDefinition,
    render_template,
    validate_component_composition,
)
from cle.harness.suite import ContextConfig, ContextSuite, ResponseConfig, warn_if_deprecated


class SharedCompositions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ComponentComposition
    speech: ComponentComposition


class SharedPromptSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: int = Field(ge=1, strict=True)
    status: SuiteStatus = "active"
    memory_mode: Literal["fresh_notes"] = "fresh_notes"
    max_notes_chars: int = Field(default=4000, ge=1, le=4000, strict=True)
    initial_placement_order: Literal["omit", "both_rounds"] = "both_rounds"
    reactive_speech: bool = False
    deterministic_batches: bool = False
    components: dict[str, ComponentDefinition]
    compositions: SharedCompositions
    phase_guidance: dict[str, str]

    @model_validator(mode="after")
    def validate_bundle(self) -> "SharedPromptSuite":
        required_phases = {
            "initial_settlement_1", "initial_settlement_2", "initial_road_1", "initial_road_2",
            "initial_placement", "discarding", "robber", "main_game",
        }
        if missing := required_phases - self.phase_guidance.keys():
            raise ValueError(f"phase_guidance is missing required routing keys: {sorted(missing)}")
        if any(not text.strip() for text in self.phase_guidance.values()):
            raise ValueError("phase_guidance values must be nonempty strings")
        if any(not name.strip() for name in self.components):
            raise ValueError("component names must be nonempty")
        for consumer in ("decision", "speech"):
            validate_component_composition(
                self.components, getattr(self.compositions, consumer), consumer=consumer,
            )
        authored = [
            *(component.template for component in self.components.values()),
            *(component.empty_text for component in self.components.values()),
            *self.phase_guidance.values(),
        ]
        if any(len(value) > MAX_COMPONENT_TEMPLATE_CHARS for value in authored):
            raise ValueError(f"suite strings may not exceed {MAX_COMPONENT_TEMPLATE_CHARS} characters")
        if sum(map(len, authored)) > MAX_SUITE_AUTHORED_CHARS:
            raise ValueError(f"suite authored text may not exceed {MAX_SUITE_AUTHORED_CHARS} characters")
        return self

    def decision_suite(self) -> ContextSuite:
        composition = self.compositions.decision
        return ContextSuite(
            id=self.id,
            version=str(self.version),
            status=self.status,
            context=ContextConfig(
                mode="shared", order=composition.order, memory_mode=self.memory_mode,
                max_notes_chars=self.max_notes_chars,
                initial_placement_order=self.initial_placement_order,
                social_context=True,
                reactive_speech=self.reactive_speech,
                deterministic_batches=self.deterministic_batches,
            ),
            sections={},
            components={name: self.components[name] for name in composition.order},
            response_component=composition.response,
            phase_guidance=self.phase_guidance,
            response=ResponseConfig(
                format="json", tags=("tool", "arguments", "notes"),
                instruction=render_template(
                    self.components[composition.response].template,
                    {"max_notes_chars": str(self.max_notes_chars)},
                ),
            ),
        )

    def communication_suite(self) -> CommunicationSuite:
        composition = self.compositions.speech
        return CommunicationSuite(
            id=self.id, version=self.version, status=self.status, order=composition.order,
            memory_mode=self.memory_mode, max_notes_chars=self.max_notes_chars,
            format="json", initial_placement_order=self.initial_placement_order,
            reactive_speech=self.reactive_speech,
            components={name: self.components[name] for name in composition.order},
            response_component=composition.response,
        )


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in result:
                    raise ValueError(f"Duplicate YAML key: {key}")
                result[key] = self.construct_object(value_node, deep=deep)
            except TypeError as exc:
                raise ValueError("YAML mapping keys must be scalar values") from exc
        return result


def default_shared_suite_path() -> Path:
    return Path(__file__).resolve().parent / "suites" / "shared_v1.yaml"


def parse_shared_prompt_suite(
    source: str,
    *,
    source_name: str = "shared prompt suite source",
) -> SharedPromptSuite:
    """Parse only local authored definitions; no includes, imports, or resolution I/O."""
    try:
        data = yaml.load(source, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError(f"Invalid shared prompt suite {source_name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Shared prompt suite {source_name} must contain a YAML mapping")
    return SharedPromptSuite.model_validate(data)


def load_shared_prompt_suite(path: str | Path | None = None) -> SharedPromptSuite:
    target = Path(path) if path is not None else default_shared_suite_path()
    suite = parse_shared_prompt_suite(target.read_text(encoding="utf-8"), source_name=str(target))
    warn_if_deprecated(suite.status, target)
    return suite
