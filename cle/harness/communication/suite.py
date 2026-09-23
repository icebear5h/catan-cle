"""Validated communication prompt suite loading and authored-size limits."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cle.harness.components import (
    MAX_COMPONENT_TEMPLATE_CHARS,
    MAX_SUITE_AUTHORED_CHARS,
    TEMPLATE_VARIABLE,
    ComponentComposition,
    ComponentDefinition,
    SuiteStatus,
    validate_component_composition,
)
from cle.harness.suite import warn_if_deprecated
from cle.harness.yaml_source import BUILTIN_SUITES_DIR, load_yaml_mapping

_COMMUNICATION_COMPONENT_ORDER = (
    "communication_policy",
    "trigger",
    "visible_events",
    "recent_table_talk",
    "commitments",
    "response_schema",
)


class CommunicationSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template: str = Field(min_length=1, max_length=MAX_COMPONENT_TEMPLATE_CHARS)


class CommunicationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    version: int
    status: SuiteStatus = "active"
    system_template: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_COMPONENT_TEMPLATE_CHARS,
    )
    user_template: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_COMPONENT_TEMPLATE_CHARS,
    )
    order: tuple[str, ...] = ()
    sections: dict[str, CommunicationSection] = Field(default_factory=dict)
    memory_mode: Literal["legacy", "fresh_notes"] = "legacy"
    max_notes_chars: int = Field(default=4000, ge=1, le=4000, strict=True)
    format: Literal["xml", "json"] = "xml"
    components: dict[str, ComponentDefinition] = Field(default_factory=dict)
    response_component: str | None = None
    initial_placement_order: Literal["omit", "both_rounds"] = "omit"
    reactive_speech: bool = False

    @model_validator(mode="after")
    def validate_composition(self) -> "CommunicationSuite":
        if self.reactive_speech and self.memory_mode != "fresh_notes":
            raise ValueError("Reactive speech requires fresh notes")
        if self.components:
            if self.memory_mode != "fresh_notes" or self.format != "json":
                raise ValueError("shared communication requires fresh_notes and JSON responses")
            if self.sections or self.user_template is not None or self.system_template is not None:
                raise ValueError("shared communication cannot contain historical templates or sections")
            validate_component_composition(
                self.components,
                ComponentComposition(
                    order=self.order, response=cast("str", self.response_component),
                ),
                consumer="speech",
            )
            return self
        if self.response_component is not None:
            raise ValueError("response_component requires shared components")
        if self.system_template is None:
            raise ValueError("historical communication requires system_template")
        if (self.memory_mode == "fresh_notes") != (self.format == "json"):
            raise ValueError("fresh_notes requires JSON responses; legacy requires XML")
        if self.user_template is not None:
            if self.order or self.sections:
                raise ValueError(
                    "legacy user_template cannot be combined with component sections"
                )
            return self

        if self.order != _COMMUNICATION_COMPONENT_ORDER:
            raise ValueError(
                "communication order must equal the fixed component order"
            )
        if set(self.sections) != set(_COMMUNICATION_COMPONENT_ORDER):
            raise ValueError(
                "communication sections must exactly match the fixed component order"
            )
        if set(TEMPLATE_VARIABLE.findall(self.system_template)) != {"color"}:
            raise ValueError(
                "communication system template must reference only {{ color }}"
            )
        for name, section in self.sections.items():
            unknown = set(TEMPLATE_VARIABLE.findall(section.template)) - {"value"}
            if unknown:
                raise ValueError(
                    f"communication section {name!r} has unknown variables: "
                    f"{sorted(unknown)}"
                )
        authored_size = len(self.system_template) + sum(
            len(section.template) for section in self.sections.values()
        )
        if authored_size > MAX_SUITE_AUTHORED_CHARS:
            raise ValueError(
                f"suite authored text may not exceed {MAX_SUITE_AUTHORED_CHARS} characters"
            )
        return self


def default_communication_suite_path() -> Path:
    return BUILTIN_SUITES_DIR / "communication_v5.yaml"


def parse_communication_suite(
    source: str,
    *,
    source_name: str = "communication suite source",
) -> CommunicationSuite:
    raw = load_yaml_mapping(source, source=f"communication suite {source_name}")
    return CommunicationSuite.model_validate(raw)


def load_communication_suite(path: str | Path | None = None) -> CommunicationSuite:
    target = Path(path) if path is not None else default_communication_suite_path()
    suite = parse_communication_suite(
        target.read_text(encoding="utf-8"),
        source_name=str(target),
    )
    warn_if_deprecated(suite.status, target)
    return suite