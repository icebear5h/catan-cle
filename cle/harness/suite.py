"""Strict loader for separately editable prompt and context suites."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cle.harness.components import (
    MAX_COMPONENT_TEMPLATE_CHARS,
    MAX_SUITE_AUTHORED_CHARS,
    ComponentComposition,
    SuiteStatus,
    ComponentDefinition,
    render_template,
    validate_component_composition,
)


_TEMPLATE_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
_COMPONENT_ORDER = (
    "trajectory",
    "strategic_memory",
    "visible_events",
    "phase_info",
    "board_state",
    "resources",
    "opponents",
    "trade_window",
    "phase_guidance",
    "legal_actions",
    "decision_request",
    "response_schema",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SystemConfig(_StrictModel):
    template: str = Field(min_length=1, max_length=MAX_COMPONENT_TEMPLATE_CHARS)


class TrajectoryConfig(_StrictModel):
    mode: Literal["full"] = "full"
    max_messages: int | None = Field(default=None, ge=2)


class ContextConfig(_StrictModel):
    order: tuple[str, ...]
    mode: Literal["legacy", "components", "shared"] = "legacy"
    memory_mode: Literal["legacy", "fresh_notes"] = "legacy"
    max_notes_chars: int = Field(default=4000, ge=1, le=4000, strict=True)
    initial_placement_order: Literal["omit", "both_rounds"] = "omit"
    trajectory: TrajectoryConfig = TrajectoryConfig()
    social_context: bool = False
    reactive_speech: bool = False
    deterministic_batches: bool = False


class SectionConfig(_StrictModel):
    template: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_COMPONENT_TEMPLATE_CHARS,
    )
    heading: str = ""
    empty: Literal["omit", "include"] = "omit"
    empty_text: str = ""


class ResponseConfig(_StrictModel):
    format: Literal["xml", "json"] = "xml"
    tags: tuple[str, ...]
    instruction: str = Field(min_length=1, max_length=MAX_COMPONENT_TEMPLATE_CHARS)
    fallback: Literal["first_legal"] = "first_legal"


class ContextSuite(_StrictModel):
    """Versioned authored content plus bounded context-layout policy."""

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: SuiteStatus = "active"
    system: SystemConfig | None = None
    context: ContextConfig
    sections: dict[str, SectionConfig]
    phase_guidance: dict[str, str]
    response: ResponseConfig
    components: dict[str, ComponentDefinition] = Field(default_factory=dict)
    response_component: str | None = None

    @model_validator(mode="after")
    def validate_layout(self) -> "ContextSuite":
        if self.context.deterministic_batches and self.context.mode != "shared":
            raise ValueError("Deterministic batches require the shared/fresh contract")
        if self.context.reactive_speech and self.context.mode != "shared":
            raise ValueError("Reactive speech requires the shared contract")
        if self.context.mode == "shared":
            if self.context.memory_mode != "fresh_notes" or self.response.format != "json":
                raise ValueError("shared context requires fresh_notes and JSON responses")
            if self.sections or self.system is not None:
                raise ValueError("shared context cannot contain historical system or sections")
            validate_component_composition(
                self.components,
                ComponentComposition(order=self.context.order, response=self.response_component),
                consumer="decision",
            )
            if set(self.response.tags) != {"tool", "arguments", "notes"}:
                raise ValueError("fresh response.tags must contain tool, arguments, notes")
            instruction = render_template(
                self.components[self.response_component].template,
                {"max_notes_chars": str(self.context.max_notes_chars)},
            )
            if self.response.instruction != instruction:
                raise ValueError("shared response instruction must match its component")
            authored = [
                *(item.template for item in self.components.values()),
                *(item.empty_text for item in self.components.values()),
                *self.phase_guidance.values(),
            ]
            if any(len(value) > MAX_COMPONENT_TEMPLATE_CHARS for value in authored):
                raise ValueError(f"suite strings may not exceed {MAX_COMPONENT_TEMPLATE_CHARS} characters")
            if sum(map(len, authored)) > MAX_SUITE_AUTHORED_CHARS:
                raise ValueError(f"suite authored text may not exceed {MAX_SUITE_AUTHORED_CHARS} characters")
            return self
        if self.components or self.response_component is not None:
            raise ValueError("shared definitions require shared context mode")
        if self.system is None:
            raise ValueError("historical suites require system.template")
        order = self.context.order
        if len(order) != len(set(order)):
            raise ValueError("context.order contains duplicate sections")
        if not order or order[0] != "trajectory":
            raise ValueError("trajectory must be the first context section")

        unknown = set(order) - {"trajectory", *self.sections.keys()}
        if unknown:
            raise ValueError(f"context.order references unknown sections: {sorted(unknown)}")

        if self.context.social_context and self.context.mode != "components":
            raise ValueError("social_context requires component mode")
        if self.context.mode == "components":
            component_order = _COMPONENT_ORDER
            if self.context.social_context:
                component_order = (
                    *_COMPONENT_ORDER[:3],
                    "recent_table_talk",
                    "commitments",
                    *_COMPONENT_ORDER[3:],
                )
            if order != component_order:
                raise ValueError(
                    "component context.order must equal the fixed component order"
                )
            if set(self.sections) != set(component_order[1:]):
                raise ValueError(
                    "component sections must exactly match the fixed component order"
                )
            system_variables = set(_TEMPLATE_VARIABLE.findall(self.system.template))
            if system_variables != {"color"}:
                raise ValueError(
                    "component system template must reference only {{ color }}"
                )
            for name, section in self.sections.items():
                if section.template is None:
                    raise ValueError(
                        f"component section {name!r} must define a template"
                    )
                unknown_variables = set(
                    _TEMPLATE_VARIABLE.findall(section.template)
                ) - {"value"}
                if unknown_variables:
                    raise ValueError(
                        f"component section {name!r} has unknown variables: "
                        f"{sorted(unknown_variables)}"
                    )

        if self.context.memory_mode == "fresh_notes" and self.response.format != "json":
            raise ValueError("fresh_notes requires JSON responses")
        if self.context.memory_mode == "fresh_notes":
            required_tags = {"tool", "arguments"}
        else:
            required_tags = (
                {"game_plan", "tool", "arguments"}
                if self.response.format == "json"
                else {"game_plan", "action"}
            )
        missing_tags = required_tags - set(self.response.tags)
        if missing_tags:
            raise ValueError(f"response.tags is missing required tags: {sorted(missing_tags)}")

        authored_strings = [
            self.system.template,
            self.response.instruction,
            *self.phase_guidance.values(),
            *(section.template or section.heading for section in self.sections.values()),
            *(section.empty_text for section in self.sections.values()),
        ]
        oversized = [
            value for value in authored_strings if len(value) > MAX_COMPONENT_TEMPLATE_CHARS
        ]
        if oversized:
            raise ValueError(
                f"suite strings may not exceed {MAX_COMPONENT_TEMPLATE_CHARS} characters"
            )
        if sum(len(value) for value in authored_strings) > MAX_SUITE_AUTHORED_CHARS:
            raise ValueError(
                f"suite authored text may not exceed {MAX_SUITE_AUTHORED_CHARS} characters"
            )
        return self


def default_suite_path() -> Path:
    """Return the built-in text-only Catan suite path."""
    return Path(__file__).resolve().parent / "suites" / "catan_v11.yaml"


def parse_context_suite(
    source: str,
    *,
    source_name: str = "context suite source",
) -> ContextSuite:
    """Parse and strictly validate one context suite from trusted text."""
    data = yaml.safe_load(source)
    if not isinstance(data, dict):
        raise ValueError(f"Context suite {source_name} must contain a YAML mapping")
    return ContextSuite.model_validate(data)


def load_context_suite(path: str | Path | None = None) -> ContextSuite:
    """Load and validate one YAML suite without caching edited content."""
    suite_path = Path(path) if path is not None else default_suite_path()
    suite = parse_context_suite(
        suite_path.read_text(encoding="utf-8"),
        source_name=str(suite_path),
    )
    warn_if_deprecated(suite.status, suite_path)
    return suite


def warn_if_deprecated(status: SuiteStatus, source: Path) -> None:
    """Loading a deprecated suite file is allowed for replay, but never silent."""
    if status == "deprecated":
        warnings.warn(f"Prompt suite {source} is deprecated", DeprecationWarning, stacklevel=3)
