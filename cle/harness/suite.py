"""Strict loader for separately editable prompt and context suites."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SystemConfig(_StrictModel):
    template: str = Field(min_length=1)


class TrajectoryConfig(_StrictModel):
    mode: Literal["full"] = "full"
    max_messages: int | None = Field(default=None, ge=2)


class ContextConfig(_StrictModel):
    order: tuple[str, ...]
    trajectory: TrajectoryConfig = TrajectoryConfig()


class SectionConfig(_StrictModel):
    heading: str = ""
    empty: Literal["omit", "include"] = "omit"
    empty_text: str = ""


class ResponseConfig(_StrictModel):
    format: Literal["xml"] = "xml"
    tags: tuple[str, ...]
    instruction: str = Field(min_length=1)
    fallback: Literal["first_legal"] = "first_legal"


class ContextSuite(_StrictModel):
    """Versioned authored content plus bounded context-layout policy."""

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    system: SystemConfig
    context: ContextConfig
    sections: dict[str, SectionConfig]
    phase_guidance: dict[str, str]
    response: ResponseConfig

    @model_validator(mode="after")
    def validate_layout(self) -> "ContextSuite":
        order = self.context.order
        if len(order) != len(set(order)):
            raise ValueError("context.order contains duplicate sections")
        if not order or order[0] != "trajectory":
            raise ValueError("trajectory must be the first context section")

        unknown = set(order) - {"trajectory", *self.sections.keys()}
        if unknown:
            raise ValueError(f"context.order references unknown sections: {sorted(unknown)}")

        required_tags = {"game_plan", "rationale", "action"}
        missing_tags = required_tags - set(self.response.tags)
        if missing_tags:
            raise ValueError(f"response.tags is missing required tags: {sorted(missing_tags)}")
        return self


def default_suite_path() -> Path:
    """Return the built-in text-only Catan suite path."""
    return Path(__file__).resolve().parent / "suites" / "catan_v5.yaml"


def load_context_suite(path: str | Path | None = None) -> ContextSuite:
    """Load and validate one YAML suite without caching edited content."""
    suite_path = Path(path) if path is not None else default_suite_path()
    with suite_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Context suite {suite_path} must contain a YAML mapping")
    return ContextSuite.model_validate(data)
