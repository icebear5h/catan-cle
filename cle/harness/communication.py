"""Validated communication prompt suite and response parsing."""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cle.harness.context import ContextAssembler
from cle.harness.response_xml import parse_response_fields
from cle.harness.models import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PromptComponent,
)
from cle.harness.suite import (
    MAX_COMPONENT_TEMPLATE_CHARS,
    MAX_SUITE_AUTHORED_CHARS,
)
from cle.players.contracts import (
    CommitmentProposal,
    CommunicationChoice,
    CommunicationMode,
    TalkContext,
)
from cle.game_engine.models.player import Color


_NEGOTIATION_INTENTS = frozenset({"TRADE", "QUESTION", "WARNING", "BRIBE"})
_TEMPLATE_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
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
    system_template: str = Field(
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

    @model_validator(mode="after")
    def validate_composition(self) -> "CommunicationSuite":
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
        if set(_TEMPLATE_VARIABLE.findall(self.system_template)) != {"color"}:
            raise ValueError(
                "communication system template must reference only {{ color }}"
            )
        for name, section in self.sections.items():
            unknown = set(_TEMPLATE_VARIABLE.findall(section.template)) - {"value"}
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
    return Path(str(files("cle.harness.suites").joinpath("communication_v5.yaml")))


def parse_communication_suite(
    source: str,
    *,
    source_name: str = "communication suite source",
) -> CommunicationSuite:
    raw = yaml.safe_load(source)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Communication suite {source_name} must contain a YAML mapping"
        )
    return CommunicationSuite.model_validate(raw)


def load_communication_suite(path: str | Path | None = None) -> CommunicationSuite:
    target = Path(path) if path is not None else default_communication_suite_path()
    return parse_communication_suite(
        target.read_text(encoding="utf-8"),
        source_name=str(target),
    )


def build_communication_request(
    context: TalkContext,
    session_id: str,
    suite: CommunicationSuite,
) -> ModelRequest:
    components = render_communication_components(context, suite)
    return ModelRequest(
        decision_id=context.context_id,
        session_id=session_id,
        messages=(
            ModelMessage("system", components[0].rendered),
            ModelMessage(
                "user",
                "\n\n".join(
                    component.rendered
                    for component in components
                    if component.channel == "environment"
                ),
            ),
        ),
        components=components,
    )


def render_communication_components(
    context: TalkContext,
    suite: CommunicationSuite,
) -> tuple[PromptComponent, ...]:
    values = {
        "color": context.player.value,
        "cause": ContextAssembler._format_events((context.cause,)),
        "game_events": ContextAssembler._format_events(context.game_events),
        "recent_messages": ContextAssembler._format_events(context.recent_messages)
        or "None",
        "commitments": "\n".join(
            f"{item.id}: {item.condition} -> {item.promise} "
            f"(expires turn {item.expires_turn})"
            for item in context.active_commitments
        )
        or "None",
    }
    components = [
        PromptComponent(
            id="system.identity",
            channel="system",
            template=suite.system_template,
            value=context.player.value,
            rendered=_render(suite.system_template, values),
            variables=(("color", context.player.value),),
        )
    ]
    if suite.user_template is not None:
        referenced = tuple(
            (name, values[name])
            for name in sorted(set(_TEMPLATE_VARIABLE.findall(suite.user_template)))
        )
        components.append(
            PromptComponent(
                id="environment.communication",
                channel="environment",
                template=suite.user_template,
                value="",
                rendered=_render(suite.user_template, values),
                variables=referenced,
            )
        )
        return tuple(components)

    component_values = {
        "communication_policy": "",
        "trigger": values["cause"],
        "visible_events": values["game_events"] or "None",
        "recent_table_talk": values["recent_messages"],
        "commitments": values["commitments"],
        "response_schema": "",
    }
    for name in suite.order:
        template = suite.sections[name].template
        value = component_values[name]
        referenced = set(_TEMPLATE_VARIABLE.findall(template))
        variables = (("value", value),) if "value" in referenced else ()
        components.append(
            PromptComponent(
                id=f"environment.{name}",
                channel="environment",
                template=template,
                value=value,
                rendered=_render(template, {"value": value}),
                variables=variables,
            )
        )
    return tuple(components)


def parse_communication_response(
    response: ModelResponse,
    *,
    speaker: Color,
    participants: tuple[Color, ...],
    instruction: str = "",
) -> CommunicationChoice:
    text = response.content or ""
    if instruction and text.lstrip().startswith(instruction):
        text = text.lstrip()[len(instruction):]
    try:
        fields, _ = parse_response_fields(text, instruction=instruction)
    except ValueError:
        return CommunicationChoice()
    if any(len(values) != 1 for values in fields.values()):
        return CommunicationChoice()
    text = fields.get("message", [""])[0]
    if not text or text.strip().upper() == "SILENCE":
        return CommunicationChoice()

    intent = fields.get("intent", [""])[0].upper()
    if intent not in _NEGOTIATION_INTENTS:
        return CommunicationChoice()

    audience_text = fields.get("audience", ["PUBLIC"])[0].upper()
    if audience_text == "PUBLIC":
        audience = tuple(color for color in participants if color != speaker)
    else:
        names = {name.strip() for name in audience_text.split(",")}
        eligible_names = {color.value for color in participants if color != speaker}
        if not names.issubset(eligible_names):
            return CommunicationChoice()
        audience = tuple(
            color
            for color in participants
            if color != speaker and color.value in names
        )

    commitment = None
    condition = fields.get("commitment_condition", [""])[0]
    promise = fields.get("commitment_promise", [""])[0]
    expires = fields.get("commitment_expires_turn", [""])[0]
    if any(name.startswith("commitment_") for name in fields):
        if not condition or not promise or not re.fullmatch(r"[0-9]+", expires):
            return CommunicationChoice()
        try:
            expires_turn = int(expires)
        except ValueError:
            return CommunicationChoice()
        commitment = CommitmentProposal(condition, promise, expires_turn)

    return CommunicationChoice(
        mode=CommunicationMode.SAY,
        text=text,
        audience=audience,
        intent=intent,
        commitment=commitment,
    )


def _render(template: str, values: dict[str, str]) -> str:
    return ContextAssembler._render_template(template, values)
