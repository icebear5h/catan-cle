"""Validated communication prompt suite and response parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cle.harness.context import ContextAssembler
from cle.harness.board_surface import BoardPresenter
from cle.harness.catan_board_surface import IndexedTileRowsBoardPresenter
from cle.harness.components import (
    ComponentComposition,
    ComponentDefinition,
    ComponentInputs,
    observation_component_values,
    parse_strict_json_object,
    render_component_definitions,
    validate_component_composition,
)
from cle.harness.response_xml import parse_response_fields
from cle.harness.public_speech import parse_public_speech
from cle.harness.models import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PromptComponent,
)
from cle.harness.suite import (
    MAX_COMPONENT_TEMPLATE_CHARS,
    MAX_SUITE_AUTHORED_CHARS,
    SuiteStatus,
    warn_if_deprecated,
)
from cle.players.contracts import (
    CommitmentProposal,
    CommunicationChoice,
    CommunicationMode,
    PlayerContext,
    TalkContext,
)
from cle.players.notes import validate_notes
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation


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
                ComponentComposition(order=self.order, response=self.response_component),
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
    suite = parse_communication_suite(
        target.read_text(encoding="utf-8"),
        source_name=str(target),
    )
    warn_if_deprecated(suite.status, target)
    return suite


def build_communication_request(
    context: TalkContext,
    session_id: str,
    suite: CommunicationSuite,
    *,
    notes: str = "",
    board_presenter: BoardPresenter | None = None,
) -> ModelRequest:
    components = render_communication_components(context, suite, notes=notes)
    board_presentation = None
    if suite.components and context.observation is not None:
        presenter = board_presenter or IndexedTileRowsBoardPresenter()
        # Presenters consume only these three facts, not a fabricated legal menu.
        board_context = _TalkBoardContext(context.context_id, context.player, context.observation)
        board_presentation = presenter.present(cast(PlayerContext, board_context))
    return ModelRequest(
        decision_id=context.context_id,
        session_id=session_id,
        messages=(
            ModelMessage("system", "\n\n".join(
                component.rendered for component in components
                if component.channel == "system" and component.rendered
            )),
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
        board_presentation=board_presentation,
        trigger_reason=context.trigger_reason,
    )


@dataclass(frozen=True, slots=True)
class _TalkBoardContext:
    context_id: str
    actor: Color
    observation: PlayerObservation


def render_communication_components(
    context: TalkContext,
    suite: CommunicationSuite,
    *,
    notes: str = "",
) -> tuple[PromptComponent, ...]:
    if suite.components:
        if context.observation is not None and context.observation.my_color != context.player:
            raise ValueError("Observation perspective does not match talk player")
        return render_component_definitions(
            suite.components,
            suite.order,
            ComponentInputs(
                color=context.player.value,
                notes=notes,
                max_notes_chars=str(suite.max_notes_chars),
                trigger=(
                    ("Seven: discards are complete; speak before the robber destination is chosen.\n"
                     if context.trigger_reason == "pre_robber" else "")
                    + ContextAssembler._format_events((context.cause,), shared=True)
                ),
                visible_events=ContextAssembler._format_events(context.game_events, shared=True),
                recent_table_talk=ContextAssembler._format_events(context.recent_messages),
                commitments=ContextAssembler._format_commitments(context.active_commitments),
                **observation_component_values(
                    context.observation,
                    include_initial_placement_order=suite.initial_placement_order == "both_rounds",
                ),
            ),
        )
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
    suite: CommunicationSuite | None = None,
) -> CommunicationChoice:
    if suite is not None and suite.reactive_speech:
        try:
            return parse_public_speech(
                parse_strict_json_object(response.content or ""), speaker=speaker,
                participants=participants, max_notes_chars=suite.max_notes_chars,
            )
        except (ValueError, TypeError, RecursionError) as exc:
            return CommunicationChoice(validation_error=f"Invalid communication response: {exc}")
    if suite is not None and suite.memory_mode == "fresh_notes":
        return _parse_fresh_communication_response(
            response, speaker=speaker, participants=participants,
            max_notes_chars=suite.max_notes_chars,
        )
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
        commitment=commitment,
    )


def _parse_fresh_communication_response(
    response: ModelResponse,
    *,
    speaker: Color,
    participants: tuple[Color, ...],
    max_notes_chars: int,
) -> CommunicationChoice:
    try:
        payload = parse_strict_json_object(response.content or "")
        notes_update = (
            validate_notes(payload["notes"], max_chars=max_notes_chars)
            if "notes" in payload else None
        )
        mode = payload.get("mode")
        if mode == "silence":
            if set(payload) - {"mode", "notes"}:
                raise ValueError("silence permits only mode and optional notes")
            return CommunicationChoice(notes_update=notes_update)
        if mode != "say":
            raise ValueError("mode must be silence or say")
        required = {"mode", "text", "audience"}
        # Historical fresh-JSON responses carried an intent label; it is ignored, never required.
        if not required <= payload.keys() or payload.keys() - required - {"notes", "commitment", "intent"}:
            raise ValueError("say requires text, audience and permits optional notes and commitment")
        text = payload["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("say text must be a nonempty string")
        eligible = tuple(color for color in participants if color != speaker)
        names = payload["audience"]
        if names == "PUBLIC":
            audience = eligible
        elif (
            isinstance(names, list) and names
            and all(isinstance(name, str) for name in names)
            and len(names) == len(set(names))
            and set(names) <= {color.value for color in eligible}
        ):
            audience = tuple(color for color in eligible if color.value in names)
        else:
            raise ValueError("audience must be PUBLIC or a nonempty list of distinct eligible colors")
        if not audience:
            raise ValueError("say requires at least one eligible recipient")
        commitment = None
        if "commitment" in payload:
            proposal = payload["commitment"]
            if not isinstance(proposal, dict) or set(proposal) != {"condition", "promise", "expires_turn"}:
                raise ValueError("commitment requires exactly condition, promise, expires_turn")
            if any(not isinstance(proposal[key], str) or not proposal[key].strip() for key in ("condition", "promise")):
                raise ValueError("commitment condition and promise must be nonempty strings")
            expires = proposal["expires_turn"]
            if type(expires) is not int or expires < 0:
                raise ValueError("commitment expires_turn must be a non-negative integer")
            commitment = CommitmentProposal(
                proposal["condition"].strip(), proposal["promise"].strip(), expires,
            )
        return CommunicationChoice(
            mode=CommunicationMode.SAY, text=text.strip(), audience=audience,
            commitment=commitment, notes_update=notes_update,
        )
    except (ValueError, TypeError, RecursionError) as exc:
        return CommunicationChoice(validation_error=f"Invalid communication response: {exc}")


def _render(template: str, values: dict[str, str]) -> str:
    return ContextAssembler._render_template(template, values)
