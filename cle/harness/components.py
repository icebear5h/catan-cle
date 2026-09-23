"""Typed shared prompt definitions, authoritative inputs, and one-pass rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cle.env.observation_formatter import CatanObservationFormatter
from cle.game_engine.board_tokens import node_token
from cle.game_engine.observation import PlayerObservation
from cle.harness.models import PromptComponent

# Lifecycle marker for authored suite files; only one bundle is active at a time.
SuiteStatus = Literal["active", "legacy", "deprecated"]
MAX_COMPONENT_TEMPLATE_CHARS = 12_000
MAX_SUITE_AUTHORED_CHARS = 60_000
TEMPLATE_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
ComponentInput = Literal[
    "color", "notes", "max_notes_chars", "visible_events", "recent_table_talk",
    "commitments", "phase_info", "board_state", "resources", "opponents",
    "trade_window", "phase_guidance", "legal_actions", "decision_request", "trigger",
]


class ComponentInputs(BaseModel):
    """Already perspective-filtered strings; never templates or executable data."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    color: str = ""
    notes: str = ""
    max_notes_chars: str = "4000"
    visible_events: str = ""
    recent_table_talk: str = ""
    commitments: str = ""
    phase_info: str = ""
    board_state: str = ""
    resources: str = ""
    opponents: str = ""
    trade_window: str = ""
    phase_guidance: str = ""
    legal_actions: str = ""
    decision_request: str = ""
    trigger: str = ""


class ComponentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: Literal["system", "environment"]
    inputs: tuple[ComponentInput, ...] = ()
    template: str = Field(min_length=1, max_length=MAX_COMPONENT_TEMPLATE_CHARS)
    empty: Literal["omit", "include"] = "include"
    empty_text: str = Field(default="", max_length=MAX_COMPONENT_TEMPLATE_CHARS)

    @model_validator(mode="after")
    def validate_inputs(self) -> "ComponentDefinition":
        if not self.template.strip():
            raise ValueError("component template must not be blank")
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError("component inputs contain duplicates")
        referenced = set(TEMPLATE_VARIABLE.findall(self.template))
        if referenced != set(self.inputs):
            raise ValueError("template variables must exactly match declared inputs")
        remainder = TEMPLATE_VARIABLE.sub("", self.template)
        if "{{" in remainder:
            raise ValueError("unsupported template variable syntax")
        return self


class ComponentComposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    order: tuple[str, ...] = Field(min_length=1)
    response: str = Field(min_length=1)


def validate_component_composition(
    components: Mapping[str, ComponentDefinition],
    composition: ComponentComposition,
    *,
    consumer: Literal["decision", "speech"],
) -> None:
    """Check reference closure and functional membership, never positional layout."""
    order = composition.order
    if len(order) != len(set(order)):
        raise ValueError("composition order contains duplicate references")
    unknown = set(order) - components.keys()
    if unknown:
        raise ValueError(f"composition references unknown components: {sorted(unknown)}")
    if composition.response not in order:
        raise ValueError("composition response must reference a component in its order")
    response = components[composition.response]
    if set(response.inputs) - {"max_notes_chars"} or response.empty != "include":
        raise ValueError("response component must be unconditional and use only max_notes_chars")
    required = {
        "color", "notes", "visible_events", "recent_table_talk", "commitments",
        "phase_info", "board_state", "resources", "opponents",
    }
    decision_inputs = {"phase_guidance", "legal_actions", "decision_request"}
    required |= decision_inputs if consumer == "decision" else {"trigger"}
    available = set(ComponentInputs.model_fields) - (
        {"trigger"} if consumer == "decision" else decision_inputs
    )
    used = {name for ref in order for name in components[ref].inputs}
    if used - available:
        raise ValueError(f"{consumer} composition has unavailable inputs: {sorted(used - available)}")
    if required - used:
        raise ValueError(f"{consumer} composition is missing required inputs: {sorted(required - used)}")
    authored_size = sum(len(item.template) + len(item.empty_text) for item in components.values())
    if authored_size > MAX_SUITE_AUTHORED_CHARS:
        raise ValueError(f"suite authored text may not exceed {MAX_SUITE_AUTHORED_CHARS} characters")


def render_template(template: str, values: Mapping[str, str]) -> str:
    missing = set(TEMPLATE_VARIABLE.findall(template)) - values.keys()
    if missing:
        raise ValueError(f"Template variables have no values: {sorted(missing)}")
    # A substitution result is data. In particular, notes cannot expand variables.
    return TEMPLATE_VARIABLE.sub(lambda match: values[match.group(1)], template).strip()


def render_component_definitions(
    components: Mapping[str, ComponentDefinition],
    order: tuple[str, ...],
    inputs: ComponentInputs,
) -> tuple[PromptComponent, ...]:
    rendered = []
    for name in order:
        definition = components[name]
        values: dict[str, str] = {key: getattr(inputs, key) for key in definition.inputs}
        if values and not any(values.values()) and definition.empty == "omit":
            continue
        variables = tuple(sorted(values.items()))
        substitutions = {
            key: value or definition.empty_text for key, value in values.items()
        }
        rendered.append(PromptComponent(
            id=f"{definition.channel}.{name}",
            channel=definition.channel,
            template=definition.template,
            value="\n".join(values.values()),
            rendered=render_template(definition.template, substitutions),
            variables=variables,
        ))
    return tuple(rendered)


def observation_component_values(
    observation: PlayerObservation | None,
    *,
    include_initial_placement_order: bool,
) -> dict[str, str]:
    if observation is None:
        return {}
    formatted = CatanObservationFormatter().format(
        observation,
        include_legal_actions=False,
        include_initial_placement_order=include_initial_placement_order,
        shared=True,
    )
    phase = formatted.strategic_context
    phase += f"\nTurn player: {observation.turn_player_color.value}; pending: {observation.current_prompt}"
    if observation.current_phase == "initial_placement":
        actor = observation.turn_player_color
        settlements = observation.my_settlements if actor == observation.my_color else observation.opponent_settlements[actor]
        road = observation.setup_road_anchor is not None
        ordinal = len(settlements) if road else len(settlements) + 1
        label = "first" if ordinal == 1 else "second"
        phase += f"\nSetup: {actor.value} must place the {label} {'road' if road else 'settlement'}."
        if road:
            phase += f" The {label} settlement is already placed at {node_token(cast('int', observation.setup_road_anchor))}; attach this road to it."
        phase += " First settlement gives no starting cards; starting hand comes only from the second settlement (one per adjacent non-desert tile). The second settlement is independently placed, not connected to the first road."
    elif observation.free_roads_available:
        phase += f"\nRoad Building: {observation.free_roads_available} free road placements remaining; use build_road, not setup placement or a paid road."
    if observation.current_phase == "discarding":
        phase += f"\nRequired discard: {sum(observation.my_resources.values()) // 2} cards if this is your discard decision."
    return {
        "phase_info": phase,
        "board_state": formatted.board_state,
        "resources": formatted.resources,
        "opponents": formatted.opponents,
        "trade_window": formatted.trade_context,
    }


def parse_strict_json_object(text: str) -> dict[str, object]:
    """Bounded JSON envelopes shared by the fresh action and speech contracts."""
    if len(text) > 128 * 1024:
        raise ValueError("Response exceeds 131072 characters.")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")

    payload = json.loads(
        text, object_pairs_hook=unique_object, parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("Return exactly one JSON object.")
    return payload
