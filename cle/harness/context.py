"""Deterministic context assembly and typed policy-response parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from cle.env.observation_formatter import CatanObservationFormatter
from cle.harness.board_surface import BoardPresenter
from cle.harness.catan_board_surface import IndexedTileRowsBoardPresenter
from cle.harness.models import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PlayerSession,
    PromptComponent,
)
from cle.harness.suite import ContextSuite
from cle.harness.response_xml import parse_response_fields
from cle.players.contracts import PlayerChoice, PlayerContext
from cle.game_engine.events import PlayerEvent
from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES, TradeOffer


_TEMPLATE_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


class ContextAssembler:
    """Render one complete active model context from typed authoritative inputs."""

    def __init__(
        self,
        suite: ContextSuite,
        *,
        board_presenter: BoardPresenter | None = None,
    ) -> None:
        self.suite = suite
        self.board_presenter = board_presenter or IndexedTileRowsBoardPresenter()

    def assemble(
        self,
        context: PlayerContext,
        session: PlayerSession,
        feedback: str | None = None,
    ) -> ModelRequest:
        components = self.render_components(context, session, feedback)
        board_presentation = self.board_presenter.present(context)
        system = components[0].rendered
        environment = "\n\n".join(
            component.rendered
            for component in components
            if component.channel == "environment" and component.rendered
        )
        return ModelRequest(
            decision_id=context.context_id,
            session_id=session.session_id,
            messages=(
                ModelMessage(role="system", content=system),
                *self._history(session),
                ModelMessage(role="user", content=environment),
            ),
            components=components,
            board_presentation=board_presentation,
        )

    def render_components(
        self,
        context: PlayerContext,
        session: PlayerSession,
        feedback: str | None = None,
    ) -> tuple[PromptComponent, ...]:
        """Render typed system/environment strings from one private perspective."""
        if context.actor != session.color:
            raise ValueError(
                f"Context actor {context.actor} does not match player {session.color}"
            )

        guidance = self.suite.phase_guidance.get(context.prompt_key)
        # Historical suites used one shared setup-road key. Restored games
        # retain their recorded source while current routing uses road ordinals.
        if guidance is None and context.prompt_key in {
            "initial_road_1",
            "initial_road_2",
        }:
            guidance = self.suite.phase_guidance.get("initial_road")
        if guidance is None:
            raise ValueError(
                f"Suite {self.suite.id}@{self.suite.version} has no guidance "
                f"for {context.prompt_key!r}"
            )

        system_values = {
            "color": self._color_name(session.color),
            "phase_guidance": guidance,
            "response_instruction": self.suite.response.instruction,
        }
        referenced_system_values = tuple(
            (name, system_values[name])
            for name in sorted(set(_TEMPLATE_VARIABLE.findall(self.suite.system.template)))
        )
        components = [
            PromptComponent(
                id="system.identity",
                channel="system",
                template=self.suite.system.template,
                value=system_values["color"],
                rendered=self._render_template(
                    self.suite.system.template,
                    system_values,
                ),
                variables=referenced_system_values,
            )
        ]
        content = self._environment_values(context, session, guidance, feedback)
        for section_name in self.suite.context.order:
            if section_name == "trajectory":
                continue
            section = self.suite.sections[section_name]
            value = content.get(section_name, "")
            if not value:
                if section.empty == "omit":
                    continue
                value = section.empty_text
            template = section.template or self._legacy_section_template(
                section.heading
            )
            referenced = set(_TEMPLATE_VARIABLE.findall(template))
            variables = (("value", value),) if "value" in referenced else ()
            rendered = self._render_template(template, {"value": value})
            components.append(
                PromptComponent(
                    id=f"environment.{section_name}",
                    channel="environment",
                    template=template,
                    value=value,
                    rendered=rendered,
                    variables=variables,
                )
            )

        if feedback and "decision_request" not in self.suite.context.order:
            correction_template = "CORRECTION FROM THE SANDBOX:\n{{ value }}"
            components.append(
                PromptComponent(
                    id="environment.correction",
                    channel="environment",
                    template=correction_template,
                    value=feedback,
                    rendered=self._render_template(
                        correction_template,
                        {"value": feedback},
                    ),
                    variables=(("value", feedback),),
                )
            )
        return tuple(components)

    def _history(self, session: PlayerSession) -> tuple[ModelMessage, ...]:
        messages = tuple(session.messages)
        maximum = self.suite.context.trajectory.max_messages
        if maximum is None or len(messages) <= maximum:
            return messages

        messages = messages[-maximum:]
        if messages and messages[0].role == "assistant":
            messages = messages[1:]
        return messages

    def _environment_values(
        self,
        context: PlayerContext,
        session: PlayerSession,
        guidance: str,
        feedback: str | None,
    ) -> dict[str, str]:
        formatter = CatanObservationFormatter()
        formatted = formatter.format(
            context.observation,
            include_legal_actions=False,
            include_initial_placement_order=(
                self.suite.context.initial_placement_order == "both_rounds"
            ),
        )
        action_descriptions = (
            formatter._format_single_action(
                action,
                context.observation,
                discard_count=(
                    context.discard_count if "discard" in self.suite.response.tags else None
                ),
            )
            for action in context.legal_actions
        )
        decision_request = "Choose exactly one zero-based index from VALID ACTIONS."
        if feedback:
            decision_request = (
                f"{decision_request}\n\nCORRECTION FROM THE SANDBOX:\n{feedback}"
            )
        visible_events = self._format_events(context.events)
        legal_actions = "\n".join(
            f"{index}. {description}"
            for index, description in enumerate(action_descriptions)
        )
        values = {
            "strategic_memory": session.strategic_memory,
            "game_events": visible_events,
            "visible_events": visible_events,
            "observation": formatted.raw_str,
            "phase_info": formatted.strategic_context,
            "board_state": formatted.board_state,
            "resources": formatted.resources,
            "opponents": formatted.opponents,
            "trade_window": formatted.trade_context,
            "phase_guidance": guidance,
            "legal_actions": legal_actions,
            "decision_request": decision_request,
            "response_schema": self.suite.response.instruction,
        }
        if self.suite.context.social_context:
            values["recent_table_talk"] = self._format_events(context.recent_messages)
            values["commitments"] = "\n".join(
                f"{item.id}: {self._color_name(item.proposer)} to "
                f"{', '.join(self._color_name(color) for color in item.audience)}: "
                f"{item.condition} -> {item.promise} (expires turn {item.expires_turn})"
                for item in context.active_commitments
            )
        return values

    @staticmethod
    def _legacy_section_template(heading: str) -> str:
        if heading:
            return f"{heading}:\n{{{{ value }}}}"
        return "{{ value }}"

    @staticmethod
    def _format_events(events: tuple[PlayerEvent, ...]) -> str:
        return "\n".join(
            f"{event.sequence}. {ContextAssembler._color_name(event.actor)}: "
            f"{event.event_type}{ContextAssembler._format_detail(event.payload)}"
            for event in events
        )

    @staticmethod
    def _format_detail(detail: Any) -> str:
        if detail is None:
            return ""
        if isinstance(detail, Color):
            return f" {detail.value}"
        if isinstance(detail, (list, tuple)):
            rendered = ", ".join(
                ContextAssembler._format_value(value) for value in detail
            )
            return f" ({rendered})"
        return f" {ContextAssembler._format_value(detail)}"

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return "hidden"
        if isinstance(value, Color):
            return value.value
        return str(value)

    @staticmethod
    def _color_name(color: Color) -> str:
        return color.value if hasattr(color, "value") else str(color)

    @staticmethod
    def _render_template(template: str, values: dict[str, str]) -> str:
        referenced = set(_TEMPLATE_VARIABLE.findall(template))
        missing = referenced - set(values)
        if missing:
            raise ValueError(f"Template variables have no values: {sorted(missing)}")

        def replace(match: re.Match[str]) -> str:
            return values[match.group(1)]

        rendered = _TEMPLATE_VARIABLE.sub(replace, template)
        return rendered.strip()


class PlayerResponseParseError(ValueError):
    """Raised when model output cannot select the advertised exact menu."""


def _parse_json_integer(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise PlayerResponseParseError("JSON integer is too large.") from exc


class PlayerResponseParser:
    """Parse the suite's response contract against the exact legal menu."""

    def __init__(self, suite: ContextSuite) -> None:
        self.suite = suite

    def parse(
        self,
        context: PlayerContext,
        response: ModelResponse,
    ) -> PlayerChoice:
        text = response.content or ""
        try:
            fields, outside_text = parse_response_fields(
                text, instruction=self.suite.response.instruction
            )
        except ValueError as exc:
            raise PlayerResponseParseError(str(exc)) from exc
        for name, values in fields.items():
            if name != "action" and len(values) != 1:
                raise PlayerResponseParseError(f"Repeated <{name}> field.")
        game_plan = fields.get("game_plan", [""])[0]
        index, warning = self._parse_action_index(fields.get("action", []), outside_text)
        if index is None or index < 0 or index >= len(context.legal_actions):
            raise PlayerResponseParseError(
                f"Choose an action index from 0 to {len(context.legal_actions) - 1}; "
                f"received {index!r}."
            )

        selected_action = context.legal_actions[index]
        trade_offer = None
        offer_text = fields.get("trade_offer", [""])[0]
        if "trade_offer" in fields and selected_action.action_type not in {
            ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER
        }:
            raise PlayerResponseParseError("<trade_offer> is only valid for trade actions.")
        if (
            selected_action.action_type
            in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}
            and isinstance(selected_action.value, str)
        ):
            if not offer_text:
                raise PlayerResponseParseError(
                    "Selected trade action requires <trade_offer>."
                )
            trade_offer = self._parse_trade_offer(
                offer_text,
                context,
                selected_action.action_type,
                selected_action.value,
            )

        discard_cards = None
        if "discard" in fields:
            if selected_action.action_type != ActionType.DISCARD:
                raise PlayerResponseParseError("<discard> is only valid for DISCARD.")
            discard_cards = self._parse_discard(fields["discard"][0], context)
        elif (
            selected_action.action_type == ActionType.DISCARD
            and "discard" in self.suite.response.tags
        ):
            raise PlayerResponseParseError("Selected DISCARD action requires <discard>.")

        return PlayerChoice(
            action_index=index,
            trade_offer=trade_offer,
            game_plan=game_plan,
            raw_response=text,
            model=response.model,
            usage=response.usage,
            latency_ms=response.latency_ms,
            parse_warning=warning,
            native_reasoning=response.native_reasoning,
            native_reasoning_details=response.native_reasoning_details,
            reasoning_request=response.reasoning_request,
            provider_response_id=response.provider_response_id,
            provider_request_id=response.provider_request_id,
            provider_native_finish_reason=response.provider_native_finish_reason,
            discard_cards=discard_cards,
        )

    @staticmethod
    def _parse_discard(value: str, context: PlayerContext) -> tuple[str, ...]:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for resource, count in pairs:
                resource = resource.upper()
                if resource in result:
                    raise PlayerResponseParseError(f"Duplicate discard resource: {resource}")
                result[resource] = count
            return result

        try:
            payload = json.loads(
                value, object_pairs_hook=unique_object, parse_int=_parse_json_integer
            )
        except json.JSONDecodeError as exc:
            raise PlayerResponseParseError(f"discard must contain valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise PlayerResponseParseError("discard must be a named resource-count JSON object.")
        for resource, count in payload.items():
            if resource not in RESOURCE_NAMES:
                raise PlayerResponseParseError(f"Unknown discard resource: {resource}")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise PlayerResponseParseError("Named discard counts must be positive integers.")
            if count > context.observation.my_resources.get(resource, 0):
                raise PlayerResponseParseError(f"Discard exceeds your {resource} holdings.")
        if sum(payload.values()) != context.discard_count:
            raise PlayerResponseParseError(f"Discard exactly {context.discard_count} cards.")
        return tuple(
            resource for resource in RESOURCE_NAMES for _ in range(payload.get(resource, 0))
        )

    @staticmethod
    def _parse_trade_offer(
        value: str,
        context: PlayerContext,
        action_type: ActionType,
        action_value: str,
    ) -> TradeOffer:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for key, item in pairs:
                if key in result:
                    raise PlayerResponseParseError(
                        f"Duplicate trade_offer JSON key: {key}"
                    )
                result[key] = item
            return result

        try:
            payload = json.loads(
                value, object_pairs_hook=unique_object, parse_int=_parse_json_integer
            )
        except json.JSONDecodeError as exc:
            raise PlayerResponseParseError(
                "trade_offer must contain valid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise PlayerResponseParseError("trade_offer must be a JSON object.")
        allowed = {"give", "receive", "give_any", "receive_any"}
        unknown = set(payload) - allowed
        if unknown:
            raise PlayerResponseParseError(
                f"Unknown trade_offer fields: {sorted(unknown)}"
            )

        def parse_bundle(field: str) -> tuple[int, int, int, int, int]:
            resource_counts = payload.get(field)
            if not isinstance(resource_counts, dict):
                raise PlayerResponseParseError(
                    f"trade_offer.{field} must be a resource-count object."
                )
            bundle = [0] * len(RESOURCE_NAMES)
            seen = set()
            for resource, count in resource_counts.items():
                if not isinstance(resource, str):
                    raise PlayerResponseParseError("Resource names must be strings.")
                resource = resource.upper()
                if resource not in RESOURCE_NAMES:
                    raise PlayerResponseParseError(
                        f"Unknown trade resource: {resource}"
                    )
                if resource in seen:
                    raise PlayerResponseParseError(
                        f"Duplicate trade resource: {resource}"
                    )
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise PlayerResponseParseError(
                        "Named trade resource counts must be positive integers."
                    )
                seen.add(resource)
                bundle[RESOURCE_NAMES.index(resource)] = count
            return tuple(bundle)

        def parse_any(field: str) -> int:
            count = payload.get(field, 0)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise PlayerResponseParseError(
                    f"trade_offer.{field} must be a non-negative integer."
                )
            return count

        try:
            parent_offer_id = None
            audience = frozenset(context.observation.opponent_resource_counts)
            if action_type == ActionType.COUNTER_OFFER:
                parent_offer_id = action_value.removeprefix("COUNTER_OFFER:").rsplit(
                    ":", 1
                )[0]
                audience = frozenset({context.observation.turn_player_color})
            return TradeOffer(
                offered_by=context.actor,
                audience=audience,
                give=parse_bundle("give"),
                receive=parse_bundle("receive"),
                give_any=parse_any("give_any"),
                receive_any=parse_any("receive_any"),
                parent_offer_id=parent_offer_id,
            )
        except ValueError as exc:
            raise PlayerResponseParseError(str(exc)) from exc

    def _parse_action_index(
        self, tagged_values: list[str], outside_text: str
    ) -> tuple[int | None, str | None]:
        # An exact authored placeholder echo is not a selection; numeric examples are.
        tagged_values = [
            value for value in tagged_values
            if not value
            or re.fullmatch(r"[0-9]+", value)
            or f"<action>{value}</action>" not in self.suite.response.instruction
        ]
        remaining = outside_text.strip()
        named_values = re.findall(
            r"\b(?:action|move)(?:_index)?\s*[:=]\s*([^\r\n]*)",
            remaining,
            flags=re.IGNORECASE,
        )
        values = tagged_values + [value.strip() for value in named_values]
        if re.fullmatch(r"[0-9]+", remaining):
            values.append(remaining)
        if any(re.fullmatch(r"[0-9]+", value) is None for value in values):
            raise PlayerResponseParseError(
                "Action selections must be whole non-negative integers."
            )
        try:
            indices = {int(value) for value in values}
        except ValueError as exc:
            raise PlayerResponseParseError("Action index is too large.") from exc
        if len(indices) > 1:
            raise PlayerResponseParseError("Response contains conflicting action selections.")
        warning = None
        if indices and not tagged_values:
            warning = "Missing numeric <action> tag; used an explicit index fallback."
        return next(iter(indices), None), warning
