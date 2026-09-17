"""Deterministic context assembly and typed policy-response parsing."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from cle.env.observation_formatter import CatanObservationFormatter
from cle.harness.action_tools import parse_tool_choice, render_action_tools, render_shared_legal_actions, trade_responder_note
from cle.players.action_batches import validate_batch_actions
from cle.harness.board_surface import BoardPresenter
from cle.harness.catan_board_surface import IndexedTileRowsBoardPresenter
from cle.harness.components import (
    ComponentInputs,
    observation_component_values,
    parse_strict_json_object,
    render_component_definitions,
)
from cle.harness.models import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PlayerSession,
    PromptComponent,
)
from cle.harness.suite import ContextSuite
from cle.harness.response_xml import parse_response_fields
from cle.harness.public_speech import parse_public_speech
from cle.players.contracts import CommunicationChoice, PlayerChoice, PlayerContext
from cle.players.notes import validate_notes
from cle.game_engine.communication import SocialCommitment
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
        system = "\n\n".join(
            component.rendered for component in components
            if component.channel == "system" and component.rendered
        )
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

        if self.suite.context.mode == "shared":
            if context.observation.my_color != context.actor:
                raise ValueError("Observation perspective does not match context actor")
            request = "Choose one action using its tool and named arguments."
            if self.suite.context.reactive_speech:
                request = (
                    "Choose one game action OR standalone say. Say keeps this game decision pending; "
                    "after bounded replies you must act."
                    if context.speech_allowed else
                    "Choose one game action now. Standalone say is unavailable for this decision."
                )
            if self.suite.context.deterministic_batches:
                request += " You may replace a single game action with a bounded deterministic actions batch, as defined in the response schema."
            if feedback:
                request += f"\n\nCORRECTION FROM THE SANDBOX:\n{feedback}"
            if self.suite.context.deterministic_batches:
                legal_actions = render_action_tools(context, shared=True)
                responder_note = trade_responder_note(context)
                if responder_note:
                    legal_actions = f"{legal_actions}\n\n{responder_note}"
            else:
                legal_actions = render_shared_legal_actions(context)
            return render_component_definitions(
                self.suite.components,
                self.suite.context.order,
                ComponentInputs(
                    color=self._color_name(session.color),
                    notes=session.strategic_memory,
                    max_notes_chars=str(self.suite.context.max_notes_chars),
                    visible_events=self._format_events(context.events, shared=True),
                    recent_table_talk=self._format_events(context.recent_messages),
                    commitments=self._format_commitments(context.active_commitments),
                    phase_guidance=guidance,
                    legal_actions=legal_actions,
                    decision_request=request,
                    **observation_component_values(
                        context.observation,
                        include_initial_placement_order=(
                            self.suite.context.initial_placement_order == "both_rounds"
                        ),
                    ),
                ),
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
        if self.suite.context.memory_mode == "fresh_notes":
            return ()
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
        if self.suite.response.format == "json":
            legal_actions = render_action_tools(context)
            decision_request = "Choose exactly one available tool with named arguments."
        else:
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
            legal_actions = "\n".join(
                f"{index}. {description}"
                for index, description in enumerate(action_descriptions)
            )
            decision_request = "Choose exactly one zero-based index from VALID ACTIONS."
        if feedback:
            decision_request = (
                f"{decision_request}\n\nCORRECTION FROM THE SANDBOX:\n{feedback}"
            )
        visible_events = self._format_events(context.events)
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
    def _format_commitments(commitments: tuple[SocialCommitment, ...]) -> str:
        return "\n".join(
            f"{item.id}: {ContextAssembler._color_name(item.proposer)} to "
            f"{', '.join(ContextAssembler._color_name(color) for color in item.audience)}: "
            f"{item.condition} -> {item.promise} (expires turn {item.expires_turn})"
            for item in commitments
        )

    @staticmethod
    def _legacy_section_template(heading: str) -> str:
        if heading:
            return f"{heading}:\n{{{{ value }}}}"
        return "{{ value }}"

    @staticmethod
    def _format_events(events: tuple[PlayerEvent, ...], *, shared: bool = False) -> str:
        return "\n".join(
            f"{event.sequence}. {ContextAssembler._color_name(event.actor)}: "
            f"{event.event_type}{ContextAssembler._shared_event_detail(event) if shared else ContextAssembler._historical_event_detail(event)}"
            for event in events
        )

    @staticmethod
    def _historical_event_detail(event: PlayerEvent) -> str:
        payload = event.payload
        if isinstance(payload, dict):
            if event.event_type in {"ACCEPT_TRADE", "REJECT_TRADE", "CANCEL_TRADE"} and "offer" in payload:
                payload = payload["offer"]["id"]
            elif event.event_type == "CONFIRM_TRADE" and "offer" in payload:
                payload = {key: payload[key] for key in ("offer_id", "turn_player", "counterparty")}
            elif event.event_type == "COUNTER_OFFER" and "original" in payload:
                payload = {key: value for key, value in payload.items() if key != "original"}
            elif event.event_type in {"CLOSE_TRADE", "CLEAR_TRADE_RESPONSE"} and "offer" in payload:
                payload = {key: value for key, value in payload.items() if key != "offer"}
        return ContextAssembler._format_detail(payload)

    @staticmethod
    def _shared_event_detail(event: PlayerEvent) -> str:
        payload = event.payload
        if not isinstance(payload, dict):
            return ContextAssembler._format_detail(payload)
        offer = payload.get("offer", payload)
        if event.event_type == "CONFIRM_TRADE" and {"give", "receive", "turn_player"} <= payload.keys():
            offer = {**payload, "offered_by": payload["turn_player"], "audience": [payload["counterparty"]]}
        if not isinstance(offer, dict) or not {"offered_by", "give", "receive"} <= offer.keys():
            return ContextAssembler._format_detail(payload)

        def terms(item: dict) -> str:
            sides = []
            for side in ("give", "receive"):
                parts = [f"{count} {resource}" for resource, count in item[side].items()]
                if item.get(f"{side}_any"):
                    parts.append(f"{item[f'{side}_any']} ANY")
                sides.append(", ".join(parts) or "nothing")
            return f"{item['offered_by']} gives {sides[0]}, receives {sides[1]}"

        detail = " " + terms(offer)
        detail += "; audience: " + ", ".join(offer.get("audience", ()))
        if payload.get("counterparty"):
            detail += f"; confirmed with {payload['counterparty']}"
        if payload.get("original"):
            detail += "; original: " + terms(payload["original"])
        if payload.get("reason"):
            detail += f"; reason: {payload['reason']}"
        if offer.get("give_any") or offer.get("receive_any"):
            detail += "; unresolved proposal, not executable"
        return detail

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
    """Raised when model output cannot select an authorized action."""


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
    ) -> PlayerChoice | CommunicationChoice:
        text = response.content or ""
        if not text.strip():
            message = (
                "The provider returned reasoning but no final answer."
                if response.native_reasoning.strip() or response.native_reasoning_details
                else "The provider returned no final answer."
            )
            if response.finish_reason == "length" or response.provider_native_finish_reason in {
                "length", "max_tokens"
            }:
                message += " The provider reported a completion-token limit."
            raise PlayerResponseParseError(message)
        if self.suite.context.reactive_speech:
            try:
                payload = parse_strict_json_object(text)
                if payload.get("tool") == "say":
                    if not context.speech_allowed:
                        raise ValueError("Standalone say is unavailable; choose a game action")
                    if set(payload) - {"tool", "arguments", "notes"} or not isinstance(payload.get("arguments"), dict):
                        raise ValueError("Say requires tool, arguments, and optional notes")
                    if set(payload["arguments"]) & {"mode", "notes"}:
                        raise ValueError("Say mode is implicit; notes belong outside arguments")
                    speech = parse_public_speech(
                        {**payload["arguments"], "mode": "say", **({"notes": payload["notes"]} if "notes" in payload else {})},
                        speaker=context.actor,
                        participants=(context.actor, *context.observation.opponent_resource_counts),
                        max_notes_chars=self.suite.context.max_notes_chars,
                    )
                    return speech
            except (ValueError, TypeError, RecursionError) as exc:
                raise PlayerResponseParseError(f"Invalid choice: {exc}") from exc
        choice = (
            self._parse_tool_response(
                context, text,
                notes_max_chars=(
                    self.suite.context.max_notes_chars
                    if self.suite.context.memory_mode == "fresh_notes" else None
                ),
                shared=self.suite.context.mode == "shared",
                batches=self.suite.context.deterministic_batches,
            )
            if self.suite.response.format == "json"
            else self._parse_indexed_response(context, text)
        )
        return replace(
            choice,
            raw_response=text,
            model=response.model,
            usage=response.usage,
            latency_ms=response.latency_ms,
            native_reasoning=response.native_reasoning,
            native_reasoning_details=response.native_reasoning_details,
            reasoning_request=response.reasoning_request,
            provider_response_id=response.provider_response_id,
            provider_request_id=response.provider_request_id,
            provider_native_finish_reason=response.provider_native_finish_reason,
        )

    @staticmethod
    def _parse_tool_response(
        context: PlayerContext, text: str, *, notes_max_chars: int | None = None, shared: bool = False,
        batches: bool = False,
    ) -> PlayerChoice:
        # JSON preserves raw atlas tokens such as <T05> without XML escaping.
        if len(text) > 128 * 1024:
            raise PlayerResponseParseError("Response exceeds 131072 characters.")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise PlayerResponseParseError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise PlayerResponseParseError(f"Invalid JSON constant: {value}")

        try:
            payload = parse_strict_json_object(text) if notes_max_chars is not None else json.loads(
                text,
                object_pairs_hook=unique_object,
                parse_int=_parse_json_integer,
                parse_constant=reject_constant,
            )
            memory_field = "notes" if notes_max_chars is not None else "game_plan"
            if isinstance(payload, dict) and "actions" in payload:
                if not batches or not shared or notes_max_chars is None:
                    raise ValueError("Action batches are unavailable in this contract")
                if set(payload) - {"actions", "notes"}:
                    raise ValueError("Batch envelope permits only actions and optional notes")
                notes = validate_notes(payload["notes"], notes_max_chars) if "notes" in payload else None
                actions = validate_batch_actions(payload["actions"])
                spatial = [
                    (field, json.dumps(value))
                    for call in actions for field, value in call["arguments"].items()
                    if field in {"node", "edge"}
                ]
                raw_spatial = re.findall(r'(?<!\\)"(node|edge)"\s*:\s*("(?:[^"\\]|\\.)*")', text)
                if spatial != raw_spatial:
                    raise ValueError("Spatial arguments must contain literal trained board tokens, not JSON escapes.")
                first = actions[0]
                choice = parse_tool_choice(context, first["tool"], first["arguments"], shared=True)
                return replace(choice, batch_actions=actions, notes_update=notes)
            # "arguments":{} on a no-parameter tool is a formality; a bare
            # {"tool":"end_turn"} means the same thing and is accepted as such.
            # Tools that do take parameters still fail on their own missing fields.
            if isinstance(payload, dict) and "tool" in payload and "arguments" not in payload:
                payload = {**payload, "arguments": {}}
            if not isinstance(payload, dict) or (
                set(payload) - {memory_field, "tool", "arguments"}
                or not {"tool", "arguments"} <= set(payload)
            ):
                if isinstance(payload, dict) and set(payload) - {memory_field, "tool", "arguments"}:
                    extra = ", ".join(sorted(set(payload) - {memory_field, "tool", "arguments"}))
                    raise ValueError(
                        f"Unexpected top-level keys: {extra}. Return one JSON object with tool, "
                        f"arguments, and optional {memory_field}; put everything else inside arguments."
                    )
                raise ValueError(
                    f"Return one JSON object with tool, arguments, and optional {memory_field}."
                )
            notes_update = None
            if notes_max_chars is not None and "notes" in payload:
                try:
                    notes_update = validate_notes(payload["notes"], max_chars=notes_max_chars)
                except TypeError as exc:
                    raise ValueError(str(exc)) from exc
            game_plan = payload.get("game_plan", "")
            if not isinstance(game_plan, str):
                raise ValueError("game_plan must be a string.")
            choice = parse_tool_choice(context, payload["tool"], payload["arguments"], shared=shared)
            if payload["tool"] in {
                "build_settlement", "upgrade_city", "build_road", "move_robber", "play_knight"
            }:
                field, token = next(iter(payload["arguments"].items()))
                literal_argument = (
                    r'(?<!\\)' + re.escape(json.dumps(field)) + r'\s*:\s*'
                    + re.escape(json.dumps(token))
                )
                if not re.search(literal_argument, text):
                    raise ValueError("Spatial arguments must contain literal trained board tokens, not JSON escapes.")
            if notes_max_chars is not None:
                return replace(choice, notes_update=notes_update)
            return replace(choice, game_plan=game_plan)
        except (ValueError, TypeError, RecursionError) as exc:
            raise PlayerResponseParseError(f"Invalid tool call: {exc}") from exc

    def _parse_indexed_response(self, context: PlayerContext, text: str) -> PlayerChoice:
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
            parse_warning=warning,
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
