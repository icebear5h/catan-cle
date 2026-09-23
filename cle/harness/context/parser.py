"""Typed policy-response parsing against the exact authorized action menu."""

from __future__ import annotations

import re
from dataclasses import replace

from cle.game_engine.models.enums import ActionType
from cle.harness.components import parse_strict_json_object
from cle.harness.context.errors import PlayerResponseParseError
from cle.harness.context.response_values import parse_discard, parse_trade_offer
from cle.harness.context.tool_response import parse_tool_response
from cle.harness.models import ModelResponse
from cle.harness.public_speech import parse_public_speech
from cle.harness.response_xml import parse_response_fields
from cle.harness.suite import ContextSuite
from cle.players.contracts import CommunicationChoice, PlayerChoice, PlayerContext


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
                    arguments = payload.get("arguments")
                    if set(payload) - {"tool", "arguments", "notes"} or not isinstance(arguments, dict):
                        raise ValueError("Say requires tool, arguments, and optional notes")
                    if set(arguments) & {"mode", "notes"}:
                        raise ValueError("Say mode is implicit; notes belong outside arguments")
                    speech = parse_public_speech(
                        {**arguments, "mode": "say", **({"notes": payload["notes"]} if "notes" in payload else {})},
                        speaker=context.actor,
                        participants=(context.actor, *context.observation.opponent_resource_counts),
                        max_notes_chars=self.suite.context.max_notes_chars,
                    )
                    return speech
            except (ValueError, TypeError, RecursionError) as exc:
                raise PlayerResponseParseError(f"Invalid choice: {exc}") from exc
        choice = (
            parse_tool_response(
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
            trade_offer = parse_trade_offer(
                offer_text,
                context,
                selected_action.action_type,
                selected_action.value,
            )

        discard_cards = None
        if "discard" in fields:
            if selected_action.action_type != ActionType.DISCARD:
                raise PlayerResponseParseError("<discard> is only valid for DISCARD.")
            discard_cards = parse_discard(fields["discard"][0], context)
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
