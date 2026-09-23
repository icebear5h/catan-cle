"""Strict JSON tool-call admission for the shared and historical action contracts."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import cast

from cle.harness.action_tools import parse_tool_choice
from cle.harness.components import parse_strict_json_object
from cle.harness.context.errors import PlayerResponseParseError, parse_json_integer
from cle.players.action_batches import validate_batch_actions
from cle.players.contracts import PlayerChoice, PlayerContext
from cle.players.data import JsonValue
from cle.players.notes import validate_notes

SPATIAL_TOOLS = frozenset({
    "build_settlement", "upgrade_city", "build_road", "move_robber", "play_knight",
})


def parse_tool_response(
    context: PlayerContext, text: str, *, notes_max_chars: int | None = None, shared: bool = False,
    batches: bool = False,
) -> PlayerChoice:
    # JSON preserves raw atlas tokens such as <T05> without XML escaping.
    if len(text) > 128 * 1024:
        raise PlayerResponseParseError("Response exceeds 131072 characters.")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
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
            parse_int=parse_json_integer,
            parse_constant=reject_constant,
        )
        memory_field = "notes" if notes_max_chars is not None else "game_plan"
        if isinstance(payload, dict) and "actions" in payload:
            if not batches or not shared or notes_max_chars is None:
                raise ValueError("Action batches are unavailable in this contract")
            if set(payload) - {"actions", "notes"}:
                raise ValueError("Batch envelope permits only actions and optional notes")
            notes = (
                validate_notes(cast("str", payload["notes"]), notes_max_chars)
                if "notes" in payload else None
            )
            actions = validate_batch_actions(cast("JsonValue", payload["actions"]))
            spatial = [
                (field, json.dumps(value))
                for call in actions
                for field, value in cast("dict[str, JsonValue]", call["arguments"]).items()
                if field in {"node", "edge"}
            ]
            raw_spatial = re.findall(r'(?<!\\)"(node|edge)"\s*:\s*("(?:[^"\\]|\\.)*")', text)
            if spatial != raw_spatial:
                raise ValueError("Spatial arguments must contain literal trained board tokens, not JSON escapes.")
            first = actions[0]
            choice = parse_tool_choice(
                context,
                cast("str", first["tool"]),
                cast("dict[str, JsonValue]", first["arguments"]),
                shared=True,
            )
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
                notes_update = validate_notes(
                    cast("str", payload["notes"]), max_chars=notes_max_chars
                )
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
        game_plan = payload.get("game_plan", "")
        if not isinstance(game_plan, str):
            raise ValueError("game_plan must be a string.")
        tool = cast("str", payload["tool"])
        arguments = cast("dict[str, JsonValue]", payload["arguments"])
        choice = parse_tool_choice(context, tool, arguments, shared=shared)
        if tool in SPATIAL_TOOLS:
            field, token = next(iter(arguments.items()))
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
