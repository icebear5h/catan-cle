"""Compact legal tool guidance without numbered action menu rows."""

from __future__ import annotations

import json

from cle.game_engine.trading import TradeOffer
from cle.harness import action_tools
from cle.players.contracts import PlayerContext
from cle.players.data import JsonValue

from .arguments import SpatialCategory


def render_action_tools(context: PlayerContext, *, shared: bool = False) -> str:
    """Compact available tools and concrete terms, without numbered menu rows."""
    if shared:
        return action_tools.SHARED_ACTION_TOOLS
    lines = []
    for tool, action_type in action_tools._TOOL_TYPES.items():
        actions = [
            action
            for action in context.legal_actions
            if action.action_type == action_type and action.color == context.actor
        ]
        if not actions:
            continue
        if tool in {"build_settlement", "upgrade_city", "build_road", "move_robber", "play_knight"}:
            category: SpatialCategory = (
                "edge"
                if tool == "build_road"
                else ("tile" if tool in {"move_robber", "play_knight"} else "node")
            )
            spatial = action_tools._spatial_values(context, category)
            values = [
                action_tools.canonical_edge(a.value) if category == "edge" else a.value
                for a in actions
            ]
            tokens = sorted(
                token
                for token, value in spatial.items()
                if (tool == "play_knight" and None in values) or value in values
            )
            if tokens:
                lines.append(f"{tool}({category}): {' '.join(tokens)}")
        elif tool == "steal_from":
            colors = sorted({a.value[0].value for a in actions if a.value[1] is None})
            if colors:
                lines.append(f"{tool}(player): {', '.join(colors)}")
        elif tool == "play_monopoly":
            lines.append(f"{tool}(resource): {', '.join(sorted({a.value for a in actions}))}")
        elif tool == "play_year_of_plenty":
            totals = "/".join(str(n) for n in sorted({len(a.value) for a in actions}))
            limits = {
                r: max(a.value.count(r) for a in actions)
                for r in action_tools.RESOURCE_NAMES
                if any(r in a.value for a in actions)
            }
            singletons = sorted({a.value[0] for a in actions if len(a.value) == 1})
            lines.append(
                f"{tool}(take): resource-count map, {totals} cards; per-resource maxima {json.dumps(limits)}"
                + (f"; allowed singletons: {json.dumps(singletons)}" if singletons else "")
            )
        elif tool == "maritime_trade":
            rates = {a.value[0]: sum(r is not None for r in a.value[:4]) for a in actions}
            receive = sorted({a.value[-1] for a in actions})
            lines.append(
                f"{tool}(give, receive): resource-count maps; give one resource at its exact rate "
                f"{json.dumps(dict(sorted(rates.items())))}; receive one different card: {', '.join(receive)}"
            )
        elif tool == "discard":
            lines.append(
                f"{tool}(cards): resource-count map, exactly {context.discard_count} cards from your hand"
            )
        elif tool in {"offer_trade", "counter_offer"}:
            signature = "offer_id, give, receive" if tool == "counter_offer" else "give, receive"
            parameterized = [a for a in actions if isinstance(a.value, str)]
            if parameterized:
                detail = "other seats"
                if tool == "counter_offer":
                    parents = sorted(
                        parent for parent in {action_tools._counter_parent(a.value) for a in parameterized}
                        if parent is not None
                    )
                    detail = f"turn owner {context.observation.turn_player_color.value}; parent offer_id: {json.dumps(parents)}"
                lines.append(
                    f"{tool}({signature}, give_any=0, receive_any=0): resource-count maps; "
                    f"audience fixed to {detail} (omit audience); without audience, matching parameterized offers "
                    "take precedence over concrete alternatives; nonempty sides, no named resource on both sides, "
                    "give only held cards; wildcards are proposals only"
                )
            concrete = []
            for action in actions:
                if not isinstance(action.value, TradeOffer):
                    continue
                payload = action.value.to_payload()
                descriptor: dict[str, JsonValue] = {
                    "give": dict(payload["give"]),
                    "receive": dict(payload["receive"]),
                    "give_any": payload["give_any"],
                    "receive_any": payload["receive_any"],
                    "audience": list(payload["audience"]),
                }
                if tool == "counter_offer":
                    descriptor["offer_id"] = action.value.parent_offer_id
                concrete.append(json.dumps(descriptor, separators=(",", ":")))
            if concrete:
                lines.append(
                    f"{tool}({signature}, give_any=0, receive_any=0[, audience]): "
                    "audience is an optional color array selecting a listed concrete offer only; "
                    "required if ambiguous or a matching parameterized offer exists; exact alternatives: ["
                    + ",".join(sorted(concrete))
                    + "]"
                )
        elif tool in {"accept_offer", "reject_offer", "cancel_trade"}:
            lines.append(f"{tool}(offer_id): {json.dumps(sorted({a.value for a in actions}))}")
        elif tool == "confirm_trade":
            candidates: dict[str, set[str]] = {}
            for action in actions:
                candidate = action.value
                candidates.setdefault(candidate.offer_id, set()).add(candidate.counterparty.value)
            lines.append(
                f"{tool}(offer_id, counterparty): "
                + json.dumps({key: sorted(value) for key, value in sorted(candidates.items())})
            )
        elif any(a.value is None for a in actions):
            lines.append(f"{tool}()")
    if not lines:
        return "No unambiguous action tools are available."
    return "\n".join(
        [
            "Available tools (choose one; argument names are exact). Spatial tokens are literal and case-sensitive.",
            "Resource maps use WOOD, BRICK, SHEEP, WHEAT, ORE (case-insensitive), with positive integer counts and no duplicate casing. "
            "Optional wildcard counts are non-negative integers. Offer IDs are exact opaque strings. All arguments must match current legality.",
            *lines,
        ]
    )


def trade_responder_note(context: PlayerContext) -> str:
    """One-line framing for off-turn trade responders; empty for the turn player."""
    turn_player = getattr(context.observation, "turn_player_color", None)
    if turn_player is None or turn_player == context.actor:
        return ""
    return (
        f"You are {context.actor.value} responding to {turn_player.value}'s open trade offer; "
        "end_turn is not available to you. Respond with accept_offer, reject_offer, or counter_offer."
    )


def legal_tool_names(context: PlayerContext) -> tuple[str, ...]:
    """Sorted tool names with at least one legal action for the actor."""
    return tuple(sorted(
        {
            action_tools._REVERSE_TOOL_TYPES[action.action_type]
            for action in context.legal_actions
            if action.color == context.actor and action.action_type in action_tools._REVERSE_TOOL_TYPES
        }
    ))


def render_shared_legal_actions(context: PlayerContext) -> str:
    """Stable tool definitions plus the actor's exact currently-legal tool names.

    Shared-mode prompts otherwise list every tool signature without saying which
    moves are legal now, which invites off-turn actors to call end_turn as a
    pass. The appended line names only the tools with a matching legal action
    for this actor, and names the trade responder's situation explicitly.
    """
    tools = action_tools.legal_tool_names(context)
    lines = [
        action_tools.SHARED_ACTION_TOOLS,
        "",
        "YOUR CURRENTLY LEGAL TOOLS (call exactly one of these): "
        + (", ".join(tools) if tools else "(none)"),
    ]
    note = action_tools.trade_responder_note(context)
    if note:
        lines.append(note)
    return "\n".join(lines)
