"""Semantic tools bound to a perspective-safe, authoritative engine menu.

Board strings are literal trained vocabulary, not XML. This module neither
parses the response envelope nor executes actions. Parameterized offers and
discards remain subject to strict engine admission after menu resolution.
"""

from __future__ import annotations

import json
from collections import Counter

from cle.game_engine.board_tokens import canonical_edge, edge_token, node_token, tile_token
from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES, ResourceBundle, TradeCandidate, TradeOffer
from cle.players.contracts import PlayerChoice, PlayerContext
from cle.players.validation import action_from_choice


_TOOL_TYPES = {
    "build_settlement": ActionType.BUILD_SETTLEMENT,
    "build_road": ActionType.BUILD_ROAD,
    "upgrade_city": ActionType.BUILD_CITY,
    "play_knight": ActionType.PLAY_KNIGHT_CARD,
    "move_robber": ActionType.MOVE_ROBBER,
    "steal_from": ActionType.STEAL,
    "play_year_of_plenty": ActionType.PLAY_YEAR_OF_PLENTY,
    "play_monopoly": ActionType.PLAY_MONOPOLY,
    "maritime_trade": ActionType.MARITIME_TRADE,
    "discard": ActionType.DISCARD,
    "offer_trade": ActionType.OFFER_TRADE,
    "accept_offer": ActionType.ACCEPT_TRADE,
    "reject_offer": ActionType.REJECT_TRADE,
    "counter_offer": ActionType.COUNTER_OFFER,
    "confirm_trade": ActionType.CONFIRM_TRADE,
    "cancel_trade": ActionType.CANCEL_TRADE,
    "buy_development_card": ActionType.BUY_DEVELOPMENT_CARD,
    "play_road_building": ActionType.PLAY_ROAD_BUILDING,
    "roll_dice": ActionType.ROLL,
    "end_turn": ActionType.END_TURN,
}


def _require_arguments(arguments: dict, *required: str, optional: tuple[str, ...] = ()) -> None:
    if set(arguments) - set(required) - set(optional) or set(required) - set(arguments):
        raise ValueError(
            f"Expected arguments {', '.join(required) or '(none)'}"
            + (f"; optional: {', '.join(optional)}" if optional else "")
        )


def _resource(value: object) -> str:
    if not isinstance(value, str) or value.upper() not in RESOURCE_NAMES:
        raise ValueError("Resource must be WOOD, BRICK, SHEEP, WHEAT, or ORE")
    return value.upper()


def _bundle(value: object) -> ResourceBundle:
    if not isinstance(value, dict):
        raise ValueError("Resources must be a named resource-count object")
    counts = {}
    for key, count in value.items():
        resource = _resource(key)
        if resource in counts:
            raise ValueError(f"Duplicate resource: {resource}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Named resource counts must be positive integers")
        counts[resource] = count
    return tuple(counts.get(resource, 0) for resource in RESOURCE_NAMES)


def _card_counts(cards) -> ResourceBundle:
    counts = Counter(cards)
    return tuple(counts[resource] for resource in RESOURCE_NAMES)


def _color(value: object) -> Color:
    if not isinstance(value, str):
        raise ValueError("Player color must be a string")
    try:
        return Color(value.upper())
    except ValueError as exc:
        raise ValueError(f"Unknown player color: {value!r}") from exc


def _offer_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("offer_id must be an exact nonempty offer ID string")
    return value


def _counter_parent(value) -> str | None:
    if isinstance(value, TradeOffer):
        return value.parent_offer_id
    if isinstance(value, str) and value.startswith("COUNTER_OFFER:"):
        parent, separator, _ = value.removeprefix("COUNTER_OFFER:").rpartition(":")
        if separator and parent:
            return parent
    return None


def _spatial_values(context: PlayerContext, category: str) -> dict:
    board_map = context.observation.board_map
    if board_map is None:
        raise ValueError("Spatial tools require the observation's actual board map")
    if category == "tile":
        return {
            tile_token(tile.id): coordinate
            for coordinate, tile in board_map.land_tiles.items()
            if coordinate != context.observation.robber_position
        }
    if category == "node":
        return {
            node_token(node): node
            for tile in board_map.land_tiles.values()
            for node in tile.nodes.values()
        }
    return {
        edge_token(edge): canonical_edge(edge)
        for tile in board_map.land_tiles.values()
        for edge in tile.edges.values()
    }


def _semantic_offer(context: PlayerContext, player: object, terms: object, *, own: bool | None = None, root: bool = False) -> TradeOffer:
    """Match visible active terms before legality filtering; never guess between IDs."""
    partner = _color(player)
    if partner == context.actor or partner not in context.observation.opponent_resource_counts:
        raise ValueError("player must identify another participant")
    if not isinstance(terms, dict):
        raise ValueError("Trade terms must be an object")
    _require_arguments(terms, "give", "receive", optional=("give_any", "receive_any"))
    give, receive = _bundle(terms["give"]), _bundle(terms["receive"])
    wild = tuple(terms.get(key, 0) for key in ("give_any", "receive_any"))
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in wild):
        raise ValueError("Wildcard counts must be non-negative integers")
    window = context.observation.trade_window
    matches = []
    if window is not None and window.status.value == "open":
        for offer in window.active_offers:
            is_own = offer.offered_by == context.actor
            if own is not None and is_own != own:
                continue
            if root and offer.parent_offer_id is not None:
                continue
            if is_own:
                if partner not in offer.audience:
                    continue
                actual = (offer.give, offer.receive, offer.give_any, offer.receive_any)
            else:
                if offer.offered_by != partner or context.actor not in offer.audience:
                    continue
                actual = (offer.receive, offer.give, offer.receive_any, offer.give_any)
            if actual == (give, receive, *wild):
                matches.append(offer)
    if not matches:
        raise ValueError("No active visible offer matches that player and your give/receive terms; it may be stale")
    if len(matches) != 1:
        raise ValueError("Ambiguous trade: multiple active offers have those player/terms; no offer was selected")
    return matches[0]


def _shared_trade_arguments(context: PlayerContext, tool: str, arguments: dict) -> dict:
    if tool == "counter_offer":
        _require_arguments(arguments, "player", "original", "proposed")
        parent = _semantic_offer(context, arguments["player"], arguments["original"], own=False, root=True)
        proposed = arguments["proposed"]
        if not isinstance(proposed, dict):
            raise ValueError("proposed must contain your give/receive terms")
        _require_arguments(proposed, "give", "receive", optional=("give_any", "receive_any"))
        return {"offer_id": parent.id, **proposed}
    _require_arguments(arguments, "player", "give", "receive", optional=("give_any", "receive_any"))
    terms = {key: value for key, value in arguments.items() if key != "player"}
    own = True if tool == "cancel_trade" else (None if tool == "confirm_trade" else False)
    offer = _semantic_offer(context, arguments["player"], terms, own=own)
    result = {"offer_id": offer.id}
    if tool == "confirm_trade":
        result["counterparty"] = arguments["player"]
    return result


def parse_tool_choice(context: PlayerContext, tool: str, arguments: dict, *, shared: bool = False) -> PlayerChoice:
    """Resolve exactly one semantic call; invalid or ambiguous calls raise ValueError."""
    if not isinstance(tool, str) or tool not in _TOOL_TYPES:
        raise ValueError("Unknown action tool")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    priority = None
    if shared and tool == "offer_trade":
        _require_arguments(arguments, "give", "receive", optional=("give_any", "receive_any", "confirm_if_accepted_by"))
        if "confirm_if_accepted_by" in arguments:
            selected = arguments["confirm_if_accepted_by"]
            if selected == "ANY":
                priority = "ANY"
            elif isinstance(selected, list) and selected:
                priority = tuple(_color(color) for color in selected)
            else:
                raise ValueError("confirm_if_accepted_by must be ANY or a nonempty ordered player array")
            arguments = {key: value for key, value in arguments.items() if key != "confirm_if_accepted_by"}
    if shared and tool in {"accept_offer", "reject_offer", "counter_offer", "confirm_trade", "cancel_trade"}:
        arguments = _shared_trade_arguments(context, tool, arguments)
    actions = [
        (index, action)
        for index, action in enumerate(context.legal_actions)
        if action.action_type == _TOOL_TYPES[tool] and action.color == context.actor
    ]
    if not actions:
        raise ValueError(f"Tool {tool} is unavailable in the current {context.phase} decision; check phase, holdings and pending actions")
    parameters = {"confirm_if_accepted_by": priority} if priority is not None else {}
    if tool in {"build_settlement", "upgrade_city", "build_road", "move_robber", "play_knight"}:
        category = (
            "edge"
            if tool == "build_road"
            else ("tile" if tool in {"move_robber", "play_knight"} else "node")
        )
        _require_arguments(arguments, category)
        token = arguments[category]
        spatial = _spatial_values(context, category)
        if not isinstance(token, str) or token not in spatial:
            raise ValueError(f"Invalid {category} token or unavailable board location: {token!r}")
        value = spatial[token]
        if tool == "play_knight":
            parameters["knight_destination"] = value
            actions = [(i, a) for i, a in actions if a.value is None]
        elif tool == "build_road":
            actions = [(i, a) for i, a in actions if canonical_edge(a.value) == value]
        else:
            actions = [(i, a) for i, a in actions if a.value == value]
    elif tool == "steal_from":
        _require_arguments(arguments, "player")
        value = (_color(arguments["player"]), None)
        actions = [(i, a) for i, a in actions if a.value == value]
    elif tool == "play_monopoly":
        _require_arguments(arguments, "resource")
        resource = _resource(arguments["resource"])
        actions = [(i, a) for i, a in actions if a.value == resource]
    elif tool in {"play_year_of_plenty", "discard"}:
        field = "take" if tool == "play_year_of_plenty" else "cards"
        _require_arguments(arguments, field)
        counts = _bundle(arguments[field])
        total = sum(counts)
        if tool == "play_year_of_plenty":
            if total not in {len(a.value) for _, a in actions} or not 1 <= total <= 2:
                raise ValueError("take must have a card count available in the Year of Plenty menu")
            actions = [(i, a) for i, a in actions if _card_counts(a.value) == counts]
        else:
            if total != context.discard_count:
                raise ValueError(f"Discard exactly {context.discard_count} cards")
            if any(
                count > context.observation.my_resources.get(resource, 0)
                for resource, count in zip(RESOURCE_NAMES, counts)
            ):
                raise ValueError("Cannot discard cards you do not hold")
            actions = [
                (i, a) for i, a in actions if a.value is None or _card_counts(a.value) == counts
            ]
            # Expand only after checking the required cardinality and actual hand.
            parameters["discard_cards"] = tuple(
                resource for resource, count in zip(RESOURCE_NAMES, counts) for _ in range(count)
            )
    elif tool == "maritime_trade":
        _require_arguments(arguments, "give", "receive")
        give, receive = _bundle(arguments["give"]), _bundle(arguments["receive"])
        if sum(count > 0 for count in give) != 1 or sum(give) not in {2, 3, 4}:
            raise ValueError("Maritime give must be one resource at its exact best port/bank rate")
        if sum(receive) != 1 or any(g and r for g, r in zip(give, receive)):
            raise ValueError("Maritime receive must be one card of a different resource")
        actions = [
            (i, a)
            for i, a in actions
            if _card_counts(a.value[:4]) == give and _card_counts(a.value[4:]) == receive
        ]
    elif tool in {"offer_trade", "counter_offer"}:
        required = (
            ("offer_id", "give", "receive") if tool == "counter_offer" else ("give", "receive")
        )
        _require_arguments(arguments, *required, optional=("give_any", "receive_any", "audience"))
        parent = _offer_id(arguments["offer_id"]) if tool == "counter_offer" else None
        audience = (
            frozenset({context.observation.turn_player_color})
            if parent is not None
            else frozenset(context.observation.opponent_resource_counts)
        )
        if "audience" in arguments:
            selected = arguments["audience"]
            if not isinstance(selected, list) or not selected:
                raise ValueError("audience must be a nonempty array of player colors")
            audience = frozenset(_color(color) for color in selected)
            if len(audience) != len(selected):
                raise ValueError("audience must not contain duplicate player colors")
        offer = TradeOffer(
            offered_by=context.actor,
            audience=audience,
            give=_bundle(arguments["give"]),
            receive=_bundle(arguments["receive"]),
            give_any=arguments.get("give_any", 0),
            receive_any=arguments.get("receive_any", 0),
            parent_offer_id=parent,
        )
        actions = [
            (i, a)
            for i, a in actions
            if (
                isinstance(a.value, str)
                and "audience" not in arguments
                and (parent is None or _counter_parent(a.value) == parent)
            )
            or (
                isinstance(a.value, TradeOffer)
                and a.value.parent_offer_id == parent
                and ("audience" not in arguments or a.value.audience == audience)
                and a.value.give == offer.give
                and a.value.receive == offer.receive
                and a.value.give_any == offer.give_any
                and a.value.receive_any == offer.receive_any
            )
        ]
        if "audience" not in arguments:
            actions = [(i, a) for i, a in actions if isinstance(a.value, str)] or actions
        parameters["trade_offer"] = offer
    elif tool in {"accept_offer", "reject_offer", "confirm_trade", "cancel_trade"}:
        required = ("offer_id", "counterparty") if tool == "confirm_trade" else ("offer_id",)
        _require_arguments(arguments, *required)
        value = _offer_id(arguments["offer_id"])
        if tool == "confirm_trade":
            value = TradeCandidate(
                value, context.observation.turn_player_color, _color(arguments["counterparty"])
            )
        actions = [(i, a) for i, a in actions if a.value == value]
    else:
        _require_arguments(arguments)
        actions = [(i, a) for i, a in actions if a.value is None]

    if not actions:
        hint = {
            "build_settlement": "Use an empty vertex with no adjacent building; outside setup it must connect to your road and cost WOOD, BRICK, SHEEP, WHEAT.",
            "build_road": "Use an unused edge connected to your network without extending through an opponent building; setup roads must touch the just-placed settlement. Paid roads cost WOOD and BRICK.",
            "upgrade_city": "Choose your own settlement and hold 2 WHEAT and 3 ORE, with a city piece available.",
            "maritime_trade": "Bank trade: give exactly your rate for that resource (4 by default, 3 with a 3:1 port, 2 with its 2:1 port) from cards you hold, for 1 card of a different resource the bank still has.",
            "steal_from": "The victim must have resource cards and a building adjacent to the current robber tile.",
        }.get(tool, "Check the current phase, exact terms, holdings and pending action; a trade may no longer be actionable.")
        raise ValueError(f"No legal {tool} action matches these arguments. {hint}")
    if len(actions) != 1:
        raise ValueError(f"Ambiguous {tool}: arguments must identify one exact menu action")
    index, action = actions[0]
    if isinstance(action.value, TradeOffer):
        parameters.pop("trade_offer", None)  # Preserve concrete menu terms and audience.
    choice = PlayerChoice(action_index=index, **parameters)
    resolved = action_from_choice(context, choice)
    if isinstance(resolved.value, TradeOffer):
        offer = resolved.value
        hand = tuple(context.observation.my_resources.get(r, 0) for r in RESOURCE_NAMES)
        if (
            any(g > h for g, h in zip(offer.give, hand))
            or sum(hand) - sum(offer.give) < offer.give_any
        ):
            raise ValueError("Cannot offer cards you do not hold")
        window = context.observation.trade_window
        if offer.parent_offer_id is not None or (
            window is not None and window.status.value == "open"
        ):
            if window is None:
                raise ValueError("Counteroffer requires an active trade window")
            window.validate_offer(offer)
    return choice


def render_action_tools(context: PlayerContext, *, shared: bool = False) -> str:
    """Compact available tools and concrete terms, without numbered menu rows."""
    if shared:
        return SHARED_ACTION_TOOLS
    lines = []
    for tool, action_type in _TOOL_TYPES.items():
        actions = [
            action
            for action in context.legal_actions
            if action.action_type == action_type and action.color == context.actor
        ]
        if not actions:
            continue
        if tool in {"build_settlement", "upgrade_city", "build_road", "move_robber", "play_knight"}:
            category = (
                "edge"
                if tool == "build_road"
                else ("tile" if tool in {"move_robber", "play_knight"} else "node")
            )
            spatial = _spatial_values(context, category)
            values = [canonical_edge(a.value) if category == "edge" else a.value for a in actions]
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
                for r in RESOURCE_NAMES
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
                    parents = sorted({_counter_parent(a.value) for a in parameterized} - {None})
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
                descriptor = {
                    field: payload[field]
                    for field in ("give", "receive", "give_any", "receive_any", "audience")
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
            candidates = {}
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


SHARED_ACTION_TOOLS = """Tool signatures (stable definitions, not a list of currently legal moves):
build_settlement(node); build_road(edge); upgrade_city(node)
roll_dice(); end_turn(); buy_development_card()
play_knight(tile): play Knight and move robber; a winning Largest Army ends play before movement.
move_robber(tile): pending robber movement; steal_from(player): subsequent victim choice, random card.
play_road_building(): subsequent build_road calls place free roads.
play_year_of_plenty(take): take two bank cards, or one if only one remains.
play_monopoly(resource); maritime_trade(give, receive): the BANK TRADE. Always available on your turn after rolling, no partner, no negotiation: give 4 of one resource for 1 of any other card the bank still holds. A 3:1 port lowers your rate to 3, a matching 2:1 port to 2. Your exact current rates are listed under YOUR RESOURCES.
discard(cards): discard the required count from your hand.
offer_trade(give, receive, give_any=0, receive_any=0, confirm_if_accepted_by optional): offer to other seats. Never pass a player argument; the offer goes out table-wide. Omit confirmation to probe and regain control after responses. For exact terms only, confirm_if_accepted_by is a nonempty ordered array of distinct audience colors (e.g. ["BLUE","RED"]) or "ANY". This is YOUR one-shot proposer authorization to confirm the exact original offer after the first complete simultaneous response batch. Select the first willing listed player; ANY uses engine seat/turn order, never response speed. Any counteroffer in that response window pauses even if someone accepts the original; nobody permitted accepts also pauses. Stale/withdrawn/expired offers or invalid hands/legality pause without fallback. Authorization is consumed once; it never applies to later offers or rounds.
accept_offer(player, give, receive, give_any=0, receive_any=0)
reject_offer(player, give, receive, give_any=0, receive_any=0)
confirm_trade(player, give, receive); cancel_trade(player, give, receive, give_any=0, receive_any=0)
counter_offer(player, original, proposed): original and proposed each contain give/receive and optional give_any/receive_any.
Every give/receive is from YOUR acting perspective; player is the other seat. Cancel withdraws your whole matching offer. Exact terms must identify one active offer; stale or ambiguous matches fail. Acceptance is non-binding; only turn-player confirmation transfers cards. Wildcards are proposals only, requiring exact terms before execution. Give only held cards.
Resource maps use WOOD, BRICK, SHEEP, WHEAT, ORE with positive integer counts; omitted resources are zero. Wildcard counts are non-negative integers. Spatial tokens are literal, case-sensitive <N00>, <E00_01> (canonical endpoints), <T00>. Empty signatures use {}. All calls undergo engine validation."""

_REVERSE_TOOL_TYPES = {action_type: tool for tool, action_type in _TOOL_TYPES.items()}


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
            _REVERSE_TOOL_TYPES[action.action_type]
            for action in context.legal_actions
            if action.color == context.actor and action.action_type in _REVERSE_TOOL_TYPES
        }
    ))


def render_shared_legal_actions(context: PlayerContext) -> str:
    """Stable tool definitions plus the actor's exact currently-legal tool names.

    Shared-mode prompts otherwise list every tool signature without saying which
    moves are legal now, which invites off-turn actors to call end_turn as a
    pass. The appended line names only the tools with a matching legal action
    for this actor, and names the trade responder's situation explicitly.
    """
    tools = legal_tool_names(context)
    lines = [
        SHARED_ACTION_TOOLS,
        "",
        "YOUR CURRENTLY LEGAL TOOLS (call exactly one of these): "
        + (", ".join(tools) if tools else "(none)"),
    ]
    note = trade_responder_note(context)
    if note:
        lines.append(note)
    return "\n".join(lines)
