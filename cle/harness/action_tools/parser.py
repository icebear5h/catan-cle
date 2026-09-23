"""Resolve one semantic call against the authoritative indexed engine menu."""

from __future__ import annotations

from typing import cast

from cle.game_engine.models.coordinate_system import Coordinate
from cle.game_engine.trading import TradeCandidate, TradeOffer
from cle.harness import action_tools
from cle.players.contracts import PlayerChoice, PlayerContext
from cle.players.data import JsonValue

from .arguments import (
    ChoiceParameters,
    ConfirmationPriority,
    SpatialCategory,
)
from .trades import _offer_actions, _validate_offer


def parse_tool_choice(
    context: PlayerContext, tool: str, arguments: dict[str, JsonValue], *, shared: bool = False,
) -> PlayerChoice:
    """Resolve exactly one semantic call; invalid or ambiguous calls raise ValueError."""
    # Keep helper replacements at the historical module path live at call time.
    if not isinstance(tool, str) or tool not in action_tools._TOOL_TYPES:
        raise ValueError("Unknown action tool")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    priority: ConfirmationPriority | None = None
    if shared and tool == "offer_trade":
        action_tools._require_arguments(
            arguments, "give", "receive",
            optional=("give_any", "receive_any", "confirm_if_accepted_by", "player", "audience"),
        )
        # A targeted offer reaches only its audience, so only those seats are
        # prompted to respond. Omitting both keeps the table-wide default.
        if "player" in arguments:
            if "audience" in arguments:
                raise ValueError("Pass player (one color) or audience (color array), not both")
            arguments = {
                **{key: value for key, value in arguments.items() if key != "player"},
                "audience": [arguments["player"]],
            }
        if "confirm_if_accepted_by" in arguments:
            selected = arguments["confirm_if_accepted_by"]
            if selected == "ANY":
                priority = "ANY"
            elif isinstance(selected, list) and selected:
                priority = tuple(action_tools._color(color) for color in selected)
            else:
                raise ValueError("confirm_if_accepted_by must be ANY or a nonempty ordered player array")
            arguments = {key: value for key, value in arguments.items() if key != "confirm_if_accepted_by"}
    if shared and tool in {"accept_offer", "reject_offer", "counter_offer", "confirm_trade", "cancel_trade"}:
        arguments = action_tools._shared_trade_arguments(context, tool, arguments)
    actions = [
        (index, action)
        for index, action in enumerate(context.legal_actions)
        if action.action_type == action_tools._TOOL_TYPES[tool] and action.color == context.actor
    ]
    if not actions:
        raise ValueError(f"Tool {tool} is unavailable in the current {context.phase} decision; check phase, holdings and pending actions")
    parameters: ChoiceParameters = {"confirm_if_accepted_by": priority} if priority is not None else {}
    if tool in {"build_settlement", "upgrade_city", "build_road", "move_robber", "play_knight"}:
        category: SpatialCategory = (
            "edge"
            if tool == "build_road"
            else ("tile" if tool in {"move_robber", "play_knight"} else "node")
        )
        action_tools._require_arguments(arguments, category)
        token = arguments[category]
        spatial = action_tools._spatial_values(context, category)
        if not isinstance(token, str) or token not in spatial:
            raise ValueError(f"Invalid {category} token or unavailable board location: {token!r}")
        location = spatial[token]
        if tool == "play_knight":
            parameters["knight_destination"] = cast(Coordinate, location)
            actions = [(i, a) for i, a in actions if a.value is None]
        elif tool == "build_road":
            actions = [(i, a) for i, a in actions if action_tools.canonical_edge(a.value) == location]
        else:
            actions = [(i, a) for i, a in actions if a.value == location]
    elif tool == "steal_from":
        action_tools._require_arguments(arguments, "player")
        victim = (action_tools._color(arguments["player"]), None)
        actions = [(i, a) for i, a in actions if a.value == victim]
    elif tool == "play_monopoly":
        action_tools._require_arguments(arguments, "resource")
        resource = action_tools._resource(arguments["resource"])
        actions = [(i, a) for i, a in actions if a.value == resource]
    elif tool in {"play_year_of_plenty", "discard"}:
        field = "take" if tool == "play_year_of_plenty" else "cards"
        action_tools._require_arguments(arguments, field)
        counts = action_tools._bundle(arguments[field])
        total = sum(counts)
        if tool == "play_year_of_plenty":
            if total not in {len(a.value) for _, a in actions} or not 1 <= total <= 2:
                raise ValueError("take must have a card count available in the Year of Plenty menu")
            actions = [(i, a) for i, a in actions if action_tools._card_counts(a.value) == counts]
        else:
            if total != context.discard_count:
                raise ValueError(f"Discard exactly {context.discard_count} cards")
            if any(
                count > context.observation.my_resources.get(resource, 0)
                for resource, count in zip(action_tools.RESOURCE_NAMES, counts)
            ):
                raise ValueError("Cannot discard cards you do not hold")
            actions = [
                (i, a) for i, a in actions
                if a.value is None or action_tools._card_counts(a.value) == counts
            ]
            # Expand only after checking the required cardinality and actual hand.
            parameters["discard_cards"] = tuple(
                resource for resource, count in zip(action_tools.RESOURCE_NAMES, counts)
                for _ in range(count)
            )
    elif tool == "maritime_trade":
        action_tools._require_arguments(arguments, "give", "receive")
        give, receive = action_tools._bundle(arguments["give"]), action_tools._bundle(arguments["receive"])
        if sum(count > 0 for count in give) != 1 or sum(give) not in {2, 3, 4}:
            raise ValueError("Maritime give must be one resource at its exact best port/bank rate")
        if sum(receive) != 1 or any(g and r for g, r in zip(give, receive)):
            raise ValueError("Maritime receive must be one card of a different resource")
        actions = [
            (i, a)
            for i, a in actions
            if action_tools._card_counts(a.value[:4]) == give
            and action_tools._card_counts(a.value[4:]) == receive
        ]
    elif tool in {"offer_trade", "counter_offer"}:
        actions, offer = _offer_actions(context, tool, arguments, actions, shared=shared)
        parameters["trade_offer"] = offer
    elif tool in {"accept_offer", "reject_offer", "confirm_trade", "cancel_trade"}:
        required = ("offer_id", "counterparty") if tool == "confirm_trade" else ("offer_id",)
        action_tools._require_arguments(arguments, *required)
        offer_id = action_tools._offer_id(arguments["offer_id"])
        target: str | TradeCandidate = offer_id
        if tool == "confirm_trade":
            target = TradeCandidate(
                offer_id, context.observation.turn_player_color,
                action_tools._color(arguments["counterparty"])
            )
        actions = [(i, a) for i, a in actions if a.value == target]
    else:
        action_tools._require_arguments(arguments)
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
    resolved = action_tools.action_from_choice(context, choice)
    if isinstance(resolved.value, TradeOffer):
        _validate_offer(context, resolved.value)
    return choice
