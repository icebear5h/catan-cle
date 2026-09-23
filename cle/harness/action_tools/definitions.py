"""Stable semantic tool names and model-facing signatures."""

from cle.game_engine.models.enums import ActionType

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
_REVERSE_TOOL_TYPES = {action_type: tool for tool, action_type in _TOOL_TYPES.items()}

SHARED_ACTION_TOOLS = """Tool signatures (stable definitions, not a list of currently legal moves):
build_settlement(node); build_road(edge); upgrade_city(node)
roll_dice(); end_turn(); buy_development_card()
play_knight(tile): play Knight and move robber; a winning Largest Army ends play before movement.
move_robber(tile): pending robber movement; steal_from(player): subsequent victim choice, random card.
play_road_building(): subsequent build_road calls place free roads.
play_year_of_plenty(take): take two bank cards, or one if only one remains.
play_monopoly(resource); maritime_trade(give, receive): the BANK TRADE. Always available on your turn after rolling, no partner, no negotiation: give 4 of one resource for 1 of any other card the bank still holds. A 3:1 port lowers your rate to 3, a matching 2:1 port to 2. Your exact current rates are listed under YOUR RESOURCES.
discard(cards): discard the required count from your hand.
offer_trade(give, receive, give_any=0, receive_any=0, player or audience optional, confirm_if_accepted_by optional): offer to other seats. Omit player and audience for a table-wide offer; player names one color (e.g. "BLUE") and audience an array of colors, and only those seats see and answer the offer. Omit confirmation to probe and regain control after responses. For exact terms only, confirm_if_accepted_by is a nonempty ordered array of distinct audience colors (e.g. ["BLUE","RED"]) or "ANY". This is YOUR one-shot proposer authorization to confirm the exact original offer after the first complete simultaneous response batch. Select the first willing listed player; ANY uses engine seat/turn order, never response speed. Any counteroffer in that response window pauses even if someone accepts the original; nobody permitted accepts also pauses. Stale/withdrawn/expired offers or invalid hands/legality pause without fallback. Authorization is consumed once; it never applies to later offers or rounds.
accept_offer(player, give, receive, give_any=0, receive_any=0)
reject_offer(player, give, receive, give_any=0, receive_any=0)
confirm_trade(player, give, receive); cancel_trade(player, give, receive, give_any=0, receive_any=0)
counter_offer(player, original, proposed): original and proposed each contain give/receive and optional give_any/receive_any.
Every give/receive is from YOUR acting perspective; player is the other seat. Cancel withdraws your whole matching offer. Exact terms must identify one active offer; stale or ambiguous matches fail. Acceptance is non-binding; only turn-player confirmation transfers cards. Wildcards are proposals only, requiring exact terms before execution. Give only held cards.
Resource maps use WOOD, BRICK, SHEEP, WHEAT, ORE with positive integer counts; omitted resources are zero. Wildcard counts are non-negative integers. Spatial tokens are literal, case-sensitive <N00>, <E00_01> (canonical endpoints), <T00>. Empty signatures use {}. All calls undergo engine validation."""
