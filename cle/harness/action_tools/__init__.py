"""Semantic tools bound to a perspective-safe, authoritative engine menu.

Board strings are literal trained vocabulary, not XML. This package neither
parses the response envelope nor executes actions. Parameterized offers and
discards remain subject to strict engine admission after menu resolution.
"""

from cle.game_engine.board_tokens import canonical_edge, edge_token, node_token, tile_token
from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import (
    RESOURCE_NAMES,
    ResourceBundle,
    TradeCandidate,
    TradeOffer,
)
from cle.players.contracts import PlayerChoice, PlayerContext
from cle.players.validation import action_from_choice

from .arguments import (
    _bundle,
    _card_counts,
    _color,
    _offer_id,
    _require_arguments,
    _resource,
    _spatial_values,
)
from .definitions import _REVERSE_TOOL_TYPES, _TOOL_TYPES, SHARED_ACTION_TOOLS
from .parser import parse_tool_choice
from .rendering import (
    legal_tool_names,
    render_action_tools,
    render_shared_legal_actions,
    trade_responder_note,
)
from .trades import _counter_parent, _semantic_offer, _shared_trade_arguments

__all__ = [
    "RESOURCE_NAMES", "SHARED_ACTION_TOOLS", "ActionType", "Color", "PlayerChoice",
    "PlayerContext", "ResourceBundle", "TradeCandidate", "TradeOffer", "action_from_choice",
    "canonical_edge", "edge_token", "node_token", "tile_token", "parse_tool_choice",
    "render_action_tools", "legal_tool_names", "render_shared_legal_actions", "trade_responder_note",
    "_TOOL_TYPES", "_REVERSE_TOOL_TYPES", "_bundle", "_card_counts", "_color", "_offer_id",
    "_require_arguments", "_resource", "_spatial_values", "_counter_parent", "_semantic_offer",
    "_shared_trade_arguments",
]
