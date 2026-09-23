"""Game event logging and resource analysis functions.

Split from one module; every historical name stays importable from this package.
"""

from .analysis import (
    RollSnapshot,
    _log_table_talk,
    analyze_action,
    analyze_transitions,
    compare_and_log_resources,
    post_analyze_action,
    stamp_message_step_indexes,
)
from .normalization import (
    _canonical_trade_log_entry,
    _message_text,
    backfill_message_log_entries,
    normalize_game_log_entries,
    normalize_public_state_game_log,
)
from .players import (
    DevCardCounts,
    PlayerHand,
    get_player_dev_cards,
    get_player_hands,
    get_player_resources,
)
from .sink import GameLogSink, log_game_event
from .trade_format import (
    TRADE_ACTION_TYPES,
    _enum_value,
    _format_named_resources,
    format_action_for_display,
    format_trade_event,
    trade_action_payload,
)

__all__ = [
    "TRADE_ACTION_TYPES",
    "DevCardCounts",
    "GameLogSink",
    "PlayerHand",
    "RollSnapshot",
    "_canonical_trade_log_entry",
    "_enum_value",
    "_format_named_resources",
    "_log_table_talk",
    "_message_text",
    "analyze_action",
    "analyze_transitions",
    "backfill_message_log_entries",
    "compare_and_log_resources",
    "format_action_for_display",
    "format_trade_event",
    "get_player_dev_cards",
    "get_player_hands",
    "get_player_resources",
    "log_game_event",
    "normalize_game_log_entries",
    "normalize_public_state_game_log",
    "post_analyze_action",
    "stamp_message_step_indexes",
    "trade_action_payload",
]
