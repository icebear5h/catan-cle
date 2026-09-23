"""Execute one Colonist row directly when no engine action matched it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ReplayPayload, ReplayRuntimeState

from . import builds, dev_cards, robber, trades
from .outcome import DirectContext

__all__ = ["handle_direct_execute"]

_BRANCHES: Final[dict[str, Callable[[DirectContext], ReplayPayload]]] = {
    "CLOSE_TRADE": trades.close_trade,
    "MARITIME_TRADE": trades.maritime_trade,
    "PLAY_MONOPOLY": dev_cards.announce_dev_card,
    "PLAY_YEAR_OF_PLENTY": dev_cards.announce_dev_card,
    "PLAY_KNIGHT_CARD": dev_cards.play_dev_card,
    "PLAY_ROAD_BUILDING": dev_cards.play_dev_card,
    "MONOPOLY_RESOURCE": dev_cards.monopoly_resource,
    "YEAR_OF_PLENTY_RESOURCES": dev_cards.year_of_plenty_resources,
    "DISCARD": robber.discard,
    "STEAL": robber.steal,
    "MOVE_ROBBER": robber.move_robber,
    "BUILD_ROAD": builds.build,
    "BUILD_SETTLEMENT": builds.build,
    "BUILD_CITY": builds.build,
    "BUY_DEVELOPMENT_CARD": dev_cards.buy_development_card,
    "END_TURN": builds.end_turn,
}


def handle_direct_execute(
    action_type: str | None,
    action_hint: ActionHint,
    state: ReplayRuntimeState,
) -> tuple[ReplayPayload | None, bool]:
    """Handle MARITIME_TRADE, dev card actions, builds, BUY_DEV_CARD, END_TURN.

    Returns (response_dict, handled) where handled=True means caller should return response_dict.
    """
    if action_type is None:
        return None, False  # Not handled
    branch = _BRANCHES.get(action_type)
    if branch is None:
        return None, False  # Not handled
    return branch(DirectContext(action_type, action_hint, state)), True
