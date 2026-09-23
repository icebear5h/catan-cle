"""Shapes of the Colonist archive fragments the replay decoder reads.

Every member is optional: Colonist streams delta-encoded state, so any given
event carries only the fields that changed.
"""

from __future__ import annotations

from typing import TypeAlias, TypedDict

from cle.game_engine.models.enums import FastResource

#: Colonist's per-player hands, as raw card-id lists straight from the archive.
ColonistHands: TypeAlias = dict[int, list[object]]


class ColonistTileState(TypedDict, total=False):
    """One land/water hex in ``mapState.tileHexStates``."""

    x: int
    y: int
    type: int
    diceNumber: int


class ColonistPortState(TypedDict, total=False):
    """One port edge in ``mapState.portEdgeStates``."""

    x: int
    y: int
    type: int


class ColonistMapState(TypedDict, total=False):
    """The ``mapState`` fragment of an initial state or event delta."""

    tileHexStates: dict[str, ColonistTileState]
    portEdgeStates: dict[str, ColonistPortState]


class ColonistInitialState(TypedDict, total=False):
    """The archive's ``initial_state`` blob."""

    mapState: ColonistMapState


class ActionHint(TypedDict, total=False):
    """One parsed Colonist row.

    The key set varies by ``type``. Fields the decoder computes itself carry
    concrete types; fields copied straight out of the Colonist archive stay
    ``object`` because the source is untrusted JSON, and consumers narrow them
    exactly where they always did.
    """

    index: int
    type: str
    player: object

    dice: tuple[int, ...]
    resource_payouts: dict[int, tuple[int, ...]]
    resource_payouts_complete: bool
    expected_resources: ColonistHands

    colonist_corner: int
    colonist_edge: int
    colonist_tile: object

    trade_id: object
    trade_num: int
    creator: object
    acceptor: object
    offered: tuple[int, ...]
    wanted: tuple[int, ...]
    given: tuple[int, ...]
    received: tuple[int, ...]
    trade_tuple: tuple[object, ...]
    is_flexible: bool
    is_counter_offer: bool
    counter_offer_to: object
    player_responses: dict[str, object]
    previous_response: object
    closed_trades: list["ActionHint"]
    trade_closures_preceded: bool
    reason: str

    cards: tuple[int, ...]
    card_type: str
    resources: list[FastResource]
    resource: FastResource | None
    amount: object
    victim: object
    stolen_resource: FastResource | None

    tile_info: dict[str, object]
    tile_index: object


class TradeLedgerRecord(TypedDict, total=False):
    """One live Colonist trade tracked by exact trade id."""

    trade_id: object
    trade_num: object
    creator: object
    offered: tuple[object, ...]
    wanted: tuple[object, ...]
    offered_any: object
    wanted_any: object
    is_flexible: bool
    is_counter_offer: bool
    counter_offer_to: object
    responses: dict[str, str]


class ResourceMismatch(TypedDict):
    """One player's engine-vs-Colonist hand difference at a replay step."""

    player_idx: int
    colonist_id: int
    engine_has: list[int]
    colonist_expects: list[int]
    diff: list[int]
