"""Colonist hand tracking used while decoding one replay archive.

Colonist streams delta-encoded ``playerStates``; the tracker keeps the running
hands so every parsed row can carry the expected state after it executes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from cle.game_engine.models.enums import FastResource
from cle.replay.colonist.constants import COLONIST_RES_TO_ENGINE
from cle.replay.colonist.types import ColonistHands
from cle.replay.contracts import as_mapping, list_field

__all__ = ["ResourceTracker", "infer_stolen_resource", "resource_count_delta"]


def resource_count_delta(
    before_cards: Iterable[object],
    after_cards: Iterable[object],
) -> dict[int, int]:
    """Count resource-card changes by Colonist resource id."""
    before_counts = {resource_id: 0 for resource_id in COLONIST_RES_TO_ENGINE}
    after_counts = {resource_id: 0 for resource_id in COLONIST_RES_TO_ENGINE}
    for card_id in before_cards:
        if isinstance(card_id, int) and card_id in before_counts:
            before_counts[card_id] += 1
    for card_id in after_cards:
        if isinstance(card_id, int) and card_id in after_counts:
            after_counts[card_id] += 1
    return {
        resource_id: after_counts[resource_id] - before_counts[resource_id]
        for resource_id in COLONIST_RES_TO_ENGINE
    }


def infer_stolen_resource(
    resources_before: ColonistHands,
    resources_after: ColonistHands,
    thief: object,
    victim: object,
) -> FastResource | None:
    """Infer the robbed card from exact before/after player resource state.

    Ids that are not Colonist player integers never matched a tracked hand, so
    they yielded empty deltas and no candidate; they return ``None`` here too.
    """
    if not isinstance(thief, int) or not isinstance(victim, int):
        return None
    thief_delta = resource_count_delta(
        resources_before.get(thief, []),
        resources_after.get(thief, []),
    )
    victim_delta = resource_count_delta(
        resources_before.get(victim, []),
        resources_after.get(victim, []),
    )
    candidates = [
        resource_id
        for resource_id in COLONIST_RES_TO_ENGINE
        if thief_delta.get(resource_id, 0) > 0 and victim_delta.get(resource_id, 0) < 0
    ]
    if len(candidates) == 1:
        return COLONIST_RES_TO_ENGINE[candidates[0]]
    return None


class ResourceTracker:
    """Colonist's running per-player hands, the source of truth for syncing."""

    def __init__(self, player_ids: Iterable[int | str] | None) -> None:
        self.expected_player_ids: frozenset[int] | None = (
            frozenset(int(player_id) for player_id in player_ids)
            if player_ids
            else None
        )
        self.hands: ColonistHands = {}

    def snapshot(self) -> ColonistHands:
        """Create a deep copy of current colonist_resources."""
        return {k: list(v) for k, v in self.hands.items()}

    def absorb(self, player_states: Mapping[str, object]) -> None:
        """Apply one event's ``playerStates`` delta to the tracked hands."""
        for player_id_str, pstate in player_states.items():
            player_state = as_mapping(pstate, f"playerStates.{player_id_str}")
            if "resourceCards" in player_state:
                hand = as_mapping(
                    player_state["resourceCards"],
                    f"playerStates.{player_id_str}.resourceCards",
                )
                self.hands[int(player_id_str)] = list_field(
                    hand, "cards", f"playerStates.{player_id_str}.resourceCards.cards"
                )

    def infer_roll_resource_payouts(
        self,
        resources_before: ColonistHands,
        resources_after: ColonistHands,
        roll_total: int,
    ) -> tuple[dict[int, tuple[int, ...]], bool]:
        """Return public positive roll deltas without exposing full hands."""
        payouts: dict[int, tuple[int, ...]] = {}
        complete = (
            self.expected_player_ids is not None
            and set(resources_before) == self.expected_player_ids
            and set(resources_after) == self.expected_player_ids
        )

        for player_id, after_cards in resources_after.items():
            if player_id not in resources_before:
                complete = False
                continue

            deltas = resource_count_delta(resources_before[player_id], after_cards)
            if any(delta < 0 for delta in deltas.values()):
                return {}, False
            if roll_total == 7 and any(deltas.values()):
                return {}, False

            payout = tuple(
                deltas[resource_id]
                for resource_id in sorted(COLONIST_RES_TO_ENGINE)
            )
            if any(payout):
                payouts[player_id] = payout

        return payouts, complete
