"""Phase, initial snake order, visible score, and live roll-state descriptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import CatanObservationFormatter, Observation


def format_strategic_context(
    self: CatanObservationFormatter,
    obs: Observation,
    *,
    include_initial_placement_order: bool,
    shared: bool = False,
) -> str:
    """Format phase and score info."""
    lines = []
    lines.append(f"Phase: {obs.current_phase}")
    if obs.current_phase == "initial_placement":
        placed = len(obs.my_settlements)
        lines.append(f"Settlements placed: {placed}/2")
        if include_initial_placement_order:
            first_round = tuple(obs.turn_order)
            if not first_round or obs.my_color not in first_round:
                raise ValueError("Initial-placement observation has an invalid turn order")
            second_round = tuple(reversed(first_round))
            first_names = " -> ".join(self._color_name(color) for color in first_round)
            second_names = " -> ".join(self._color_name(color) for color in second_round)
            player_count = len(first_round)
            lines.extend((
                "Initial placement order (each settlement is immediately "
                "followed by that player's road):",
                f"  Round 1 (first settlement + road): {first_names}",
                f"  Round 2 (second settlement + road): {second_names}",
                "  Your positions: "
                f"round 1 = {first_round.index(obs.my_color) + 1}/{player_count}; "
                f"round 2 = {second_round.index(obs.my_color) + 1}/{player_count}.",
            ))
    actual_vp = getattr(obs, "my_actual_vp", None)
    if shared and actual_vp is not None:
        lines.append(f"Your actual VP: {actual_vp}/10 (public: {obs.my_vp})")
    else:
        lines.append(f"Your VP: {obs.my_vp}/10")
    has_rolled = getattr(obs, "turn_player_has_rolled", None)
    if shared and has_rolled is not None and obs.current_phase != "initial_placement":
        turn_player = "you" if obs.is_my_turn else self._color_name(obs.turn_player_color)
        if has_rolled:
            lines.append(
                f"Dice this turn: ALREADY ROLLED {obs.last_dice_roll} by {turn_player}. "
                "roll_dice is not available again until the next turn."
            )
        else:
            lines.append(f"Dice this turn: NOT ROLLED YET by {turn_player}.")
            if obs.last_dice_roll:
                lines.append(f"Previous turn's dice roll: {obs.last_dice_roll}")
    elif obs.last_dice_roll:
        lines.append(f"Last dice roll: {obs.last_dice_roll}")
    return "\n".join(lines)
