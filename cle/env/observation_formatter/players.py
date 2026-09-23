"""Public opponent buildings, counts, and achievements."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from . import CatanObservationFormatter, Observation


def format_opponents(
    self: CatanObservationFormatter, obs: Observation, *, shared: bool = False,
) -> str:
    """Format opponent state with threat assessment."""
    lines = ["OPPONENTS:"]
    for color in obs.opponent_vps.keys():
        color_str = color.name if hasattr(color, 'name') else str(color)
        vp = obs.opponent_vps[color]
        settlements = obs.opponent_settlements.get(color, [])
        cities = obs.opponent_cities.get(color, [])
        roads = len(obs.opponent_roads.get(color, []))
        resources = obs.opponent_resource_counts.get(color, 0)
        dev_cards = obs.opponent_dev_card_counts.get(color, 0)
        lines.append(
            f"  {color_str}: {vp} {'public VP' if shared else 'VP'} ({len(settlements)} settlements, {len(cities)} cities, "
            f"{roads} roads, {resources} resources, {dev_cards} dev cards)"
        )
        if obs.current_phase == "initial_placement" and (settlements or cities):
            placement_details = []
            for node_id in settlements:
                context = self._get_node_strategic_context(node_id, obs)
                placement_details.append(f"{self._format_node(node_id, self._node_coords)}: {context}")
            for node_id in cities:
                context = self._get_node_strategic_context(node_id, obs)
                placement_details.append(f"{self._format_node(node_id, self._node_coords)} (city): {context}")
            if placement_details:
                lines.append("    Placements:")
                for detail in placement_details:
                    lines.append(f"      - {detail}")
    if obs.longest_road_holder:
        holder_name = obs.longest_road_holder.name if hasattr(obs.longest_road_holder, 'name') else str(obs.longest_road_holder)
        lines.append(f"  Longest road: {holder_name} (+2 VP)")
    if obs.largest_army_holder:
        holder_name = obs.largest_army_holder.name if hasattr(obs.largest_army_holder, 'name') else str(obs.largest_army_holder)
        lines.append(f"  Largest army: {holder_name} (+2 VP)")
    return "\n".join(lines)


def color_name(color: Color | str) -> str:
    """Get string name from color."""
    if hasattr(color, 'value'):
        return str(color.value)
    return str(color)
