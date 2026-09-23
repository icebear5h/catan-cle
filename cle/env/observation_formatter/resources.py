"""Private resource inventory, affordability, and shared bank/card readouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.enums import RESOURCES, ActionType
from cle.game_engine.models.map import CatanMap

if TYPE_CHECKING:
    from . import CatanObservationFormatter, Observation


def format_resources(
    self: CatanObservationFormatter, obs: Observation, *, shared: bool = False,
) -> str:
    """Format resources with building possibilities."""
    lines = ["YOUR RESOURCES:"]
    total = sum(obs.my_resources.values())
    if total == 0:
        lines.append("  No resources")
        if not shared:
            return "\n".join(lines)
    for resource in RESOURCES:
        resource_name = getattr(resource, 'name') if hasattr(resource, 'name') else str(resource)
        count = obs.my_resources.get(resource_name, 0)
        lines.append(f"  {resource_name}: {count}")
    lines.append(f"  Total: {total} cards")
    if shared:
        # Always show the bank rate, even when no partner offers a trade.
        owned_ports = set()
        my_nodes = set(obs.my_settlements) | set(obs.my_cities)
        board_map: CatanMap = obs.board_map
        for port_resource, node_ids in board_map.port_nodes.items():
            if my_nodes.intersection(node_ids):
                owned_ports.add(
                    getattr(port_resource, "name") if hasattr(port_resource, "name") else port_resource
                )
        base_rate = 3 if None in owned_ports else 4
        rates = []
        for resource in RESOURCES:
            name = getattr(resource, "name") if hasattr(resource, "name") else str(resource)
            if name in owned_ports:
                rates.append(f"{name} 2 (2:1 port)")
            elif base_rate == 3:
                rates.append(f"{name} 3 (3:1 port)")
            else:
                rates.append(f"{name} 4")
        lines.append(
            "  BANK TRADE (always available, no partner; call the maritime_trade tool): "
            f"give this many of one resource for 1 of any other: {', '.join(rates)}"
        )
        if total > 7:
            lines.append(
                f"  DISCARD EXPOSURE: {total} cards held; any 7 rolled costs you {total // 2} "
                "cards until you are at 7 or fewer"
            )
    affordable = self._get_affordable_buildings(obs.my_resources)
    if affordable:
        lines.append(f"  Can afford: {', '.join(affordable)}")
    total_dev = sum(obs.my_dev_cards.values())
    if total_dev > 0 or shared:
        lines.append(f"\n  Development cards: {total_dev}")
        for card_name, count in obs.my_dev_cards.items():
            if count > 0 or shared:
                status = ""
                if shared:
                    action_type = {
                        "KNIGHT": ActionType.PLAY_KNIGHT_CARD,
                        "ROAD_BUILDING": ActionType.PLAY_ROAD_BUILDING,
                        "MONOPOLY": ActionType.PLAY_MONOPOLY,
                        "YEAR_OF_PLENTY": ActionType.PLAY_YEAR_OF_PLENTY,
                    }.get(card_name)
                    playable = any(a.color == obs.my_color and a.action_type == action_type
                                   for a in obs.valid_actions)
                    status = " (passive VP; not played)" if card_name == "VICTORY_POINT" else (
                        " (playable now)" if playable else " (not playable now)"
                    )
                lines.append(f"    {card_name}: {count}{status}")
    return "\n".join(lines)


def get_affordable_buildings(resources: dict[str, int]) -> list[str]:
    """Determine what buildings can be afforded with current resources."""
    affordable = []
    if (resources.get('WOOD', 0) >= 1 and
        resources.get('BRICK', 0) >= 1 and
        resources.get('SHEEP', 0) >= 1 and
        resources.get('WHEAT', 0) >= 1):
        affordable.append('settlement')
    if (resources.get('WHEAT', 0) >= 2 and resources.get('ORE', 0) >= 3):
        affordable.append('city')
    if (resources.get('WOOD', 0) >= 1 and resources.get('BRICK', 0) >= 1):
        affordable.append('road')
    if (resources.get('SHEEP', 0) >= 1 and
        resources.get('WHEAT', 0) >= 1 and
        resources.get('ORE', 0) >= 1):
        affordable.append('dev card')
    return affordable
