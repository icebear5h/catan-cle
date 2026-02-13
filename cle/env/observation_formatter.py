"""
Text-based observation formatter for LLM agents.

Converts Catanatron game state into semantic text descriptions
following the FLE (Factorio Learning Environment) pattern.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from engine.state_functions import (
    get_player_freqdeck,
    get_player_buildings,
    get_visible_victory_points,
    get_dev_cards_in_hand,
    player_key,
    get_longest_road_length,
)
from engine.models.enums import (
    RESOURCES,
    SETTLEMENT,
    CITY,
    ROAD,
    Action,
)
from engine.models.player import Color


@dataclass
class CatanObservation:
    """
    Structured observation of Catan game state.

    This intermediate representation makes it easier to format
    semantic text for the LLM.
    """
    # Player info
    my_color: Color

    # Board state
    my_settlements: List[int]  # node IDs
    my_cities: List[int]
    my_roads: List[tuple]  # edge tuples

    opponent_settlements: Dict[Color, List[int]]
    opponent_cities: Dict[Color, List[int]]
    opponent_roads: Dict[Color, List[tuple]]

    # Resources and cards
    my_resources: Dict[str, int]  # resource name -> count
    my_dev_cards: Dict[str, int]  # dev card name -> count
    opponent_resource_counts: Dict[Color, int]  # total visible count

    # Game state
    current_turn: int
    current_phase: str  # initial_placement, main_game, discarding, moving_robber
    last_dice_roll: Optional[int]
    robber_position: Any  # coordinate

    # Victory points
    my_vp: int
    opponent_vps: Dict[Color, int]

    # Special achievements
    longest_road_holder: Optional[Color]
    largest_army_holder: Optional[Color]
    my_longest_road_length: int

    # Available actions
    valid_actions: List[Action]

    # Board map (for strategic context)
    board_map: Any  # CatanMap
    buildings_dict: Dict[int, tuple]  # node_id -> (color, building_type)

    # Trading state
    active_trades: Dict  # color -> {offered, wanted, acceptees, rejecters, ...}
    counter_offers: Dict  # color -> {offered, wanted, acceptees, ...}
    is_my_turn: bool  # Whether this player is the turn player
    turn_player_color: Color  # Who's turn it is

    # Event history since last observation
    recent_events: List[Action] = None

    def __post_init__(self):
        if self.recent_events is None:
            self.recent_events = []


@dataclass
class FormattedObservation:
    """
    Semantic text representation of game state for LLM.

    Following FLE pattern: structured data -> formatted text
    """
    raw_str: str  # Complete formatted observation
    board_state: str
    resources: str
    opponents: str
    valid_actions: str
    strategic_context: str
    trade_context: str  # Active trades and counter-offers


class CatanObservationFormatter:
    """
    Formats Catan observations into semantic text for LLMs.

    Inspired by Factorio Learning Environment's BasicObservationFormatter.
    Converts structured game data into rich text descriptions that help
    LLMs reason strategically.
    """

    def format(self, obs: CatanObservation) -> FormattedObservation:
        """
        Convert structured observation to semantic text.

        Args:
            obs: Structured observation data

        Returns:
            Formatted observation with semantic text fields
        """
        # Build node coordinate map once for the whole format pass
        self._node_coords = self._build_node_coordinate_map(obs.board_map)

        board_state = self._format_board_state(obs)
        resources = self._format_resources(obs)
        opponents = self._format_opponents(obs)
        valid_actions = self._format_valid_actions(obs)
        strategic_context = self._format_strategic_context(obs)
        trade_context = self._format_trade_context(obs)
        events_section = self._format_events(obs)

        # Only include trade section if there's active trading
        trade_section = f"\n\n{trade_context}" if trade_context else ""

        # Only include events section if there are recent events
        events_block = f"\n\n{events_section}" if events_section else ""

        raw_str = f"""
=== GAME STATE (Turn {obs.current_turn}) ===

{strategic_context}{events_block}

{board_state}

{resources}

{opponents}{trade_section}

{valid_actions}
        """.strip()

        return FormattedObservation(
            raw_str=raw_str,
            board_state=board_state,
            resources=resources,
            opponents=opponents,
            valid_actions=valid_actions,
            strategic_context=strategic_context,
            trade_context=trade_context,
        )

    def _format_board_state(self, obs: CatanObservation) -> str:
        """Format board state with strategic context."""
        lines = ["YOUR BUILDINGS:"]

        if not obs.my_settlements and not obs.my_cities:
            lines.append("  No buildings yet (initial placement phase)")
            return "\n".join(lines)

        # Format settlements with resource access
        if obs.my_settlements:
            lines.append(f"  Settlements ({len(obs.my_settlements)}):")
            for node_id in obs.my_settlements:
                context = self._get_node_strategic_context(node_id, obs)
                lines.append(f"    - {self._format_node(node_id, self._node_coords)}: {context}")

        # Format cities
        if obs.my_cities:
            lines.append(f"  Cities ({len(obs.my_cities)}):")
            for node_id in obs.my_cities:
                context = self._get_node_strategic_context(node_id, obs)
                lines.append(f"    - {self._format_node(node_id, self._node_coords)}: {context}")

        # Format roads
        if obs.my_roads:
            lines.append(f"  Roads ({len(obs.my_roads)}): {len(obs.my_roads)} connections")
            if obs.my_longest_road_length > 0:
                lines.append(f"    Longest road length: {obs.my_longest_road_length}")

        return "\n".join(lines)

    def _get_node_strategic_context(self, node_id: int, obs: CatanObservation) -> str:
        """Get strategic context for a node (ports, tiles, resources)."""
        parts = []

        # Get adjacent tiles and their resources
        if node_id in obs.board_map.adjacent_tiles:
            tiles = obs.board_map.adjacent_tiles[node_id]
            resources_with_numbers = []
            total_pips = 0

            for tile in tiles:
                if tile.resource is not None and tile.number is not None:
                    resource_name = str(tile.resource)
                    pips = self._number_to_pips(tile.number)
                    total_pips += pips
                    resources_with_numbers.append(
                        f"{resource_name}(dice={tile.number},pips={pips})"
                    )

            if resources_with_numbers:
                parts.append(", ".join(resources_with_numbers))
                parts.append(f"{total_pips} pips")

        # Check for ports
        port_type = None
        for resource, nodes in obs.board_map.port_nodes.items():
            if node_id in nodes:
                if resource is None:
                    port_type = "3:1 port"
                else:
                    port_type = f"{resource} 2:1 port"
                break

        if port_type:
            parts.append(port_type)

        return " | ".join(parts) if parts else "no production"

    def _check_nearby_opponents(self, node_id: int, obs: CatanObservation) -> str:
        """Check if any opponent settlements/cities are near this node."""
        nearby = []

        # Check opponent settlements
        for color, settlements in obs.opponent_settlements.items():
            for settlement_node in settlements:
                # In Catan, settlements must be at least 2 edges apart
                # For simplicity, we'll check if they're in the same local area
                if abs(settlement_node - node_id) <= 5:  # Rough proximity check
                    color_str = color.name if hasattr(color, 'name') else str(color)
                    nearby.append(f"{color_str} settlement")

        # Check opponent cities
        for color, cities in obs.opponent_cities.items():
            for city_node in cities:
                if abs(city_node - node_id) <= 5:
                    color_str = color.name if hasattr(color, 'name') else str(color)
                    nearby.append(f"{color_str} city")

        return ", ".join(nearby) if nearby else ""

    def _number_to_pips(self, number: int) -> int:
        """Convert dice number to pips (dots showing probability)."""
        pips_map = {
            2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
            8: 5, 9: 4, 10: 3, 11: 2, 12: 1
        }
        return pips_map.get(number, 0)

    def _format_resources(self, obs: CatanObservation) -> str:
        """Format resources with building possibilities."""
        lines = ["YOUR RESOURCES:"]

        total = sum(obs.my_resources.values())
        if total == 0:
            lines.append("  No resources")
            return "\n".join(lines)

        # List each resource
        for resource in RESOURCES:
            resource_name = resource.name if hasattr(resource, 'name') else str(resource)
            count = obs.my_resources.get(resource_name, 0)
            if count > 0:
                lines.append(f"  {resource_name}: {count}")

        lines.append(f"  Total: {total} cards")

        # Add what you can afford
        affordable = self._get_affordable_buildings(obs.my_resources)
        if affordable:
            lines.append(f"  Can afford: {', '.join(affordable)}")

        # Dev cards
        total_dev = sum(obs.my_dev_cards.values())
        if total_dev > 0:
            lines.append(f"\n  Development cards: {total_dev}")
            for card_name, count in obs.my_dev_cards.items():
                if count > 0:
                    lines.append(f"    {card_name}: {count}")

        return "\n".join(lines)

    def _get_affordable_buildings(self, resources: Dict[str, int]) -> List[str]:
        """Determine what buildings can be afforded with current resources."""
        affordable = []

        # Settlement: wood, brick, sheep, wheat
        if (resources.get('WOOD', 0) >= 1 and
            resources.get('BRICK', 0) >= 1 and
            resources.get('SHEEP', 0) >= 1 and
            resources.get('WHEAT', 0) >= 1):
            affordable.append('settlement')

        # City: 2 wheat, 3 ore
        if (resources.get('WHEAT', 0) >= 2 and
            resources.get('ORE', 0) >= 3):
            affordable.append('city')

        # Road: wood, brick
        if (resources.get('WOOD', 0) >= 1 and
            resources.get('BRICK', 0) >= 1):
            affordable.append('road')

        # Dev card: sheep, wheat, ore
        if (resources.get('SHEEP', 0) >= 1 and
            resources.get('WHEAT', 0) >= 1 and
            resources.get('ORE', 0) >= 1):
            affordable.append('dev card')

        return affordable

    def _format_opponents(self, obs: CatanObservation) -> str:
        """Format opponent state with threat assessment."""
        lines = ["OPPONENTS:"]

        for color in obs.opponent_vps.keys():
            color_str = color.name if hasattr(color, 'name') else str(color)
            vp = obs.opponent_vps[color]
            settlements = obs.opponent_settlements.get(color, [])
            cities = obs.opponent_cities.get(color, [])
            roads = len(obs.opponent_roads.get(color, []))
            resources = obs.opponent_resource_counts.get(color, 0)

            lines.append(
                f"  {color_str}: {vp} VP ({len(settlements)} settlements, {len(cities)} cities, "
                f"{roads} roads, {resources} resources)"
            )

            # During initial placement, show where opponents have placed
            if obs.current_phase == "initial_placement" and (settlements or cities):
                placement_details = []
                for node_id in settlements:
                    context = self._get_node_strategic_context(node_id, obs)
                    placement_details.append(f"{self._format_node(node_id, self._node_coords)}: {context}")
                for node_id in cities:
                    context = self._get_node_strategic_context(node_id, obs)
                    placement_details.append(f"{self._format_node(node_id, self._node_coords)} (city): {context}")

                if placement_details:
                    lines.append(f"    Placements:")
                    for detail in placement_details:
                        lines.append(f"      - {detail}")

        # Special achievements
        if obs.longest_road_holder:
            holder_name = obs.longest_road_holder.name if hasattr(obs.longest_road_holder, 'name') else str(obs.longest_road_holder)
            lines.append(f"  Longest road: {holder_name} (+2 VP)")

        if obs.largest_army_holder:
            holder_name = obs.largest_army_holder.name if hasattr(obs.largest_army_holder, 'name') else str(obs.largest_army_holder)
            lines.append(f"  Largest army: {holder_name} (+2 VP)")

        return "\n".join(lines)

    def _format_valid_actions(self, obs: CatanObservation) -> str:
        """Format valid actions with strategic implications."""
        from engine.models.enums import ActionType

        lines = ["VALID ACTIONS:"]

        if not obs.valid_actions:
            lines.append("  No actions available (waiting for turn)")
            return "\n".join(lines)

        # Group actions by type
        action_groups = {}
        for action in obs.valid_actions:
            action_type = action.action_type if hasattr(action, 'action_type') else str(type(action))
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append(action)

        # Format each group
        for action_type, actions in action_groups.items():
            type_name = action_type.name if hasattr(action_type, 'name') else str(action_type)
            lines.append(f"  {type_name}: {len(actions)} options")

            # For settlement placement (especially initial placement), show ALL options with context
            # For other actions, show first few examples
            is_settlement_action = action_type == ActionType.BUILD_SETTLEMENT if hasattr(action_type, 'name') else False

            if is_settlement_action and obs.current_phase == "initial_placement":
                # Show ALL settlement options during initial placement with full context
                for i, action in enumerate(actions):
                    action_desc = self._format_single_action(action, obs)
                    lines.append(f"    {i}. {action_desc}")
            else:
                # Show first few examples for other actions
                for i, action in enumerate(actions[:5]):
                    action_desc = self._format_single_action(action, obs)
                    lines.append(f"    {i}. {action_desc}")

                if len(actions) > 5:
                    lines.append(f"    ... and {len(actions) - 5} more")

        return "\n".join(lines)

    def _format_single_action(self, action: Action, obs: CatanObservation) -> str:
        """Format a single action with context."""
        from engine.models.enums import ActionType

        # For settlement building actions, show strategic context
        if hasattr(action, 'action_type') and action.action_type == ActionType.BUILD_SETTLEMENT:
            if hasattr(action, 'value'):
                node_id = action.value
                context = self._get_node_strategic_context(node_id, obs)
                node_str = self._format_node(node_id, self._node_coords)

                # Check if any opponent is nearby
                nearby_opponents = self._check_nearby_opponents(node_id, obs)

                if nearby_opponents:
                    return f"Build settlement at {node_str}: {context} | Near: {nearby_opponents}"
                else:
                    return f"Build settlement at {node_str}: {context}"

        # For road building actions with spatial context
        if hasattr(action, 'action_type') and action.action_type == ActionType.BUILD_ROAD:
            if hasattr(action, 'value') and isinstance(action.value, tuple):
                edge = action.value
                edge_str = self._format_edge(edge, self._node_coords)
                return f"Build road on {edge_str}"

        # Default: just convert to string
        return str(action)

    def _format_trade_context(self, obs: CatanObservation) -> str:
        """Format active trades and counter-offers for LLM decision-making."""
        if not obs.active_trades and not obs.counter_offers:
            return ""

        lines = ["TRADING:"]

        # Format active trades (offers on the table)
        if obs.active_trades:
            lines.append("  Active Trade Offers:")
            for creator_color, trade_info in obs.active_trades.items():
                offered = trade_info.get('offered', (0,0,0,0,0))
                wanted = trade_info.get('wanted', (0,0,0,0,0))
                offered_any = trade_info.get('offered_any', 0)
                wanted_any = trade_info.get('wanted_any', 0)
                acceptees = trade_info.get('acceptees', set())
                rejecters = trade_info.get('rejecters', set())

                offered_str = self._format_resource_tuple_with_any(offered, offered_any)
                wanted_str = self._format_resource_tuple_with_any(wanted, wanted_any)

                has_wildcards = offered_any > 0 or wanted_any > 0

                creator_name = self._color_name(creator_color)
                is_mine = creator_color == obs.my_color

                if is_mine:
                    line = f"    YOUR OFFER: Giving {offered_str} for {wanted_str}"
                else:
                    line = f"    {creator_name}'s offer: Giving {offered_str} for {wanted_str}"

                # Show responses
                responses = []
                if acceptees:
                    accepted_names = [self._color_name(c) for c in acceptees]
                    responses.append(f"accepted by {', '.join(accepted_names)}")
                if rejecters:
                    rejected_names = [self._color_name(c) for c in rejecters]
                    responses.append(f"rejected by {', '.join(rejected_names)}")

                if responses:
                    line += f" [{'; '.join(responses)}]"
                elif has_wildcards:
                    line += " [has wildcards - must counter-offer to specify resources]"
                else:
                    line += " [awaiting responses]"

                lines.append(line)

                # If it's my offer and there are acceptees, explain I can confirm
                if is_mine and acceptees:
                    lines.append(f"      -> You can CONFIRM_TRADE to complete with one of: {', '.join(self._color_name(c) for c in acceptees)}")

        # Format counter-offers (offers directed at the turn player)
        if obs.counter_offers:
            lines.append("  Counter-Offers (to turn player):")
            for creator_color, counter_info in obs.counter_offers.items():
                offered = counter_info.get('offered', (0,0,0,0,0))
                wanted = counter_info.get('wanted', (0,0,0,0,0))
                acceptees = counter_info.get('acceptees', set())

                offered_str = self._format_resource_tuple(offered)
                wanted_str = self._format_resource_tuple(wanted)

                creator_name = self._color_name(creator_color)
                is_mine = creator_color == obs.my_color

                if is_mine:
                    line = f"    YOUR COUNTER: Offering {offered_str} for {wanted_str}"
                else:
                    line = f"    {creator_name}'s counter: Offering {offered_str} for {wanted_str}"

                # Show who else has joined this counter
                if acceptees:
                    joined_names = [self._color_name(c) for c in acceptees]
                    line += f" [also offered by: {', '.join(joined_names)}]"

                lines.append(line)

                # If I'm the turn player, explain I can accept this
                if obs.is_my_turn and not is_mine:
                    available_partners = [creator_name] + [self._color_name(c) for c in acceptees]
                    lines.append(f"      -> You can ACCEPT_COUNTER_OFFER to trade with: {', '.join(available_partners)}")

        # Add guidance based on role
        if obs.is_my_turn:
            if obs.active_trades and any(t.get('acceptees') for t in obs.active_trades.values()):
                pass  # Already explained CONFIRM_TRADE above
            if not obs.active_trades:
                lines.append("  (You can OFFER_TRADE to propose a trade)")
        else:
            # Not my turn - I can respond to trades or make counters
            if obs.active_trades:
                lines.append("  (You can ACCEPT_TRADE, REJECT_TRADE, or COUNTER_OFFER)")
            if obs.counter_offers:
                lines.append("  (You can JOIN_COUNTER_OFFER to match an existing counter)")

        return "\n".join(lines)

    def _format_resource_tuple(self, resources: tuple) -> str:
        """Format a 5-tuple of resources into readable string."""
        resource_names = ['WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE']
        parts = []
        for i, count in enumerate(resources):
            if count > 0:
                parts.append(f"{count} {resource_names[i]}")
        return ", ".join(parts) if parts else "nothing"

    def _format_resource_tuple_with_any(self, resources: tuple, any_count: int) -> str:
        """Format a 5-tuple of resources with optional wildcard count."""
        resource_names = ['WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE']
        parts = []
        for i, count in enumerate(resources):
            if count > 0:
                parts.append(f"{count} {resource_names[i]}")
        if any_count > 0:
            parts.append(f"{any_count} ANY")
        return ", ".join(parts) if parts else "nothing"

    def _color_name(self, color) -> str:
        """Get string name from color."""
        if hasattr(color, 'value'):
            return str(color.value)
        return str(color)

    def _build_node_coordinate_map(self, board_map) -> Dict[int, str]:
        """Build a mapping from node_id to integer cube coordinate string.

        Each node sits at the vertex of exactly 3 hex tiles (land/port/water).
        Summing their integer cube coordinates gives a unique integer cube
        coordinate for every node, e.g. '(1, 1, -2)'.
        """
        from collections import defaultdict
        node_coords_accum: Dict[int, List[tuple]] = defaultdict(list)

        for coord, tile in board_map.tiles.items():
            if hasattr(tile, 'nodes'):
                for node_id in tile.nodes.values():
                    node_coords_accum[node_id].append(coord)

        result = {}
        for node_id, coords in node_coords_accum.items():
            x = sum(c[0] for c in coords)
            y = sum(c[1] for c in coords)
            z = sum(c[2] for c in coords)
            result[node_id] = f"({x}, {y}, {z})"

        return result

    def _format_node(self, node_id: int, node_coords: Dict[int, str]) -> str:
        """Format a node_id with its coordinate."""
        coord = node_coords.get(node_id, "")
        if coord:
            return f"node {node_id} {coord}"
        return f"node {node_id}"

    def _format_edge(self, edge, node_coords: Dict[int, str]) -> str:
        """Format an edge (node_id, node_id) with coordinates."""
        if isinstance(edge, tuple) and len(edge) == 2:
            n1 = self._format_node(edge[0], node_coords)
            n2 = self._format_node(edge[1], node_coords)
            return f"edge ({n1} -- {n2})"
        return f"edge {edge}"

    def _format_events(self, obs: CatanObservation) -> str:
        """Format recent game events since last observation."""
        if not obs.recent_events:
            return ""

        node_coords = self._node_coords
        lines = ["RECENT EVENTS:"]

        for action in obs.recent_events:
            color_str = action.color.name if hasattr(action.color, 'name') else str(action.color)
            at = action.action_type
            val = action.value

            if at.name == "ROLL":
                if isinstance(val, tuple) and len(val) == 2:
                    lines.append(f"  {color_str}: ROLL {val[0]+val[1]} ({val[0]}+{val[1]})")
                else:
                    lines.append(f"  {color_str}: ROLL {val}")

            elif at.name == "BUILD_SETTLEMENT":
                lines.append(f"  {color_str}: BUILD_SETTLEMENT at {self._format_node(val, node_coords)}")

            elif at.name == "BUILD_CITY":
                lines.append(f"  {color_str}: BUILD_CITY at {self._format_node(val, node_coords)}")

            elif at.name == "BUILD_ROAD":
                lines.append(f"  {color_str}: BUILD_ROAD at {self._format_edge(val, node_coords)}")

            elif at.name == "BUY_DEVELOPMENT_CARD":
                lines.append(f"  {color_str}: BUY_DEVELOPMENT_CARD")

            elif at.name == "PLAY_KNIGHT_CARD":
                lines.append(f"  {color_str}: PLAY_KNIGHT_CARD")

            elif at.name == "PLAY_YEAR_OF_PLENTY":
                if isinstance(val, tuple) and len(val) == 2:
                    r1 = val[0].name if hasattr(val[0], 'name') else str(val[0])
                    r2 = val[1].name if hasattr(val[1], 'name') else str(val[1])
                    lines.append(f"  {color_str}: PLAY_YEAR_OF_PLENTY {r1}, {r2}")
                else:
                    lines.append(f"  {color_str}: PLAY_YEAR_OF_PLENTY {val}")

            elif at.name == "PLAY_MONOPOLY":
                r = val.name if hasattr(val, 'name') else str(val)
                lines.append(f"  {color_str}: PLAY_MONOPOLY {r}")

            elif at.name == "PLAY_ROAD_BUILDING":
                lines.append(f"  {color_str}: PLAY_ROAD_BUILDING")

            elif at.name == "MARITIME_TRADE":
                # value is 5-resource tuple, last resource is what's received
                if isinstance(val, tuple) and len(val) >= 5:
                    offered_parts = []
                    for i in range(len(val) - 1):
                        if val[i] is not None:
                            r = val[i].name if hasattr(val[i], 'name') else str(val[i])
                            offered_parts.append(r)
                    received = val[-1]
                    r_name = received.name if hasattr(received, 'name') else str(received)
                    count = len(offered_parts)
                    # All offered resources are the same type in maritime trade
                    if offered_parts:
                        lines.append(f"  {color_str}: MARITIME_TRADE {count} {offered_parts[0]} for 1 {r_name}")
                    else:
                        lines.append(f"  {color_str}: MARITIME_TRADE {val}")
                else:
                    lines.append(f"  {color_str}: MARITIME_TRADE {val}")

            elif at.name == "OFFER_TRADE":
                if isinstance(val, tuple) and len(val) >= 10:
                    offered = self._format_resource_tuple(val[:5])
                    wanted = self._format_resource_tuple(val[5:10])
                    lines.append(f"  {color_str}: OFFER_TRADE offering {offered} for {wanted}")
                else:
                    lines.append(f"  {color_str}: OFFER_TRADE {val}")

            elif at.name == "ACCEPT_TRADE":
                lines.append(f"  {color_str}: ACCEPT_TRADE")

            elif at.name == "REJECT_TRADE":
                lines.append(f"  {color_str}: REJECT_TRADE")

            elif at.name == "COUNTER_OFFER":
                if isinstance(val, tuple) and len(val) >= 10:
                    offered = self._format_resource_tuple(val[:5])
                    wanted = self._format_resource_tuple(val[5:10])
                    lines.append(f"  {color_str}: COUNTER_OFFER offering {offered} for {wanted}")
                else:
                    lines.append(f"  {color_str}: COUNTER_OFFER {val}")

            elif at.name == "JOIN_COUNTER_OFFER":
                target = val.name if hasattr(val, 'name') else str(val)
                lines.append(f"  {color_str}: JOIN_COUNTER_OFFER with {target}")

            elif at.name == "CONFIRM_TRADE":
                partner = val.name if hasattr(val, 'name') else str(val)
                lines.append(f"  {color_str}: CONFIRM_TRADE with {partner}")

            elif at.name == "ACCEPT_COUNTER_OFFER":
                if isinstance(val, tuple):
                    target = val[0].name if hasattr(val[0], 'name') else str(val[0])
                    lines.append(f"  {color_str}: ACCEPT_COUNTER_OFFER from {target}")
                else:
                    target = val.name if hasattr(val, 'name') else str(val)
                    lines.append(f"  {color_str}: ACCEPT_COUNTER_OFFER from {target}")

            elif at.name == "CANCEL_TRADE":
                lines.append(f"  {color_str}: CANCEL_TRADE")

            elif at.name == "MOVE_ROBBER":
                lines.append(f"  {color_str}: MOVE_ROBBER to {val}")

            elif at.name == "STEAL":
                if isinstance(val, tuple) and len(val) >= 2:
                    victim = val[0].name if hasattr(val[0], 'name') else str(val[0])
                    lines.append(f"  {color_str}: STEAL from {victim}")
                else:
                    lines.append(f"  {color_str}: STEAL {val}")

            elif at.name == "DISCARD":
                if isinstance(val, (list, tuple)):
                    lines.append(f"  {color_str}: DISCARD {len(val)} cards")
                else:
                    lines.append(f"  {color_str}: DISCARD")

            elif at.name == "END_TURN":
                lines.append(f"  {color_str}: END_TURN")

            else:
                lines.append(f"  {color_str}: {at.name} {val}")

        return "\n".join(lines)

    def _format_strategic_context(self, obs: CatanObservation) -> str:
        """Format high-level strategic context."""
        lines = []

        # Phase
        lines.append(f"Phase: {obs.current_phase}")

        if obs.current_phase == "initial_placement":
            placed = len(obs.my_settlements)
            if placed <= 0:
                lines.append("Initial placement rule: you do NOT receive starting resources for your 1st settlement.")
                lines.append("Initial placement rule: you DO receive starting resources after placing your 2nd settlement (one from each adjacent non-desert tile).")
            elif placed == 1:
                lines.append("Initial placement: you have placed 1/2 settlements. No starting resources are gained from the 1st settlement.")
                lines.append("Initial placement: you will gain starting resources after placing your 2nd settlement (one from each adjacent non-desert tile).")
            else:
                lines.append("Initial placement: you have already placed both settlements. Starting resources come only from the 2nd settlement.")

        # Score status
        lines.append(f"Your VP: {obs.my_vp}/10")

        if obs.last_dice_roll:
            lines.append(f"Last dice roll: {obs.last_dice_roll}")

        return "\n".join(lines)


def create_observation_from_state(
    game_state,
    player_color: Color,
    recent_events: Optional[List[Action]] = None,
) -> CatanObservation:
    """
    Convert Catanatron game state to structured CatanObservation.

    Args:
        game_state: Catanatron State object
        player_color: Color of the observing player
        recent_events: Actions that occurred since this player last observed

    Returns:
        CatanObservation with all relevant game data
    """
    # Get player resources
    resource_freqdeck = get_player_freqdeck(game_state, player_color)
    my_resources = {
        str(RESOURCES[i]): count
        for i, count in enumerate(resource_freqdeck)
    }

    # Get dev cards
    my_dev_cards = {
        'KNIGHT': get_dev_cards_in_hand(game_state, player_color, 'KNIGHT'),
        'VICTORY_POINT': get_dev_cards_in_hand(game_state, player_color, 'VICTORY_POINT'),
        'ROAD_BUILDING': get_dev_cards_in_hand(game_state, player_color, 'ROAD_BUILDING'),
        'MONOPOLY': get_dev_cards_in_hand(game_state, player_color, 'MONOPOLY'),
        'YEAR_OF_PLENTY': get_dev_cards_in_hand(game_state, player_color, 'YEAR_OF_PLENTY'),
    }

    # Get buildings
    my_settlements = get_player_buildings(game_state, player_color, SETTLEMENT)
    my_cities = get_player_buildings(game_state, player_color, CITY)
    my_roads = get_player_buildings(game_state, player_color, ROAD)

    # Get opponent info
    opponent_colors = [c for c in game_state.colors if c != player_color]
    opponent_settlements = {
        color: get_player_buildings(game_state, color, SETTLEMENT)
        for color in opponent_colors
    }
    opponent_cities = {
        color: get_player_buildings(game_state, color, CITY)
        for color in opponent_colors
    }
    opponent_roads = {
        color: get_player_buildings(game_state, color, ROAD)
        for color in opponent_colors
    }
    opponent_vps = {
        color: get_visible_victory_points(game_state, color)
        for color in opponent_colors
    }
    opponent_resource_counts = {
        color: sum(get_player_freqdeck(game_state, color))
        for color in opponent_colors
    }

    # Determine phase
    if game_state.is_initial_build_phase:
        phase = "initial_placement"
    elif game_state.is_discarding:
        phase = "discarding"
    elif game_state.is_moving_knight:
        phase = "moving_robber"
    else:
        phase = "main_game"

    # Get special holders
    longest_road_holder = game_state.board.road_color
    largest_army_holder = None
    for color in game_state.colors:
        key = player_key(game_state, color)
        if game_state.player_state.get(f"{key}_HAS_ARMY"):
            largest_army_holder = color
            break

    # Get trading state
    active_trades = getattr(game_state, 'active_trades', {})
    counter_offers = getattr(game_state, 'counter_offers', {})
    turn_player_color = game_state.colors[game_state.current_turn_index]
    is_my_turn = player_color == turn_player_color

    return CatanObservation(
        my_color=player_color,
        my_settlements=my_settlements,
        my_cities=my_cities,
        my_roads=my_roads,
        opponent_settlements=opponent_settlements,
        opponent_cities=opponent_cities,
        opponent_roads=opponent_roads,
        my_resources=my_resources,
        my_dev_cards=my_dev_cards,
        opponent_resource_counts=opponent_resource_counts,
        current_turn=game_state.num_turns,
        current_phase=phase,
        last_dice_roll=None,  # TODO: Track last dice roll
        robber_position=game_state.board.robber_coordinate,
        my_vp=get_visible_victory_points(game_state, player_color),
        opponent_vps=opponent_vps,
        longest_road_holder=longest_road_holder,
        largest_army_holder=largest_army_holder,
        my_longest_road_length=get_longest_road_length(game_state, player_color),
        valid_actions=game_state.playable_actions,
        board_map=game_state.board.map,
        buildings_dict=game_state.board.buildings,
        active_trades=active_trades,
        counter_offers=counter_offers,
        is_my_turn=is_my_turn,
        turn_player_color=turn_player_color,
        recent_events=recent_events if recent_events is not None else [],
    )
