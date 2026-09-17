"""
Text-based observation formatter for LLM agents.

Converts Catanatron game state into semantic text descriptions
following the FLE (Factorio Learning Environment) pattern.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from cle.game_engine.state_functions import (
    get_player_freqdeck,
    get_player_buildings,
    get_visible_victory_points,
    get_dev_cards_in_hand,
    player_key,
    get_longest_road_length,
)
from cle.game_engine.models.enums import (
    RESOURCES,
    SETTLEMENT,
    CITY,
    ROAD,
    Action,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeCandidate, TradeOffer


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
    opponent_dev_card_counts: Dict[Color, int]  # total dev cards in hand

    # Game state
    current_turn: int
    current_phase: str  # initial_placement, main_game, discarding, moving_robber
    turn_order: tuple[Color, ...]
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
    trade_window: Any
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

    def format(
        self,
        obs: CatanObservation,
        *,
        include_legal_actions: bool = True,
        include_initial_placement_order: bool = True,
        shared: bool = False,
    ) -> FormattedObservation:
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
        resources = self._format_resources(obs, shared=shared)
        opponents = self._format_opponents(obs, shared=shared)
        valid_actions = (
            self._format_valid_actions(obs) if include_legal_actions else ""
        )
        strategic_context = self._format_strategic_context(
            obs,
            include_initial_placement_order=include_initial_placement_order,
            shared=shared,
        )
        trade_context = self._format_trade_context(obs)
        events_section = self._format_events(obs)

        # Only include trade section if there's active trading
        trade_section = f"\n<trading>\n{trade_context}\n</trading>" if trade_context else ""

        # Only include events and legal actions when requested by the caller.
        events_block = f"\n<recent_events>\n{events_section}\n</recent_events>" if events_section else ""
        actions_block = (
            f"\n<valid_actions>\n{valid_actions}\n</valid_actions>"
            if include_legal_actions
            else ""
        )

        color_name = obs.my_color.value if hasattr(obs.my_color, 'value') else str(obs.my_color)
        raw_str = f"""<game_state turn="{obs.current_turn}" player="{color_name}">{events_block}
<phase_info>
{strategic_context}
</phase_info>
<board_state>
{board_state}
</board_state>
<resources>
{resources}
</resources>
<opponents>
{opponents}
</opponents>{trade_section}{actions_block}
</game_state>""".strip()

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

    def _number_to_pips(self, number: int) -> int:
        """Convert a production number to its two-dice probability dots."""
        pips_map = {
            2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
            8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
        }
        return pips_map.get(number, 0)

    def _format_resources(self, obs: CatanObservation, *, shared: bool = False) -> str:
        """Format resources with building possibilities."""
        lines = ["YOUR RESOURCES:"]

        total = sum(obs.my_resources.values())
        if total == 0:
            lines.append("  No resources")
            if not shared:
                return "\n".join(lines)

        # List all resources (always show all 5 so the model knows what's at 0)
        for resource in RESOURCES:
            resource_name = resource.name if hasattr(resource, 'name') else str(resource)
            count = obs.my_resources.get(resource_name, 0)
            lines.append(f"  {resource_name}: {count}")

        lines.append(f"  Total: {total} cards")

        if shared:
            # The bank is always a counterparty; models that never see a rate
            # sit on 8-card hands and discard instead of trading 4:1.
            owned_ports = set()
            my_nodes = set(obs.my_settlements) | set(obs.my_cities)
            for port_resource, node_ids in obs.board_map.port_nodes.items():
                if my_nodes.intersection(node_ids):
                    owned_ports.add(
                        port_resource.name if hasattr(port_resource, "name") else port_resource
                    )
            base_rate = 3 if None in owned_ports else 4
            rates = []
            for resource in RESOURCES:
                name = resource.name if hasattr(resource, "name") else str(resource)
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

        # Add what you can afford
        affordable = self._get_affordable_buildings(obs.my_resources)
        if affordable:
            lines.append(f"  Can afford: {', '.join(affordable)}")

        # Dev cards
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
                        playable = any(a.color == obs.my_color and a.action_type == action_type for a in obs.valid_actions)
                        status = " (passive VP; not played)" if card_name == "VICTORY_POINT" else (
                            " (playable now)" if playable else " (not playable now)"
                        )
                    lines.append(f"    {card_name}: {count}{status}")

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

    def _format_opponents(self, obs: CatanObservation, *, shared: bool = False) -> str:
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
                    lines.append("    Placements:")
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
        """Format the complete legal menu without changing engine ordering."""
        lines = [
            "EXACT LEGAL ACTION MENU "
            "(all entries are legal and affordable; choose its zero-based index):"
        ]
        if not obs.valid_actions:
            lines.append("  No actions available (waiting for turn)")
            return "\n".join(lines)

        lines.extend(
            f"  {index}. {self._format_single_action(action, obs)}"
            for index, action in enumerate(obs.valid_actions)
        )
        return "\n".join(lines)

    def _describe_node(self, node_id: int, obs: CatanObservation) -> str:
        """Describe a node with both dice numbers and production pips."""
        if node_id not in obs.board_map.adjacent_tiles:
            return f"node {node_id}"

        tiles = obs.board_map.adjacent_tiles[node_id]
        tile_descs = []
        total_pips = 0
        for tile in tiles:
            if tile.resource is not None and tile.number is not None:
                pips = self._number_to_pips(tile.number)
                total_pips += pips
                tile_descs.append(f"{tile.resource} dice={tile.number}({pips}pip)")

        if not tile_descs:
            return f"node {node_id}"

        # Check for port
        port_str = ""
        for resource, nodes in obs.board_map.port_nodes.items():
            if node_id in nodes:
                port_str = " (3:1 port)" if resource is None else f" ({resource} 2:1 port)"
                break

        return f"{'/'.join(tile_descs)} [{total_pips}pips]{port_str}"

    def _format_single_action(
        self,
        action: Action,
        obs: CatanObservation,
        *,
        discard_count: int | None = None,
    ) -> str:
        """Format a single action in Catan lingo."""
        at = action.action_type

        if at == ActionType.END_TURN:
            return "End turn"

        if at == ActionType.ROLL:
            return "Roll dice"

        if at == ActionType.BUILD_SETTLEMENT:
            desc = self._describe_node(action.value, obs)
            return f"Build settlement at {desc}"

        if at == ActionType.BUILD_CITY:
            desc = self._describe_node(action.value, obs)
            return f"Upgrade to city at {desc}"

        if at == ActionType.BUILD_ROAD:
            if isinstance(action.value, tuple) and len(action.value) == 2:
                n1_desc = self._describe_node(action.value[0], obs)
                n2_desc = self._describe_node(action.value[1], obs)
                return f"Build road between {n1_desc} and {n2_desc}"

        if at == ActionType.BUY_DEVELOPMENT_CARD:
            return "Buy development card"

        if at == ActionType.MARITIME_TRADE:
            val = action.value
            if isinstance(val, tuple) and len(val) >= 2:
                offered = [r for r in val[:-1] if r is not None]
                received = val[-1]
                if offered:
                    give_name = offered[0].name if hasattr(offered[0], 'name') else str(offered[0])
                    get_name = received.name if hasattr(received, 'name') else str(received)
                    return f"Trade {len(offered)} {give_name} for 1 {get_name}"

        if at == ActionType.OFFER_TRADE:
            if isinstance(action.value, TradeOffer):
                return (
                    f"Offer: give {self._format_resource_tuple_with_any(action.value.give, action.value.give_any)}, "
                    f"receive {self._format_resource_tuple_with_any(action.value.receive, action.value.receive_any)}"
                )
            return str(action.value)

        if at in {ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE}:
            offer = (
                obs.trade_window.offers.get(action.value)
                if obs.trade_window is not None
                else None
            )
            verb = "Signal willingness for" if at == ActionType.ACCEPT_TRADE else "Decline"
            if offer is None:
                return f"{verb} offer {action.value}"
            give = self._format_resource_tuple_with_any(
                offer.receive,
                offer.receive_any,
            )
            receive = self._format_resource_tuple_with_any(
                offer.give,
                offer.give_any,
            )
            offered_by = self._color_name(offer.offered_by)
            return (
                f"{verb} {offer.id} from {offered_by}: "
                f"give {give}, receive {receive}"
            )

        if at == ActionType.COUNTER_OFFER:
            if isinstance(action.value, TradeOffer):
                return (
                    f"Counter {action.value.parent_offer_id}: give "
                    f"{self._format_resource_tuple_with_any(action.value.give, action.value.give_any)}, "
                    f"receive {self._format_resource_tuple_with_any(action.value.receive, action.value.receive_any)}"
                )
            return str(action.value)

        if at == ActionType.CONFIRM_TRADE and isinstance(
            action.value,
            TradeCandidate,
        ):
            candidate = action.value
            offer = (
                obs.trade_window.offers.get(candidate.offer_id)
                if obs.trade_window is not None
                else None
            )
            partner = self._color_name(candidate.counterparty)
            if offer is None:
                return f"Confirm {candidate.offer_id} with {partner}"
            if offer.offered_by == candidate.turn_player:
                give, give_any = offer.give, offer.give_any
                receive, receive_any = offer.receive, offer.receive_any
            else:
                give, give_any = offer.receive, offer.receive_any
                receive, receive_any = offer.give, offer.give_any
            return (
                f"Confirm {candidate.offer_id} with {partner}: give "
                f"{self._format_resource_tuple_with_any(give, give_any)}, "
                f"receive "
                f"{self._format_resource_tuple_with_any(receive, receive_any)}"
            )

        if at == ActionType.CANCEL_TRADE:
            return f"Withdraw offer {action.value}"

        if at == ActionType.PLAY_KNIGHT_CARD:
            return "Play knight card"

        if at == ActionType.PLAY_YEAR_OF_PLENTY:
            if isinstance(action.value, tuple) and len(action.value) in {1, 2}:
                resources = " and ".join(
                    resource.name if hasattr(resource, "name") else str(resource)
                    for resource in action.value
                )
                return f"Year of Plenty: take {resources}"
            return "Play Year of Plenty"

        if at == ActionType.PLAY_MONOPOLY:
            r = action.value.name if hasattr(action.value, 'name') else str(action.value)
            return f"Monopoly on {r}"

        if at == ActionType.PLAY_ROAD_BUILDING:
            return "Play road building card"

        if at == ActionType.MOVE_ROBBER:
            return f"Move robber to {action.value}"

        if at == ActionType.STEAL:
            if action.value is not None:
                color_name = action.value.name if hasattr(action.value, 'name') else str(action.value)
                return f"Steal from {color_name}"
            return "Steal (no targets)"

        if at == ActionType.DISCARD:
            if discard_count is not None:
                return (
                    f"Discard exactly {discard_count} resource cards; "
                    "specify named resource counts in <discard>."
                )
            return "Discard resources"

        # Default
        type_name = at.name if hasattr(at, 'name') else str(at)
        return f"{type_name}: {action.value}"

    def _format_trade_context(self, obs: CatanObservation) -> str:
        """Format the bounded offer board."""
        window = getattr(obs, "trade_window", None)
        if window is None or not window.active_offers:
            return ""
        lines = [
            f"TRADING WINDOW {window.id} (round {window.round}):",
            f"  Remaining root slots: {window.remaining_root_slots}",
            f"  Remaining counter slots: {window.remaining_counter_slots}",
        ]
        if any(offer.parent_offer_id for offer in window.active_offers):
            lines.append(
                "  Counteroffers cannot be accepted; the turn player executes one with confirm_trade."
            )
        for offer in window.active_offers:
            give = self._format_resource_tuple_with_any(
                offer.give,
                offer.give_any,
            )
            receive = self._format_resource_tuple_with_any(
                offer.receive,
                offer.receive_any,
            )
            parent = (
                f" counter to {offer.parent_offer_id}"
                if offer.parent_offer_id
                else ""
            )
            line = (
                f"  {offer.id}: {self._color_name(offer.offered_by)} "
                f"gives {give} for {receive}{parent}"
            )
            if offer.willing_by:
                line += " [willing: " + ", ".join(
                    self._color_name(color)
                    for color in offer.willing_by
                ) + "]"
            if offer.declined_by:
                line += " [declined: " + ", ".join(
                    self._color_name(color)
                    for color in offer.declined_by
                ) + "]"
            lines.append(line)
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

            elif at.name in {"OFFER_TRADE", "COUNTER_OFFER"}:
                if isinstance(val, TradeOffer):
                    give = self._format_resource_tuple_with_any(
                        val.give,
                        val.give_any,
                    )
                    receive = self._format_resource_tuple_with_any(
                        val.receive,
                        val.receive_any,
                    )
                    parent = (
                        f" countering {val.parent_offer_id}"
                        if val.parent_offer_id
                        else ""
                    )
                    lines.append(
                        f"  {color_str}: {at.name}{parent}, "
                        f"giving {give} for {receive}"
                    )
                else:
                    lines.append(f"  {color_str}: {at.name} {val}")

            elif at.name == "ACCEPT_TRADE":
                lines.append(f"  {color_str}: WILLING_TO_TRADE on offer {val}")

            elif at.name == "REJECT_TRADE":
                lines.append(f"  {color_str}: DECLINED offer {val}")

            elif at.name == "CONFIRM_TRADE":
                if isinstance(val, TradeCandidate):
                    partner = self._color_name(val.counterparty)
                    lines.append(
                        f"  {color_str}: CONFIRM_TRADE offer "
                        f"{val.offer_id} with {partner}"
                    )
                else:
                    partner = val.name if hasattr(val, 'name') else str(val)
                    lines.append(f"  {color_str}: CONFIRM_TRADE with {partner}")

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

    def _format_strategic_context(
        self,
        obs: CatanObservation,
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
                    raise ValueError(
                        "Initial-placement observation has an invalid turn order"
                    )
                second_round = tuple(reversed(first_round))
                first_names = " -> ".join(
                    self._color_name(color) for color in first_round
                )
                second_names = " -> ".join(
                    self._color_name(color) for color in second_round
                )
                player_count = len(first_round)
                lines.extend(
                    (
                        "Initial placement order (each settlement is immediately "
                        "followed by that player's road):",
                        f"  Round 1 (first settlement + road): {first_names}",
                        f"  Round 2 (second settlement + road): {second_names}",
                        "  Your positions: "
                        f"round 1 = {first_round.index(obs.my_color) + 1}/{player_count}; "
                        f"round 2 = {second_round.index(obs.my_color) + 1}/{player_count}.",
                    )
                )

        actual_vp = getattr(obs, "my_actual_vp", None)
        if shared and actual_vp is not None:
            lines.append(f"Your actual VP: {actual_vp}/10 (public: {obs.my_vp})")
        else:
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
    opponent_dev_card_counts = {
        color: get_dev_cards_in_hand(game_state, color)
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
        opponent_dev_card_counts=opponent_dev_card_counts,
        current_turn=game_state.num_turns,
        current_phase=phase,
        turn_order=tuple(game_state.colors),
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
        trade_window=getattr(game_state, "trade_window", None),
        is_my_turn=is_my_turn,
        turn_player_color=turn_player_color,
        recent_events=recent_events if recent_events is not None else [],
    )
