"""
Colonist.io Replay Decoder

Decodes replay JSON files and reconstructs game state step by step.
Extracts valid moves at each game state for training data generation.
"""

import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import IntEnum


class TileType(IntEnum):
    DESERT = 0
    WHEAT = 1
    BRICK = 2
    SHEEP = 3
    WOOD = 4
    ORE = 5


class ResourceType(IntEnum):
    WHEAT = 1
    BRICK = 2
    SHEEP = 3
    WOOD = 4
    ORE = 5
    ANY = 9  # Used in trade offers for 3:1 ports


class PortType(IntEnum):
    GENERIC = 1  # 3:1
    WHEAT = 2
    BRICK = 3
    ORE = 4
    SHEEP = 5
    WOOD = 6


class BuildingType(IntEnum):
    NONE = 0
    SETTLEMENT = 1
    CITY = 2


class PieceEnum(IntEnum):
    ROAD = 0
    SHIP = 1
    SETTLEMENT = 2
    CITY = 3
    WALL = 4
    ROBBER = 5


class ActionState(IntEnum):
    """Game action states - what the current player can/must do"""
    MAIN_TURN = 0  # Can roll, trade, build, end turn
    SETUP_PLACE_SETTLEMENT = 1  # Must place settlement (setup phase)
    SETUP_PLACE_ROAD = 3  # Must place road (setup phase)
    MUST_ROLL_DICE = 24  # Must roll dice
    MUST_MOVE_ROBBER = 27  # Must move robber (after rolling 7)
    ROBBER_STEAL = 28  # Must steal from player
    ROAD_BUILDING_1 = 30  # Playing road building dev card
    ROAD_BUILDING_2 = 31  # Second road from road building


class DevCardType(IntEnum):
    KNIGHT = 11
    YEAR_OF_PLENTY = 13
    ROAD_BUILDING = 14
    MONOPOLY = 15
    VICTORY_POINT = 16  # Guess


class GameLogType(IntEnum):
    TURN_START = 1
    BUILD_PIECE = 4
    BUY_PIECE = 5
    DICE_ROLL = 10
    ROBBER_MOVED = 11
    CARDS_RECEIVED = 14
    CARDS_DISCARDED = 15
    ROBBER_STEAL = 16
    DEV_CARD_PLAYED = 20
    YEAR_OF_PLENTY = 21
    GAME_END = 44
    GAME_WINNER = 45
    RESOURCE_DISTRIBUTION = 47
    ROBBER_ON_TILE = 49
    DISCARD_ON_7 = 55
    ACHIEVEMENT = 66
    MONOPOLY = 86
    EMBARGO = 113
    REMOVE_EMBARGO = 114
    PLAYER_TRADE = 115
    BANK_TRADE = 116
    COUNTER_OFFER = 117
    TRADE_OFFER = 118


@dataclass
class HexTile:
    index: int
    x: int
    y: int
    tile_type: TileType
    dice_number: int

    def produces_resource(self) -> Optional[ResourceType]:
        """Return resource type this tile produces, or None for desert"""
        if self.tile_type == TileType.DESERT:
            return None
        # tile_type 1-5 maps to resource 1-5
        return ResourceType(self.tile_type)


@dataclass
class Port:
    edge_id: int
    x: int
    y: int
    z: int
    port_type: PortType

    def trade_ratio(self) -> int:
        return 3 if self.port_type == PortType.GENERIC else 2


@dataclass
class Corner:
    index: int
    x: int
    y: int
    z: int
    owner: Optional[int] = None
    building_type: BuildingType = BuildingType.NONE


@dataclass
class Edge:
    index: int
    x: int
    y: int
    z: int
    owner: Optional[int] = None
    road_type: int = 0  # 1 = road


@dataclass
class PlayerState:
    color: int
    is_connected: bool = True
    resources: List[int] = field(default_factory=list)  # List of resource type ints
    dev_cards: List[int] = field(default_factory=list)
    dev_cards_used: List[int] = field(default_factory=list)
    victory_points: Dict[str, int] = field(default_factory=dict)
    bank_trade_ratios: Dict[int, int] = field(default_factory=dict)  # resource -> ratio
    settlements_remaining: int = 5
    cities_remaining: int = 4
    roads_remaining: int = 15

    def resource_count(self, resource: ResourceType) -> int:
        return self.resources.count(resource)

    def total_resources(self) -> int:
        return len(self.resources)


@dataclass
class GameState:
    # Board layout (static)
    tiles: Dict[int, HexTile] = field(default_factory=dict)
    ports: Dict[int, Port] = field(default_factory=dict)
    corners: Dict[int, Corner] = field(default_factory=dict)
    edges: Dict[int, Edge] = field(default_factory=dict)

    # Dynamic state
    players: Dict[int, PlayerState] = field(default_factory=dict)
    current_player: int = 0
    action_state: ActionState = ActionState.SETUP_PLACE_SETTLEMENT
    completed_turns: int = 0
    robber_tile: int = 0
    dice_value: Tuple[int, int] = (0, 0)

    # Bank state
    bank_resources: Dict[int, int] = field(default_factory=dict)
    bank_dev_cards: List[int] = field(default_factory=list)

    # Longest road / largest army
    longest_road_player: Optional[int] = None
    longest_road_length: int = 0
    largest_army_player: Optional[int] = None
    largest_army_size: int = 0


class ReplayDecoder:
    def __init__(self, replay_path: str):
        with open(replay_path) as f:
            self.raw_data = json.load(f)

        self.data = self.raw_data['data']
        self.event_history = self.data['eventHistory']
        self.events = self.event_history['events']
        self.initial_state = self.event_history.get('initialState', {})
        self.play_order = self.data.get('playOrder', [])

        self.state = GameState()
        self._initialize_state()

    def _initialize_state(self):
        """Initialize game state from initialState"""
        init = self.initial_state

        # Initialize board
        map_state = init.get('mapState', {})

        # Tiles
        for tid, tdata in map_state.get('tileHexStates', {}).items():
            self.state.tiles[int(tid)] = HexTile(
                index=int(tid),
                x=tdata['x'],
                y=tdata['y'],
                tile_type=TileType(tdata['type']),
                dice_number=tdata['diceNumber']
            )

        # Ports
        for eid, pdata in map_state.get('portEdgeStates', {}).items():
            self.state.ports[int(eid)] = Port(
                edge_id=int(eid),
                x=pdata['x'],
                y=pdata['y'],
                z=pdata['z'],
                port_type=PortType(pdata['type'])
            )

        # Corners
        for cid, cdata in map_state.get('tileCornerStates', {}).items():
            self.state.corners[int(cid)] = Corner(
                index=int(cid),
                x=cdata['x'],
                y=cdata['y'],
                z=cdata['z']
            )

        # Edges
        for eid, edata in map_state.get('tileEdgeStates', {}).items():
            self.state.edges[int(eid)] = Edge(
                index=int(eid),
                x=edata['x'],
                y=edata['y'],
                z=edata['z']
            )

        # Players
        for pid, pdata in init.get('playerStates', {}).items():
            color = pdata['color']
            self.state.players[color] = PlayerState(
                color=color,
                is_connected=pdata.get('isConnected', True),
                resources=list(pdata.get('resourceCards', {}).get('cards', [])),
                bank_trade_ratios={int(k): v for k, v in pdata.get('bankTradeRatiosState', {}).items()}
            )

        # Current state
        curr = init.get('currentState', {})
        self.state.current_player = curr.get('currentTurnPlayerColor', self.play_order[0] if self.play_order else 0)
        self.state.action_state = ActionState(curr.get('actionState', 1))
        self.state.completed_turns = curr.get('completedTurns', 0)

        # Robber
        robber = init.get('mechanicRobberState', {})
        self.state.robber_tile = robber.get('locationTileIndex', 0)

        # Bank
        bank = init.get('bankState', {})
        for rid, count in bank.get('resourceCards', {}).items():
            self.state.bank_resources[int(rid)] = count

        # Dev cards bank
        dev_bank = init.get('mechanicDevelopmentCardsState', {}).get('bankDevelopmentCards', {})
        self.state.bank_dev_cards = dev_bank.get('cards', [])

    def apply_event(self, event: Dict) -> Dict[str, Any]:
        """Apply a single event to the game state, return action info"""
        state_change = event.get('stateChange', {})
        action_info = {
            'delta_s': event.get('input', {}).get('deltaS', 0),
            'action_type': None,
            'player': None,
            'details': {}
        }

        # Apply map state changes (buildings, roads)
        if 'mapState' in state_change:
            ms = state_change['mapState']

            # Corner states (settlements, cities)
            for cid, cdata in ms.get('tileCornerStates', {}).items():
                corner = self.state.corners[int(cid)]
                if 'owner' in cdata:
                    corner.owner = cdata['owner']
                    action_info['player'] = cdata['owner']
                if 'buildingType' in cdata:
                    old_type = corner.building_type
                    corner.building_type = BuildingType(cdata['buildingType'])
                    if old_type == BuildingType.NONE:
                        action_info['action_type'] = 'build_settlement'
                    elif old_type == BuildingType.SETTLEMENT:
                        action_info['action_type'] = 'build_city'
                    action_info['details']['corner'] = int(cid)

            # Edge states (roads)
            for eid, edata in ms.get('tileEdgeStates', {}).items():
                edge = self.state.edges[int(eid)]
                if 'owner' in edata:
                    edge.owner = edata['owner']
                    action_info['player'] = edata['owner']
                if 'type' in edata:
                    edge.road_type = edata['type']
                    action_info['action_type'] = 'build_road'
                    action_info['details']['edge'] = int(eid)

        # Apply current state changes
        if 'currentState' in state_change:
            cs = state_change['currentState']
            if 'currentTurnPlayerColor' in cs:
                self.state.current_player = cs['currentTurnPlayerColor']
            if 'actionState' in cs:
                self.state.action_state = ActionState(cs['actionState'])
            if 'completedTurns' in cs:
                self.state.completed_turns = cs['completedTurns']

        # Apply dice state changes
        if 'diceState' in state_change:
            ds = state_change['diceState']
            if 'dice1' in ds and 'dice2' in ds:
                self.state.dice_value = (ds['dice1'], ds['dice2'])
                action_info['action_type'] = 'roll_dice'
                action_info['details']['dice'] = self.state.dice_value

        # Apply player state changes
        if 'playerStates' in state_change:
            for pid, pdata in state_change['playerStates'].items():
                color = int(pid)
                if color not in self.state.players:
                    continue
                player = self.state.players[color]

                if 'resourceCards' in pdata:
                    player.resources = list(pdata['resourceCards'].get('cards', []))

                if 'victoryPointsState' in pdata:
                    for vptype, vpcount in pdata['victoryPointsState'].items():
                        player.victory_points[vptype] = vpcount

                if 'bankTradeRatiosState' in pdata:
                    for rid, ratio in pdata['bankTradeRatiosState'].items():
                        player.bank_trade_ratios[int(rid)] = ratio

        # Apply robber state changes
        if 'mechanicRobberState' in state_change:
            rs = state_change['mechanicRobberState']
            if 'locationTileIndex' in rs:
                self.state.robber_tile = rs['locationTileIndex']
                action_info['action_type'] = 'move_robber'
                action_info['details']['tile'] = self.state.robber_tile

        return action_info

    def get_valid_moves(self) -> List[Dict]:
        """Get all valid moves for current game state"""
        moves = []
        player = self.state.players.get(self.state.current_player)
        if not player:
            return moves

        action_state = self.state.action_state

        # Setup phase - place settlement
        if action_state == ActionState.SETUP_PLACE_SETTLEMENT:
            for cid, corner in self.state.corners.items():
                if self._can_place_settlement(cid, is_setup=True):
                    moves.append({'type': 'build_settlement', 'corner': cid})

        # Setup phase - place road
        elif action_state == ActionState.SETUP_PLACE_ROAD:
            for eid, edge in self.state.edges.items():
                if self._can_place_road(eid):
                    moves.append({'type': 'build_road', 'edge': eid})

        # Must roll dice
        elif action_state == ActionState.MUST_ROLL_DICE:
            moves.append({'type': 'roll_dice'})
            # Can also play knight before rolling
            if DevCardType.KNIGHT in player.dev_cards:
                moves.append({'type': 'play_knight'})

        # Must move robber
        elif action_state == ActionState.MUST_MOVE_ROBBER:
            for tid, tile in self.state.tiles.items():
                if tid != self.state.robber_tile:
                    moves.append({'type': 'move_robber', 'tile': tid})

        # Main turn
        elif action_state == ActionState.MAIN_TURN:
            moves.append({'type': 'end_turn'})

            # Build settlement
            if self._can_afford_settlement(player):
                for cid, corner in self.state.corners.items():
                    if self._can_place_settlement(cid):
                        moves.append({'type': 'build_settlement', 'corner': cid})

            # Build city
            if self._can_afford_city(player):
                for cid, corner in self.state.corners.items():
                    if corner.owner == player.color and corner.building_type == BuildingType.SETTLEMENT:
                        moves.append({'type': 'build_city', 'corner': cid})

            # Build road
            if self._can_afford_road(player):
                for eid, edge in self.state.edges.items():
                    if self._can_place_road(eid):
                        moves.append({'type': 'build_road', 'edge': eid})

            # Buy dev card
            if self._can_afford_dev_card(player) and len(self.state.bank_dev_cards) > 0:
                moves.append({'type': 'buy_dev_card'})

            # Play dev cards
            for card in set(player.dev_cards):
                if card not in player.dev_cards_used:
                    moves.append({'type': 'play_dev_card', 'card': card})

            # Bank trades (simplified - just check if can trade 4:1 or better)
            for res in ResourceType:
                if res == ResourceType.ANY:
                    continue
                ratio = player.bank_trade_ratios.get(res, 4)
                if player.resource_count(res) >= ratio:
                    for target in ResourceType:
                        if target != res and target != ResourceType.ANY:
                            if self.state.bank_resources.get(target, 0) > 0:
                                moves.append({'type': 'bank_trade', 'give': res, 'get': target, 'ratio': ratio})

        return moves

    def _can_place_settlement(self, corner_id: int, is_setup: bool = False) -> bool:
        """Check if settlement can be placed at corner"""
        corner = self.state.corners.get(corner_id)
        if not corner or corner.building_type != BuildingType.NONE:
            return False

        # Check distance rule - no adjacent settlements
        for other_cid, other in self.state.corners.items():
            if other_cid != corner_id and other.building_type != BuildingType.NONE:
                if self._corners_adjacent(corner_id, other_cid):
                    return False

        # During normal play, must connect to own road
        if not is_setup:
            has_connected_road = False
            for eid, edge in self.state.edges.items():
                if edge.owner == self.state.current_player:
                    if self._edge_connects_corner(eid, corner_id):
                        has_connected_road = True
                        break
            if not has_connected_road:
                return False

        return True

    def _can_place_road(self, edge_id: int) -> bool:
        """Check if road can be placed at edge"""
        edge = self.state.edges.get(edge_id)
        if not edge or edge.road_type != 0:
            return False

        # Must connect to own settlement/city or road
        player_color = self.state.current_player
        has_connection = False

        # Check corners
        for cid, corner in self.state.corners.items():
            if corner.owner == player_color:
                if self._edge_connects_corner(edge_id, cid):
                    has_connection = True
                    break

        # Check other roads
        if not has_connection:
            for eid, other_edge in self.state.edges.items():
                if other_edge.owner == player_color and eid != edge_id:
                    if self._edges_connected(edge_id, eid):
                        has_connection = True
                        break

        return has_connection

    def _corners_adjacent(self, c1: int, c2: int) -> bool:
        """Check if two corners are adjacent (share an edge)"""
        corner1 = self.state.corners[c1]
        corner2 = self.state.corners[c2]
        cx, cy, cz = corner1.x, corner1.y, corner1.z
        ocx, ocy, ocz = corner2.x, corner2.y, corner2.z

        if cz == 0:
            # z=0 corner is adjacent to specific z=1 corners
            return (ocz == 1 and ocx == cx and ocy == cy) or \
                   (ocz == 1 and ocx == cx + 1 and ocy == cy - 1) or \
                   (ocz == 1 and ocx == cx - 1 and ocy == cy)
        else:  # cz == 1
            # z=1 corner is adjacent to specific z=0 corners
            return (ocz == 0 and ocx == cx and ocy == cy) or \
                   (ocz == 0 and ocx == cx - 1 and ocy == cy + 1) or \
                   (ocz == 0 and ocx == cx + 1 and ocy == cy)

    def _edge_connects_corner(self, edge_id: int, corner_id: int) -> bool:
        """Check if edge connects to corner using hex grid geometry"""
        edge = self.state.edges[edge_id]
        corner = self.state.corners[corner_id]
        ex, ey, ez = edge.x, edge.y, edge.z
        cx, cy, cz = corner.x, corner.y, corner.z

        if cz == 0:
            # z=0 corner (upper corner) is touched by:
            # - ez=0 edge at same position
            # - ez=1 edge at (cx+1, cy-1)
            # - ez=2 edge at same position OR at (cx+1, cy-1)
            return (ez == 0 and ex == cx and ey == cy) or \
                   (ez == 1 and ex == cx + 1 and ey == cy - 1) or \
                   (ez == 2 and ((ex == cx and ey == cy) or (ex == cx + 1 and ey == cy - 1)))
        else:  # cz == 1
            # z=1 corner (lower corner) is touched by:
            # - ez=0 edge at (cx, cy+1)
            # - ez=1 edge at same position
            # - ez=2 edge at same position
            return (ez == 0 and ex == cx and ey == cy + 1) or \
                   (ez == 1 and ex == cx and ey == cy) or \
                   (ez == 2 and ex == cx and ey == cy)

    def _edges_connected(self, e1: int, e2: int) -> bool:
        """Check if two edges share a corner"""
        # Two edges are connected if they both touch the same corner
        for cid in self.state.corners:
            if self._edge_connects_corner(e1, cid) and self._edge_connects_corner(e2, cid):
                return True
        return False

    def _can_afford_settlement(self, player: PlayerState) -> bool:
        return (player.resource_count(ResourceType.WHEAT) >= 1 and
                player.resource_count(ResourceType.BRICK) >= 1 and
                player.resource_count(ResourceType.SHEEP) >= 1 and
                player.resource_count(ResourceType.WOOD) >= 1)

    def _can_afford_city(self, player: PlayerState) -> bool:
        return (player.resource_count(ResourceType.WHEAT) >= 2 and
                player.resource_count(ResourceType.ORE) >= 3)

    def _can_afford_road(self, player: PlayerState) -> bool:
        return (player.resource_count(ResourceType.BRICK) >= 1 and
                player.resource_count(ResourceType.WOOD) >= 1)

    def _can_afford_dev_card(self, player: PlayerState) -> bool:
        return (player.resource_count(ResourceType.WHEAT) >= 1 and
                player.resource_count(ResourceType.SHEEP) >= 1 and
                player.resource_count(ResourceType.ORE) >= 1)

    def replay_with_states(self) -> List[Dict]:
        """Replay the game and yield state + valid moves at each step"""
        results = []

        # Initial state
        results.append({
            'event_idx': -1,
            'state': self._snapshot_state(),
            'valid_moves': self.get_valid_moves(),
            'action_taken': None
        })

        # Apply each event
        for i, event in enumerate(self.events):
            action_info = self.apply_event(event)

            # Skip chat messages and other non-game actions
            if action_info['action_type'] is not None:
                results.append({
                    'event_idx': i,
                    'state': self._snapshot_state(),
                    'valid_moves': self.get_valid_moves(),
                    'action_taken': action_info
                })

        return results

    def _get_corner_tiles(self, corner_id: int) -> List[int]:
        """Get tile indices that a corner touches (for resource production)"""
        corner = self.state.corners[corner_id]
        cx, cy, cz = corner.x, corner.y, corner.z
        touching_tiles = []

        for tid, tile in self.state.tiles.items():
            tx, ty = tile.x, tile.y
            # A corner touches a tile based on hex grid geometry
            if cz == 0:  # Upper corner type
                # Touches tiles at: (cx, cy), (cx-1, cy), (cx, cy-1)
                if (tx == cx and ty == cy) or \
                   (tx == cx - 1 and ty == cy) or \
                   (tx == cx and ty == cy - 1):
                    touching_tiles.append(tid)
            else:  # cz == 1, Lower corner type
                # Touches tiles at: (cx, cy), (cx+1, cy), (cx, cy+1)
                if (tx == cx and ty == cy) or \
                   (tx == cx + 1 and ty == cy) or \
                   (tx == cx and ty == cy + 1):
                    touching_tiles.append(tid)

        return touching_tiles

    def _snapshot_state(self) -> Dict:
        """Create a serializable snapshot of current state"""
        return {
            'current_player': self.state.current_player,
            'action_state': self.state.action_state.name,
            'completed_turns': self.state.completed_turns,
            'dice': self.state.dice_value,
            'robber_tile': self.state.robber_tile,
            'players': {
                color: {
                    'resources': list(p.resources),
                    'victory_points': dict(p.victory_points),
                    'bank_trade_ratios': dict(p.bank_trade_ratios)
                }
                for color, p in self.state.players.items()
            },
            'buildings': {
                cid: {'owner': c.owner, 'type': c.building_type.name}
                for cid, c in self.state.corners.items()
                if c.owner is not None
            },
            'roads': {
                eid: {'owner': e.owner}
                for eid, e in self.state.edges.items()
                if e.owner is not None
            }
        }

    def get_board_state(self) -> Dict:
        """Get the static board layout for replay inspection."""
        return {
            'tiles': {
                tid: {
                    'x': t.x,
                    'y': t.y,
                    'type': t.tile_type.name,
                    'dice_number': t.dice_number
                }
                for tid, t in self.state.tiles.items()
            },
            'ports': {
                pid: {
                    'x': p.x,
                    'y': p.y,
                    'z': p.z,
                    'type': p.port_type.name,
                    'ratio': p.trade_ratio()
                }
                for pid, p in self.state.ports.items()
            },
            'corner_tiles': {
                cid: self._get_corner_tiles(cid)
                for cid in self.state.corners
            }
        }



def main():

    if len(sys.argv) < 2:
        print("Usage: python replay_decoder.py <replay_file.json>")
        sys.exit(1)

    decoder = ReplayDecoder(sys.argv[1])

    print("=== GAME INFO ===")
    print(f"Play order: {decoder.play_order}")
    print(f"Total events: {len(decoder.events)}")
    print()

    print("=== BOARD LAYOUT ===")
    for tid, tile in sorted(decoder.state.tiles.items()):
        print(f"Tile {tid}: {tile.tile_type.name} ({tile.dice_number}) at ({tile.x}, {tile.y})")

    print()
    print("=== REPLAYING GAME ===")

    states = decoder.replay_with_states()
    print(f"Generated {len(states)} state snapshots")

    # Show first few
    for i, s in enumerate(states[:5]):
        print(f"\n--- State {i} (event {s['event_idx']}) ---")
        print(f"Current player: {s['state']['current_player']}")
        print(f"Action state: {s['state']['action_state']}")
        print(f"Valid moves: {len(s['valid_moves'])}")
        if s['action_taken']:
            print(f"Action taken: {s['action_taken']}")


if __name__ == '__main__':
    main()
