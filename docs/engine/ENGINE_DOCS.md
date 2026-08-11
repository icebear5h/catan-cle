# Catan Engine Documentation

Complete reference for the game engine (`engine/` directory).

## Overview

Pure Python implementation of Settlers of Catan rules. No dependencies on ML/RL frameworks.

**Location:** `engine/`

**Key Files:**
- `game.py` - Main game loop and execution
- `state.py` - Game state management
- `models/` - Data structures (Player, Board, Actions, Enums)
- `json.py` - JSON serialization for web clients
- `state_functions.py` - Helper functions for querying state

---

## Quick Start

```python
from engine.game import Game
from engine.models.player import SimplePlayer, Color

# Create 4 players
players = [
    SimplePlayer(Color.RED),
    SimplePlayer(Color.BLUE),
    SimplePlayer(Color.WHITE),
    SimplePlayer(Color.ORANGE),
]

# Initialize game
game = Game(players)

# Game loop
while game.winning_color() is None:
    actions = game.state.playable_actions
    player = game.state.current_player()
    action = player.decide(game, actions)
    game.execute(action)

# Get winner
winner = game.winning_color()
print(f"Winner: {winner}")
```

---

## Core Classes

### Game (`engine/game.py`)

Main game controller.

```python
class Game:
    def __init__(self, players: List[Player], catan_map=None, seed=None)

    # Execute an action. force=True bypasses playable-action validation only
    # when all randomness-sensitive values are explicit.
    def execute(
        self,
        action: Action,
        validate_action: bool = True,
        save_history: bool = True,
        force: bool = False,
    ) -> Action

    # Get current player
    def state.current_player() -> Player

    # Check if game is over
    def winning_color() -> Optional[Color]

    # Game state (read-only)
    game.state: State
```

**Key Properties:**
- `game.state` - Current game state (State object)
- `game.state.playable_actions` - List of valid actions for current player
- `game.state.current_color` - Color of current player
- `game.state.player_state` - Dict with all player state (resources, VP, etc)
- `game.state.board` - Board object (tiles, nodes, edges)

### State (`engine/state.py`)

Immutable game state snapshot.

```python
class State:
    # Current player
    current_color: Color
    current_playable_actions: List[Action]

    # Board
    board: Board

    # Player data (flattened dict)
    player_state: Dict[str, Any]
    # Keys like:
    # "P0_VICTORY_POINTS", "P0_WOOD_IN_HAND", "P0_HAS_ARMY", etc.

    # Buildings by color
    buildings_by_color: Dict[Color, Dict[BuildingType, List[NodeId]]]

    # Resources
    resource_freqdeck: List[Resource]
    development_listdeck: List[DevelopmentCard]

    # Methods
    def current_player() -> Player
```

**Player State Keys:**

Resources (only for P0 - your player):
```
P0_WOOD_IN_HAND
P0_BRICK_IN_HAND
P0_SHEEP_IN_HAND
P0_WHEAT_IN_HAND
P0_ORE_IN_HAND
```

Victory Points (all players):
```
P0_VICTORY_POINTS         # Public VP
P0_ACTUAL_VICTORY_POINTS  # True VP (includes hidden dev cards)
P1_VICTORY_POINTS
P2_VICTORY_POINTS
P3_VICTORY_POINTS
```

Buildings (all players):
```
P0_ROADS_AVAILABLE        # 15 max
P0_SETTLEMENTS_AVAILABLE  # 5 max
P0_CITIES_AVAILABLE       # 4 max
P0_LONGEST_ROAD_LENGTH
```

Special:
```
P0_HAS_ARMY              # Largest army (3+ knights)
P0_HAS_ROAD              # Longest road (5+ continuous)
P0_HAS_ROLLED            # Has rolled dice this turn
```

### Board (`engine/models/board.py`)

Hex board with tiles, nodes, edges.

```python
class Board:
    # Tiles
    map: CatanMap
    robber_coordinate: Coordinate  # [x, y, z]

    # Buildings (dict: node_id -> (color, building_type))
    buildings: Dict[NodeId, Tuple[Color, BuildingType]]

    # Roads (dict: edge_id -> color)
    roads: Dict[EdgeId, Color]

    # Methods
    def get_node_color(node_id: int) -> Optional[Color]
    def get_edge_color(edge: Tuple[int, int]) -> Optional[Color]
    def buildable_node_ids(color: Color, initial_build: bool) -> List[int]
```

**Coordinate System:**

Cube coordinates where `x + y + z = 0`

```python
# Tile at origin
[0, 0, 0]

# Six neighbors (directions)
Direction.NORTHEAST = [1, 0, -1]
Direction.EAST = [1, -1, 0]
Direction.SOUTHEAST = [0, -1, 1]
Direction.SOUTHWEST = [-1, 0, 1]
Direction.WEST = [-1, 1, 0]
Direction.NORTHWEST = [0, 1, -1]
```

### Actions (`engine/models/actions.py`)

All possible game actions.

```python
class ActionType(Enum):
    ROLL = "ROLL"
    MOVE_ROBBER = "MOVE_ROBBER"
    DISCARD = "DISCARD"
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"
    BUILD_ROAD = "BUILD_ROAD"
    BUILD_CITY = "BUILD_CITY"
    BUY_DEVELOPMENT_CARD = "BUY_DEVELOPMENT_CARD"
    PLAY_KNIGHT_CARD = "PLAY_KNIGHT_CARD"
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"
    PLAY_MONOPOLY = "PLAY_MONOPOLY"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"
    MARITIME_TRADE = "MARITIME_TRADE"
    END_TURN = "END_TURN"

class Action:
    action_type: ActionType
    color: Color
    # Action-specific payload
```

**Common Actions:**

```python
# Roll dice
Action(color, ActionType.ROLL)

# Build settlement at node 15
Action(color, ActionType.BUILD_SETTLEMENT, value=15)

# Build road between nodes 10 and 11
Action(color, ActionType.BUILD_ROAD, value=(10, 11))

# Build city at node 15
Action(color, ActionType.BUILD_CITY, value=15)

# Maritime trade (4:1)
Action(color, ActionType.MARITIME_TRADE, value=(WOOD, BRICK))
```

### Player (`engine/models/player.py`)

Player interface - implement to create custom AI.

```python
class Player:
    color: Color

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """Choose an action from playable_actions."""
        raise NotImplementedError

# Built-in players
class SimplePlayer(Player):
    """Always chooses first valid action."""

class WeightedRandomPlayer(Player):
    """Random with preferences for settlements/cities."""
```

---

## JSON Serialization (`engine/json.py`)

Convert game state to JSON for web clients.

```python
from engine.json import GameEncoder
import json

# Serialize entire game
json_str = json.dumps(game, cls=GameEncoder)

# Returns structure matching GAME_STATE_DOCS.md
{
    "tiles": [...],
    "nodes": {...},
    "edges": [...],
    "player_state": {...},
    "current_color": "RED",
    "winning_color": null,
    "longest_roads_by_player": {
        "RED": 5,
        "BLUE": 3,
        "WHITE": 2,
        "ORANGE": 4
    },
    "played_knights_by_player": {
        "RED": 2,
        "BLUE": 0,
        "WHITE": 1,
        "ORANGE": 3
    },
    ...
}
```

**Exposed Aggregates:**

- `longest_roads_by_player` - Dictionary mapping each color to their longest continuous road length
- `played_knights_by_player` - Dictionary mapping each color to the number of knight cards they've played

---

## Helper Functions (`engine/state_functions.py`)

Utility functions for querying state.

```python
from engine.state_functions import (
    get_player_buildings,
    player_key,
    player_num_resource_cards,
    player_num_dev_cards,
)

# Get player's buildings
settlements = get_player_buildings(state, Color.RED, SETTLEMENT)
cities = get_player_buildings(state, Color.RED, CITY)
roads = get_player_buildings(state, Color.RED, ROAD)

# Get player state key prefix
key = player_key(state, Color.RED)  # "P0", "P1", etc.

# Count resources
num_resources = player_num_resource_cards(state, Color.RED)

# Count dev cards
num_dev_cards = player_num_dev_cards(state, Color.RED)
```

---

## Creating Custom Players

```python
from engine.models.player import Player, Color
from engine.models.enums import Action, ActionType

class MyPlayer(Player):
    def decide(self, game, playable_actions):
        # Access game state
        state = game.state

        # My resources (if this is P0)
        my_wood = state.player_state.get("P0_WOOD_IN_HAND", 0)
        my_vp = state.player_state[f"P{self.get_player_index()}_VICTORY_POINTS"]

        # Current board
        board = state.board

        # Prefer building settlements
        for action in playable_actions:
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                return action

        # Otherwise random
        import random
        return random.choice(playable_actions)

    def get_player_index(self):
        # Helper to find your player index
        for i, color in enumerate(game.state.colors):
            if color == self.color:
                return i
        return 0
```

---

## Map Generation (`engine/models/map.py`)

```python
from engine.models.map import build_map

# Standard Catan map (19 hexes, random resource placement)
catan_map = build_map("BASE")

# Mini map (7 hexes, for testing)
catan_map = build_map("MINI")

# Tournament map (balanced resource placement)
catan_map = build_map("TOURNAMENT")

# Use in game
game = Game(players, catan_map=catan_map)
```

---

## Important Enums

```python
from engine.models.enums import (
    RESOURCES,        # [WOOD, BRICK, SHEEP, WHEAT, ORE]
    DEVELOPMENT_CARDS,  # [KNIGHT, VICTORY_POINT, ROAD_BUILDING, ...]
    SETTLEMENT,
    CITY,
    ROAD,
)

# Resource types
WOOD = "WOOD"
BRICK = "BRICK"
SHEEP = "SHEEP"
WHEAT = "WHEAT"
ORE = "ORE"

# Building types
SETTLEMENT = "SETTLEMENT"
CITY = "CITY"
ROAD = "ROAD"

# Dev cards
KNIGHT = "KNIGHT"
VICTORY_POINT = "VICTORY_POINT"
ROAD_BUILDING = "ROAD_BUILDING"
YEAR_OF_PLENTY = "YEAR_OF_PLENTY"
MONOPOLY = "MONOPOLY"
```

---

## Building Costs

```python
from engine.models.enums import SETTLEMENT_BUILD_COST, CITY_BUILD_COST, ROAD_BUILD_COST

# Settlement: 1 wood, 1 brick, 1 sheep, 1 wheat
SETTLEMENT_BUILD_COST = {
    WOOD: 1, BRICK: 1, SHEEP: 1, WHEAT: 1
}

# City: 3 ore, 2 wheat
CITY_BUILD_COST = {
    ORE: 3, WHEAT: 2
}

# Road: 1 wood, 1 brick
ROAD_BUILD_COST = {
    WOOD: 1, BRICK: 1
}

# Dev card: 1 ore, 1 wheat, 1 sheep
DEVELOPMENT_CARD_COST = {
    ORE: 1, WHEAT: 1, SHEEP: 1
}
```

---

## Hex Math

Convert between cube coordinates and pixel positions:

```typescript
// Cube to Axial (for rendering)
const q = cube[0];  // x
const r = cube[2];  // z
// Note: cube[1] is y, where x + y + z = 0

// Axial to Pixel (flat-top hex)
const hexSize = 50;
const x = hexSize * (3/2 * q);
const y = hexSize * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
```

Node positions from tile coordinate + direction:

```typescript
// Corner offsets for flat-top hex
const offsets = {
  'NORTH': { x: hexSize/2, y: -hexSize*√3/2 },
  'NORTHEAST': { x: hexSize, y: 0 },
  'SOUTHEAST': { x: hexSize/2, y: hexSize*√3/2 },
  'SOUTH': { x: -hexSize/2, y: hexSize*√3/2 },
  'SOUTHWEST': { x: -hexSize, y: 0 },
  'NORTHWEST': { x: -hexSize/2, y: -hexSize*√3/2 },
};
```

---

## Game Flow

1. **Initial Placement** (2 rounds)
   - Each player places 2 settlements + 2 roads
   - Second settlement gives starting resources

2. **Main Game Loop**
   - Roll dice (or play dev card first)
   - Distribute resources based on roll
   - Build/trade/buy dev cards
   - End turn

3. **Winning**
   - First to 10 VP wins
   - VP sources: settlements (1), cities (2), longest road (2), largest army (2), VP dev cards (1 each)

---

## Common Queries

**Get all buildings for a player:**
```python
from engine.state_functions import get_player_buildings

settlements = get_player_buildings(state, color, SETTLEMENT)
# Returns: List[int] of node IDs
```

**Check if player can afford something:**
```python
key = player_key(state, color)
wood = state.player_state[f"{key}_WOOD_IN_HAND"]
brick = state.player_state[f"{key}_BRICK_IN_HAND"]

can_build_road = wood >= 1 and brick >= 1
```

**Get resource production for a tile:**
```python
from engine.models.map import number_probability

tile = state.board.map.tiles[coordinate]
prob = number_probability(tile.number)  # 0.0 to 5/36

# 6 and 8 have highest probability (5/36 each)
# 2 and 12 have lowest (1/36 each)
```

---

## File Structure

```
engine/
├── game.py              # Main Game class
├── state.py             # State management
├── json.py              # JSON serialization
├── state_functions.py   # Helper functions
└── models/
    ├── player.py        # Player interface, SimplePlayer
    ├── board.py         # Board, hex grid
    ├── map.py           # Map generation
    ├── actions.py       # Action types
    ├── enums.py         # Resource types, building types
    ├── decks.py         # Resource/dev card decks
    └── coordinate_system.py  # Hex math
```

---

## Integration Examples

**React/Next.js Client:**
```typescript
import type { GameState } from './types';

// Fetch state
const response = await fetch('http://localhost:5001/api/state');
const { game } = await response.json();
const gameState: GameState = game;

// Access data
const tiles = gameState.tiles;
const nodes = Object.values(gameState.nodes);
const myResources = {
  wood: gameState.player_state.P0_WOOD_IN_HAND,
  brick: gameState.player_state.P0_BRICK_IN_HAND,
};
```

**WebSocket (real-time):**
```typescript
import { io } from 'socket.io-client';

const socket = io('http://localhost:5001');

socket.on('game_state', (data) => {
  const gameState = data.game;
  const isRunning = data.running;
  // Update UI
});
```

---

## Tips

1. **Read-only state:** Never mutate `game.state` directly. Use `game.execute(action)`.

2. **Player indices:** P0 is always the first player, P1 second, etc. Map to colors via `state.colors`.

3. **Hidden information:** Only P0's resources are visible in `player_state`. Other players show resource counts but not types.

4. **Node/Edge IDs:** Nodes are 0-53 for base map. Edges are tuples `(node1, node2)` with `node1 < node2`.

5. **Action validation:** Always choose from `state.playable_actions`. Invalid actions will raise errors.

6. **Dice probabilities:**
   - 6, 8: 5/36 (marked red)
   - 5, 9: 4/36
   - 4, 10: 3/36
   - 3, 11: 2/36
   - 2, 12: 1/36

---

## See Also

- [GAME_STATE_DOCS.md](GAME_STATE_DOCS.md) - Complete JSON state reference
- [QUICK_START.md](../guides/QUICK_START.md) - UI integration guide
- [game_state.types.ts](game_state.types.ts) - TypeScript type definitions
- [example_game_state.json](../../example_game_state.json) - Real game state example
