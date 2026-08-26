# Catan Game State Documentation

Complete reference for building a Catan UI from the game engine state.

## Overview

The game state is exposed via the Flask server at `http://localhost:5001/api/state` and broadcast via WebSocket on every update. It uses JSON serialization from [engine/json.py](../../engine/json.py).

---

## Top-Level Structure

```typescript
interface GameState {
  tiles: PlacedTile[];           // Hex tiles with resources
  nodes: Record<string, Node>;   // Vertices (settlement/city locations)
  edges: Edge[];                 // Edges (road locations)
  player_state: PlayerState;     // Resources, VP, dev cards
  colors: Color[];               // Player colors in turn order
  current_color: Color;          // Whose turn it is
  robber_coordinate: [number, number, number];  // Robber position
  is_initial_build_phase: boolean;
  current_prompt: string;        // "Roll dice", "Build a settlement", etc
  current_playable_actions: Action[];  // Valid moves right now
  actions: Action[];             // Full game history
  winning_color: Color | null;  // Winner (if game over)
  state_index: number;           // Turn number
  bot_colors: Color[];           // Which players are bots
  longest_roads_by_player: Record<Color, number>;
  adjacent_tiles: Record<string, any>;  // Internal - ignore
}
```

---

## 1. Board Geometry (Hex Grid)

### Tiles (Hexagons)

```typescript
interface PlacedTile {
  coordinate: [number, number, number];  // Cube coordinates [x, y, z]
  tile: ResourceTile | DesertTile | PortTile;
}

interface ResourceTile {
  id: number;
  type: "RESOURCE_TILE";
  resource: "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";
  number: 2 | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 11 | 12;  // Dice roll
}

interface DesertTile {
  id: number;
  type: "DESERT";
}

interface PortTile {
  id: number;
  type: "PORT";
  direction: Direction;
  resource: "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE" | null;  // null = 3:1 port
  emoji: string;  // "🪵" | "🧱" | "🐑" | "🌾" | "⛰️" | "3:1"
}
```

**Example:**
```json
{
  "coordinate": [0, 0, 0],
  "tile": {
    "id": 0,
    "type": "RESOURCE_TILE",
    "resource": "WOOD",
    "number": 10
  }
}
```

**Coordinate System:**
- Uses **cube coordinates** where `x + y + z = 0`
- Center tile: `[0, 0, 0]`
- Six neighbors of `[0,0,0]`:
  - `[1, -1, 0]` (East)
  - `[1, 0, -1]` (Northeast)
  - `[0, 1, -1]` (Northwest)
  - `[-1, 1, 0]` (West)
  - `[-1, 0, 1]` (Southwest)
  - `[0, -1, 1]` (Southeast)

**Rendering Hexagons:**
```javascript
// Convert cube coordinates to pixel position
function hexToPixel(q, r, s, size) {
  const x = size * (3/2 * q);
  const y = size * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
  return { x, y };
}

// Or use q = x, r = z
const { x, y } = hexToPixel(coord[0], coord[2], coord[1], hexSize);
```

### Nodes (Vertices)

```typescript
interface Node {
  id: number;                    // Unique node ID (0-53 for standard map)
  tile_coordinate: [number, number, number];
  direction: Direction;          // Which corner of the tile
  building: "SETTLEMENT" | "CITY" | null;
  color: Color | null;           // Owner color
}

type Direction =
  | "NORTH" | "NORTHEAST" | "SOUTHEAST"
  | "SOUTH" | "SOUTHWEST" | "NORTHWEST"
  | "EAST" | "WEST";  // For edges/ports
```

**Example:**
```json
{
  "0": {
    "id": 0,
    "tile_coordinate": [1, 0, -1],
    "direction": "SOUTHWEST",
    "building": "SETTLEMENT",
    "color": "BLUE"
  }
}
```

**Key Points:**
- Each node is at the corner where 3 tiles meet
- `nodes` is a **dict** keyed by node ID (convert to array for iteration)
- Settlements are worth 1 VP, Cities are worth 2 VP
- Initial placement: 2 settlements, 2 roads per player

### Edges (Roads)

```typescript
interface Edge {
  id: [number, number];          // Two node IDs this edge connects
  tile_coordinate: [number, number, number];
  direction: Direction;
  color: Color | null;           // Owner color
}
```

**Example:**
```json
{
  "id": [1, 2],
  "tile_coordinate": [1, -1, 0],
  "direction": "WEST",
  "color": "RED"
}
```

**Rendering Roads:**
- Draw a line between `nodes[edge.id[0]]` and `nodes[edge.id[1]]`
- Use `color` to determine player ownership

### Robber

```typescript
robber_coordinate: [number, number, number];  // Tile coordinate
```

**Purpose:**
- Blocks resource production on that tile
- Starts on desert, moves when 7 is rolled or knight played
- Render as a black circle/pawn on the tile

---

## 2. Player State

```typescript
interface PlayerState {
  // Victory Points (Public)
  "P0_VICTORY_POINTS": number;           // Visible VP (settlements + cities + cards)
  "P0_ACTUAL_VICTORY_POINTS": number;    // True VP (includes hidden dev cards)

  // Special Achievements
  "P0_HAS_ARMY": boolean;                // Largest army (3+ knights)
  "P0_HAS_ROAD": boolean;                // Longest road (5+ roads)
  "P0_LONGEST_ROAD_LENGTH": number;

  // Buildings Left
  "P0_ROADS_AVAILABLE": number;          // Max 15
  "P0_SETTLEMENTS_AVAILABLE": number;    // Max 5
  "P0_CITIES_AVAILABLE": number;         // Max 4

  // Resources (ONLY for P0 - your player)
  "P0_WOOD_IN_HAND": number;
  "P0_BRICK_IN_HAND": number;
  "P0_SHEEP_IN_HAND": number;
  "P0_WHEAT_IN_HAND": number;
  "P0_ORE_IN_HAND": number;

  // Development Cards (ONLY for P0)
  "P0_KNIGHT_IN_HAND": number;
  "P0_ROAD_BUILDING_IN_HAND": number;
  "P0_YEAR_OF_PLENTY_IN_HAND": number;
  "P0_MONOPOLY_IN_HAND": number;
  "P0_VICTORY_POINT_IN_HAND": number;    // Hidden VP cards
  "P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN": boolean;

  // Development Cards Played (Public for all players)
  "P0_PLAYED_KNIGHT": number;
  "P0_PLAYED_ROAD_BUILDING": number;
  "P0_PLAYED_YEAR_OF_PLENTY": number;
  "P0_PLAYED_MONOPOLY": number;

  // Turn State
  "P0_HAS_ROLLED": boolean;

  // Repeat for P1, P2, P3...
}
```

**Player Indexing:**
- `P0` = You (or first player in turn order from your perspective)
- `P1`, `P2`, `P3` = Other players in clockwise order
- **Hidden information:** Only P0's resources and dev cards are visible
- Other players show `P1_NUM_RESOURCES_IN_HAND` (total count only)

**Colors Mapping:**
```typescript
colors: ["RED", "BLUE", "WHITE", "ORANGE"];  // Turn order
current_color: "RED";  // Whose turn it is
```

---

## 3. Game Flow

### Current Turn State

```typescript
{
  current_color: "RED",
  current_prompt: "Roll dice" | "Build a settlement" | "Move robber" | "Discard resources",
  current_playable_actions: Action[],
  is_initial_build_phase: boolean,
  state_index: number  // Turn number
}
```

**Initial Build Phase:**
- Each player places 2 settlements + 2 roads
- Order: P0 → P1 → P2 → P3 → P3 → P2 → P1 → P0 (snake draft)
- `is_initial_build_phase: true` during this phase

**Normal Turn Flow:**
1. Roll dice (`current_prompt: "Roll dice"`)
2. Collect resources (automatic)
3. Trade/Build (`current_prompt: "End turn"` - can do actions)
4. End turn

### Actions

```typescript
type Action =
  | [Color, "ROLL", null]
  | [Color, "BUILD_SETTLEMENT", number]           // node_id
  | [Color, "BUILD_CITY", number]                 // node_id
  | [Color, "BUILD_ROAD", [number, number]]       // edge as [node1, node2]
  | [Color, "BUY_DEVELOPMENT_CARD", null]
  | [Color, "PLAY_KNIGHT_CARD", null]
  | [Color, "MOVE_ROBBER", [[x,y,z], Color?, any]]  // coordinate, victim color
  | [Color, "MARITIME_TRADE", any]
  | [Color, "END_TURN", null]
  | [Color, "DISCARD", null];
```

**Example:**
```json
["RED", "BUILD_SETTLEMENT", 5]
["BLUE", "BUILD_ROAD", [3, 7]]
["WHITE", "MOVE_ROBBER", [[0, 0, 0], "RED", null]]
```

---

## 4. Building Costs

```typescript
const COSTS = {
  SETTLEMENT: { WOOD: 1, BRICK: 1, SHEEP: 1, WHEAT: 1 },
  CITY: { WHEAT: 2, ORE: 3 },
  ROAD: { WOOD: 1, BRICK: 1 },
  DEVELOPMENT_CARD: { SHEEP: 1, WHEAT: 1, ORE: 1 }
};
```

---

## 5. Rendering the Board

### Minimal Board Component

```typescript
interface GameBoardProps {
  gameState: GameState;
}

function GameBoard({ gameState }: GameBoardProps) {
  return (
    <svg width="800" height="600">
      {/* Render tiles */}
      {gameState.tiles.map(({ coordinate, tile }) => (
        <HexTile
          key={`${coordinate}`}
          coord={coordinate}
          tile={tile}
          hasRobber={arraysEqual(coordinate, gameState.robber_coordinate)}
        />
      ))}

      {/* Render roads */}
      {gameState.edges.map(edge => (
        edge.color && (
          <Road
            key={`${edge.id}`}
            from={gameState.nodes[edge.id[0]]}
            to={gameState.nodes[edge.id[1]]}
            color={edge.color}
          />
        )
      ))}

      {/* Render settlements/cities */}
      {Object.values(gameState.nodes).map(node => (
        node.building && (
          <Building
            key={node.id}
            node={node}
            type={node.building}
            color={node.color}
          />
        )
      ))}
    </svg>
  );
}
```

### Hex Tile Example

```typescript
function HexTile({ coord, tile, hasRobber }) {
  const [q, r, s] = coord;
  const size = 50;
  const { x, y } = hexToPixel(q, s, r, size);

  // Generate hex points
  const points = Array.from({ length: 6 }, (_, i) => {
    const angle = (Math.PI / 3) * i;
    return `${x + size * Math.cos(angle)},${y + size * Math.sin(angle)}`;
  }).join(' ');

  return (
    <g>
      <polygon points={points} fill={getResourceColor(tile.resource)} />
      {tile.type === "RESOURCE_TILE" && (
        <text x={x} y={y}>{tile.number}</text>
      )}
      {hasRobber && <circle cx={x} cy={y} r={15} fill="black" />}
    </g>
  );
}

function getResourceColor(resource: string) {
  const colors = {
    WOOD: "#0a5f38",
    BRICK: "#b7410e",
    SHEEP: "#90ee90",
    WHEAT: "#f4c430",
    ORE: "#708090"
  };
  return colors[resource] || "#d2b48c"; // Desert
}
```

---

## 6. Player Info Panel

```typescript
function PlayerPanel({ gameState, playerIndex }: { gameState: GameState, playerIndex: number }) {
  const prefix = `P${playerIndex}`;
  const color = gameState.colors[playerIndex];
  const isCurrentTurn = color === gameState.current_color;

  return (
    <div className={isCurrentTurn ? "active-player" : ""}>
      <h3>{color}</h3>
      <div>VP: {gameState.player_state[`${prefix}_VICTORY_POINTS`]}/10</div>
      <div>Roads: {15 - gameState.player_state[`${prefix}_ROADS_AVAILABLE`]}/15</div>
      <div>Settlements: {5 - gameState.player_state[`${prefix}_SETTLEMENTS_AVAILABLE`]}/5</div>
      <div>Cities: {4 - gameState.player_state[`${prefix}_CITIES_AVAILABLE`]}/4</div>

      {gameState.player_state[`${prefix}_HAS_ARMY`] && <div>⚔️ Largest Army</div>}
      {gameState.player_state[`${prefix}_HAS_ROAD`] && <div>🛣️ Longest Road</div>}

      {/* Only show resources for P0 (you) */}
      {playerIndex === 0 && (
        <div>
          <h4>Resources</h4>
          <div>🪵 Wood: {gameState.player_state.P0_WOOD_IN_HAND}</div>
          <div>🧱 Brick: {gameState.player_state.P0_BRICK_IN_HAND}</div>
          <div>🐑 Sheep: {gameState.player_state.P0_SHEEP_IN_HAND}</div>
          <div>🌾 Wheat: {gameState.player_state.P0_WHEAT_IN_HAND}</div>
          <div>⛰️ Ore: {gameState.player_state.P0_ORE_IN_HAND}</div>
        </div>
      )}
    </div>
  );
}
```

---

## 7. WebSocket Integration

```typescript
import { io } from 'socket.io-client';

const socket = io('http://localhost:5001');

socket.on('game_state', (data) => {
  // data.game = public game projection
  // data.running = boolean
  setGameState(data.game);
});
```

---

## 8. Common Patterns

### Check if game is over
```typescript
if (gameState.winning_color) {
  console.log(`${gameState.winning_color} wins!`);
}
```

### Get current player
```typescript
const currentPlayer = gameState.colors.indexOf(gameState.current_color);
const currentPlayerVP = gameState.player_state[`P${currentPlayer}_VICTORY_POINTS`];
```

### Find all settlements owned by a player
```typescript
const redSettlements = Object.values(gameState.nodes)
  .filter(node => node.color === "RED" && node.building === "SETTLEMENT");
```

### Calculate resource production for a tile
```typescript
function getProductionValue(number: number): number {
  const dots = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1
  };
  return dots[number] || 0;
}
```

---

## 9. API Endpoints

### Start Game
```http
POST /api/start-game
Content-Type: application/json

{
  "mode": "random"
}

Response: { "status": "started", "players": ["RED", "BLUE", "WHITE", "ORANGE"] }
```

### Execute One Step
```http
POST /api/step

Response: { "status": "ok", "action": "...", "game_over": false }
```

### Get Current State
```http
GET /api/state

Response: { "game": GameState, "running": true }
```

---

## Quick Reference

**Resource Types:** `WOOD`, `BRICK`, `SHEEP`, `WHEAT`, `ORE`
**Building Types:** `SETTLEMENT`, `CITY`
**Colors:** `RED`, `BLUE`, `WHITE`, `ORANGE`
**Dice Numbers:** `2, 3, 4, 5, 6, 8, 9, 10, 11, 12` (no 7)

**Win Condition:** First to 10 Victory Points
**VP Sources:**
- Settlement = 1 VP
- City = 2 VP
- Longest Road (5+ roads) = 2 VP
- Largest Army (3+ knights) = 2 VP
- Victory Point dev cards = 1 VP each

**Standard Map:** 19 land tiles (18 resource + 1 desert), 9 ports
