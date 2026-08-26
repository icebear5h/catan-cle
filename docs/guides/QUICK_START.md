# Quick Start for v0 UI Development

## What You're Building

A real-time Catan game viewer that connects to the Flask server and displays:
1. Hex board with tiles, roads, settlements, cities
2. Player info (VP, resources, dev cards)
3. Thin game controls (start and one complete sandbox step)
4. LLM decision log

## Data Source

**WebSocket:** `http://localhost:5001` (socket.io)
- Event: `game_state`
- Emits on every game update

**REST API:** `http://localhost:5001/api/`
- `POST /api/start-game` - Start new game
- `POST /api/step` - Execute one turn
- `GET /api/state` - Get current state

## Core Data Structure

```typescript
interface GameState {
  // BOARD
  tiles: Array<{
    coordinate: [number, number, number],  // [x, y, z] cube coords
    tile: {
      id: number,
      type: "RESOURCE_TILE" | "DESERT" | "PORT",
      resource?: "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE",
      number?: number  // 2-12 (dice roll)
    }
  }>;

  // BUILDINGS
  nodes: Record<string, {  // Dict keyed by node ID
    id: number,
    building: "SETTLEMENT" | "CITY" | null,
    color: "RED" | "BLUE" | "WHITE" | "ORANGE" | null
  }>;

  // ROADS
  edges: Array<{
    id: [number, number],  // [nodeId1, nodeId2]
    color: "RED" | "BLUE" | "WHITE" | "ORANGE" | null
  }>;

  // ROBBER
  robber_coordinate: [number, number, number];

  // PLAYERS
  player_state: {
    P0_VICTORY_POINTS: number,
    P0_WOOD_IN_HAND: number,
    P0_BRICK_IN_HAND: number,
    // ... see full docs
  };

  // GAME FLOW
  colors: ["RED", "BLUE", "WHITE", "ORANGE"],
  current_color: "RED" | "BLUE" | "WHITE" | "ORANGE",
  winning_color: string | null,
  current_prompt: string,  // "Roll dice", "Build settlement", etc
  is_initial_build_phase: boolean
}
```

## Minimal Working Example

```tsx
import { useEffect, useState } from 'react';
import { io } from 'socket.io-client';

export default function CatanGame() {
  const [gameState, setGameState] = useState(null);
  const [socket, setSocket] = useState(null);

  // Connect to WebSocket
  useEffect(() => {
    const newSocket = io('http://localhost:5001');

    newSocket.on('game_state', (data) => {
      setGameState(data.game);
    });

    setSocket(newSocket);
    return () => newSocket.close();
  }, []);

  // Start game
  const startGame = async () => {
    await fetch('http://localhost:5001/api/start-game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: "random" })
    });
  };

  if (!gameState) {
    return <button onClick={startGame}>Start Game</button>;
  }

  return (
    <div>
      <h1>Catan Game Viewer</h1>

      {/* Game Board */}
      <svg width="800" height="600">
        {gameState.tiles.map(({ coordinate, tile }) => {
          const [x, y, z] = coordinate;
          const pixelX = 400 + (x * 60);
          const pixelY = 300 + (z * 52);

          return (
            <g key={`${x},${y},${z}`}>
              {/* Hexagon */}
              <circle cx={pixelX} cy={pixelY} r={30} fill="#ddd" />

              {/* Dice number */}
              {tile.type === "RESOURCE_TILE" && (
                <text x={pixelX} y={pixelY} textAnchor="middle">
                  {tile.number}
                </text>
              )}
            </g>
          );
        })}

        {/* Settlements */}
        {Object.values(gameState.nodes).map((node) => {
          if (!node.building) return null;

          // You'll need proper node positioning - see docs
          return (
            <circle
              key={node.id}
              cx={100}  // Calculate from node position
              cy={100}
              r={node.building === "CITY" ? 8 : 5}
              fill={node.color.toLowerCase()}
            />
          );
        })}
      </svg>

      {/* Player Info */}
      <div>
        {gameState.colors.map((color, i) => (
          <div key={color}>
            <h3>{color}</h3>
            <div>VP: {gameState.player_state[`P${i}_VICTORY_POINTS`]}</div>
          </div>
        ))}
      </div>

      {/* Winner */}
      {gameState.winning_color && (
        <h2>Winner: {gameState.winning_color}!</h2>
      )}
    </div>
  );
}
```

## Hex Coordinate Math

```javascript
// Convert cube coordinates to pixel position
function hexToPixel(q, r, s, size = 50) {
  const x = size * (3/2 * q);
  const y = size * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
  return { x, y };
}

// Use it
const [q, r, s] = coordinate;  // [x, y, z]
const { x, y } = hexToPixel(q, s, r, size);
```

## Drawing a Hexagon

```javascript
function Hexagon({ x, y, size, fill }) {
  const points = Array.from({ length: 6 }, (_, i) => {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    return [
      x + size * Math.cos(angle),
      y + size * Math.sin(angle)
    ];
  });

  return (
    <polygon
      points={points.map(p => p.join(',')).join(' ')}
      fill={fill}
      stroke="#333"
      strokeWidth={2}
    />
  );
}
```

## Resource Colors & Emojis

```javascript
const RESOURCE_COLORS = {
  WOOD: "#0a5f38",    // Dark green
  BRICK: "#b7410e",   // Orange-red
  SHEEP: "#90ee90",   // Light green
  WHEAT: "#f4c430",   // Gold
  ORE: "#708090"      // Slate gray
};

const RESOURCE_EMOJIS = {
  WOOD: "🪵",
  BRICK: "🧱",
  SHEEP: "🐑",
  WHEAT: "🌾",
  ORE: "⛰️"
};

// Ports include an emoji field for easy display
// port.emoji will be "🪵", "🧱", "🐑", "🌾", "⛰️", or "3:1"
```

## What to Show

**Essential:**
- Hex tiles with resource type and dice numbers
- Player settlements (small circles) and cities (larger circles)
- Roads connecting nodes
- Robber (black circle on a tile)
- Current player's turn indicator
- Victory points per player

**Nice to have:**
- Player resource counts (only P0 visible)
- Current game phase (initial placement vs normal play)
- LLM decision log
- Build costs reference
- Longest road / largest army indicators

## File References

- Full docs: [GAME_STATE_DOCS.md](../engine/GAME_STATE_DOCS.md)
- Example JSON: [example_game_state.json](../../example_game_state.json)
- Flask server: `cle/eval/game_viewer_server.py`
- Game engine: `game_engine/`

## Test the Server

```bash
# Server should be running at http://localhost:5001
curl http://localhost:5001/api/health

# Start a game
curl -X POST http://localhost:5001/api/start-game \
  -H "Content-Type: application/json" \
  -d '{"mode": "random"}'

# Get state
curl http://localhost:5001/api/state
```

## Common Gotchas

1. **Nodes is a dict, not array** - Use `Object.values(gameState.nodes)` to iterate
2. **Coordinates are [x, y, z]** - Not [q, r] like some hex libraries
3. **Player state uses prefixes** - P0, P1, P2, P3 (not indexed by color)
4. **Only P0 resources visible** - Other players show total count only
5. **Edge IDs are unsorted** - `[5, 3]` and `[3, 5]` might both appear

## Next Steps

1. Get basic hex grid rendering
2. Add settlements/cities/roads
3. Wire up WebSocket for live updates
4. Add player panels
5. Style it up
6. Add game controls
