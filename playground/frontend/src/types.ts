/**
 * TypeScript types for Catan game state
 * Copy this into your v0 project
 */

export type Color = "RED" | "BLUE" | "WHITE" | "ORANGE";

export type Resource = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";

export type Building = "SETTLEMENT" | "CITY";

export type Direction =
  | "NORTH"
  | "NORTHEAST"
  | "SOUTHEAST"
  | "SOUTH"
  | "SOUTHWEST"
  | "NORTHWEST"
  | "EAST"
  | "WEST";

export type Coordinate = [number, number, number]; // [x, y, z] cube coordinates

// Tiles
export interface ResourceTile {
  id: number;
  type: "RESOURCE_TILE";
  resource: Resource;
  number: 2 | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 11 | 12;
}

export interface DesertTile {
  id: number;
  type: "DESERT";
}

export interface PortTile {
  id: number;
  type: "PORT";
  direction: Direction;
  resource: Resource | null; // null = 3:1 port
  emoji: string;
  port_nodes: [number, number]; // The 2 node IDs this port connects
}

export type Tile = ResourceTile | DesertTile | PortTile;

export interface PlacedTile {
  coordinate: Coordinate;
  tile: Tile;
}

// Nodes (vertices)
export interface Node {
  id: number;
  tile_coordinate: Coordinate;
  direction: Direction;
  building: Building | null;
  color: Color | null;
}

// Edges (roads)
export interface Edge {
  id: [number, number]; // [node_id_1, node_id_2]
  tile_coordinate: Coordinate;
  direction: Direction;
  color: Color | null;
}

// Actions
export type Action =
  | [Color, "ROLL", null]
  | [Color, "BUILD_SETTLEMENT", number]
  | [Color, "BUILD_CITY", number]
  | [Color, "BUILD_ROAD", [number, number]]
  | [Color, "BUY_DEVELOPMENT_CARD", null]
  | [Color, "PLAY_KNIGHT_CARD", null]
  | [Color, "PLAY_ROAD_BUILDING", null]
  | [Color, "PLAY_MONOPOLY", Resource]
  | [Color, "PLAY_YEAR_OF_PLENTY", [Resource] | [Resource, Resource]]
  | [Color, "MOVE_ROBBER", [Coordinate, Color | null, any]]
  | [Color, "MARITIME_TRADE", any]
  | [Color, "DISCARD", null]
  | [Color, "END_TURN", null];

// Player State (partial - see docs for full list)
export interface PlayerState {
  // Victory Points
  P0_VICTORY_POINTS: number;
  P0_ACTUAL_VICTORY_POINTS: number;
  P1_VICTORY_POINTS: number;
  P2_VICTORY_POINTS: number;
  P3_VICTORY_POINTS: number;

  // Achievements
  P0_HAS_ARMY: boolean;
  P0_HAS_ROAD: boolean;
  P0_LONGEST_ROAD_LENGTH: number;
  P1_HAS_ARMY: boolean;
  P1_HAS_ROAD: boolean;
  P2_HAS_ARMY: boolean;
  P2_HAS_ROAD: boolean;
  P3_HAS_ARMY: boolean;
  P3_HAS_ROAD: boolean;

  // Buildings available
  P0_ROADS_AVAILABLE: number;
  P0_SETTLEMENTS_AVAILABLE: number;
  P0_CITIES_AVAILABLE: number;
  P1_ROADS_AVAILABLE: number;
  P1_SETTLEMENTS_AVAILABLE: number;
  P1_CITIES_AVAILABLE: number;
  P2_ROADS_AVAILABLE: number;
  P2_SETTLEMENTS_AVAILABLE: number;
  P2_CITIES_AVAILABLE: number;
  P3_ROADS_AVAILABLE: number;
  P3_SETTLEMENTS_AVAILABLE: number;
  P3_CITIES_AVAILABLE: number;

  // Resources (only P0 - your hand)
  P0_WOOD_IN_HAND: number;
  P0_BRICK_IN_HAND: number;
  P0_SHEEP_IN_HAND: number;
  P0_WHEAT_IN_HAND: number;
  P0_ORE_IN_HAND: number;

  // Development cards (only P0 - your hand)
  P0_KNIGHT_IN_HAND: number;
  P0_ROAD_BUILDING_IN_HAND: number;
  P0_YEAR_OF_PLENTY_IN_HAND: number;
  P0_MONOPOLY_IN_HAND: number;
  P0_VICTORY_POINT_IN_HAND: number;
  P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN: boolean;

  // Development cards played (public for all)
  P0_PLAYED_KNIGHT: number;
  P0_PLAYED_ROAD_BUILDING: number;
  P0_PLAYED_YEAR_OF_PLENTY: number;
  P0_PLAYED_MONOPOLY: number;
  P1_PLAYED_KNIGHT: number;
  P1_PLAYED_ROAD_BUILDING: number;
  P1_PLAYED_YEAR_OF_PLENTY: number;
  P1_PLAYED_MONOPOLY: number;
  P2_PLAYED_KNIGHT: number;
  P2_PLAYED_ROAD_BUILDING: number;
  P2_PLAYED_YEAR_OF_PLENTY: number;
  P2_PLAYED_MONOPOLY: number;
  P3_PLAYED_KNIGHT: number;
  P3_PLAYED_ROAD_BUILDING: number;
  P3_PLAYED_YEAR_OF_PLENTY: number;
  P3_PLAYED_MONOPOLY: number;

  // Turn state
  P0_HAS_ROLLED: boolean;
  P1_HAS_ROLLED: boolean;
  P2_HAS_ROLLED: boolean;
  P3_HAS_ROLLED: boolean;

  // Other players' hand sizes (hidden info - only totals)
  P1_NUM_RESOURCES_IN_HAND?: number;
  P1_NUM_DEVS_IN_HAND?: number;
  P2_NUM_RESOURCES_IN_HAND?: number;
  P2_NUM_DEVS_IN_HAND?: number;
  P3_NUM_RESOURCES_IN_HAND?: number;
  P3_NUM_DEVS_IN_HAND?: number;

  [key: string]: any; // Allow dynamic access
}

// Main game state
export interface GameState {
  // Board
  tiles: PlacedTile[];
  nodes: Record<string, Node>; // Dict keyed by node ID
  edges: Edge[];
  robber_coordinate: Coordinate;
  adjacent_tiles: Record<string, Tile[]>; // Internal - can ignore

  // Players
  colors: Color[];
  current_color: Color;
  bot_colors: Color[];
  player_state: PlayerState;
  longest_roads_by_player: Record<Color, number>;
  played_knights_by_player: Record<Color, number>;

  // Game flow
  is_initial_build_phase: boolean;
  current_prompt: string;
  current_playable_actions: Action[];
  actions: Action[]; // Full game history
  winning_color: Color | null;
  state_index: number; // Turn number
}

// WebSocket message
export interface GameStateMessage {
  game: GameState;
  running: boolean;
  llm_thinking: LLMDecision[];
  all_player_resources?: Record<Color, Record<Resource, number>>;
}

export interface LLMDecision {
  color: string;
  is_llm: boolean;
  action: string;
  timestamp: number;
  game_over?: boolean;
  winner?: string;
}

export interface ColonistPlayer {
  username: string;
  color: string;  // Colonist color like "red", "blue", etc.
  userId: string;
}

export interface ReplayInfo {
  game_id: string;
  event_index: number;
  total_events: number;
  colonist_players: ColonistPlayer[];
  play_order: number[];
  progress?: string;
}

export interface ReplayLLMAction {
  index: number;
  action: string;
  description: string;
}

export interface ReplayActivityWindow {
  start_replay_index: number;
  end_replay_index: number;
  row_count: number;
  truncated: boolean;
}

export interface TableTalkEntry {
  replayIndex: number;
  player: string;
  message: string;
  model: string;
}

export interface ReplayLLMResponse {
  context_version: string;
  game_id: string;
  replay_index: number;
  player_color: string;
  requested_model: string;
  model: string;
  goals: string;
  reasoning: string;
  message: string;
  action_index: number | null;
  action: string | null;
  action_description: string | null;
  parse_error: string | null;
  finish_reason: string | null;
  response_truncated: boolean;
  observation: string;
  recent_activity: string[];
  activity_window: ReplayActivityWindow;
  available_actions: ReplayLLMAction[];
  raw_response: string;
  latency_ms: number | null;
  usage: Record<string, unknown>;
  system_prompt: string;
  context_prompt: string;
  stale: boolean;
}

// Helper types
export interface HexPosition {
  x: number;
  y: number;
}

export const RESOURCE_COLORS: Record<Resource, string> = {
  WOOD: "#0a5f38",
  BRICK: "#b7410e",
  SHEEP: "#90ee90",
  WHEAT: "#f4c430",
  ORE: "#708090",
};

export const PLAYER_COLORS: Record<Color, string> = {
  RED: "#e74c3c",
  BLUE: "#3498db",
  WHITE: "#ecf0f1",
  ORANGE: "#e67e22",
};

export const BUILDING_COSTS = {
  SETTLEMENT: { WOOD: 1, BRICK: 1, SHEEP: 1, WHEAT: 1 },
  CITY: { WHEAT: 2, ORE: 3 },
  ROAD: { WOOD: 1, BRICK: 1 },
  DEVELOPMENT_CARD: { SHEEP: 1, WHEAT: 1, ORE: 1 },
} as const;
