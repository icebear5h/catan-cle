import type {
  AllPlayerResources,
  Color,
  Coordinate,
  Edge,
  Node,
  PlacedTile,
  Resource,
  Tile,
} from './primitives';

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
  all_player_resources?: AllPlayerResources;
}

