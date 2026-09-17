/**
 * TypeScript types for Catan game state
 * Copy this into your v0 project
 */

export type Color =
  | "RED"
  | "BLUE"
  | "ORANGE"
  | "WHITE"
  | "BLACK"
  | "GREEN"
  | "BRONZE"
  | "SILVER"
  | "GOLD"
  | "PINK"
  | "MYSTIC_BLUE";

export type LiveColorPalette = "random_all" | "canonical_four";

export type Resource = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";

export type PlayerResourceCounts = Partial<Record<Resource, number>> & {
  TOTAL?: number;
};

export type AllPlayerResources = Partial<Record<Color, PlayerResourceCounts>>;

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
  all_player_resources?: AllPlayerResources;
}

export interface TraceRequestMetadata {
  context_policy?: 'fresh_notes' | null;
  memory_revision?: number | null;
  input_next_sequence?: number | null;
  channel?: string | null;
  trigger_reason?: string | null;
}

export interface TraceRequest extends TraceRequestMetadata {
  decision_id: string;
  session_id: string;
  messages: Array<{ role: string; content: string }>;
  components?: Array<{
    id: string;
    channel: 'system' | 'environment';
    template: string;
    value: string;
    rendered: string;
    variables?: Record<string, string> | Array<[string, string]>;
  }>;
  board_presentation?: unknown;
}

export interface LiveReasoningTrace extends TraceRequestMetadata {
  schema: 'live-reasoning-trace-v2';
  context_id: string;
  player_color: Color;
  turn_number: number | null;
  phase: string | null;
  prompt_key: string | null;
  call_kind?: 'decision' | 'communication';
  accepted?: boolean;
  request?: TraceRequest | null;
  notes_update?: string | null;
  communication_mode?: string;
  respondents?: Color[] | null;
  text?: string;
  action_index: number | null;
  action_type: string | null;
  action_sequence?: string[];
  batch_actions?: { tool: string; arguments: Record<string, unknown> }[];
  knight_destination?: [number, number, number] | null;
  game_plan: string;
  native_reasoning: string;
  native_reasoning_details: unknown[];
  native_reasoning_source: 'provider_response' | null;
  native_reasoning_requested: boolean;
  native_reasoning_returned: boolean;
  native_reasoning_missing: boolean;
  reasoning_request: Record<string, unknown>;
  reasoning_tokens: number | null;
  finish_reason: string | null;
  provider_native_finish_reason: string | null;
  provider_response_id: string | null;
  provider_request_id: string | null;
  model: string | null;
  latency_ms: number | null;
  usage: Record<string, unknown>;
}

export interface ColonistPlayer {
  username: string;
  color: string;  // Colonist color like "red", "blue", etc.
  userId: string;
}

export interface ReplayTranscriptSegment {
  start_s: number;
  end_s: number;
  text: string;
  source_start_index: number;
  source_end_index: number;
  source_segment_count: number;
}

export interface ReplayTranscriptWindow {
  schema: string;
  alignment_version: string;
  video_id: string | null;
  video_url: string | null;
  pairing_status: string;
  verified: boolean;
  narrator: {
    username: string | null;
    colonist_color: number | null;
    status: string | null;
  };
  replay_index: number;
  status: 'ready' | 'empty' | 'clock_anomaly' | 'complete' | 'unavailable';
  window_start_s: number | null;
  window_end_s: number | null;
  clock_anomaly: boolean;
  segments: ReplayTranscriptSegment[];
  history_raw_segment_count: number;
  history_segments: ReplayTranscriptSegment[];
}

export type ReplayNarratorReasoningKind =
  | 'decision_reasoning'
  | 'board_observation'
  | 'opponent_assessment'
  | 'reaction'
  | 'reflection';

export type ReplayNarratorReasoningAnchor =
  | 'decision'
  | 'observation'
  | 'complete';

export interface ReplayNarratorReasoningParagraph {
  paragraph_id: string;
  kind: ReplayNarratorReasoningKind;
  text: string;
  evidence_ids: string[];
  uncertainties: string[];
  start_s: number;
  end_s: number;
  source_start_index: number;
  source_end_index: number;
  subject_replay_index: number;
  available_replay_index: number;
  anchor_kind: ReplayNarratorReasoningAnchor;
  decision_ids: string[];
}

export interface ReplayNarratorReasoningGroup {
  subject_replay_index: number;
  available_replay_index: number;
  anchor_kind: ReplayNarratorReasoningAnchor;
  decision_ids: string[];
  paragraphs: ReplayNarratorReasoningParagraph[];
}

export interface ReplayNarratorReasoningWindow {
  schema: string;
  game_id: string;
  replay_index: number;
  model_id: string;
  model_label: string;
  generator_version: string;
  narrator: {
    username: string | null;
    colonist_color: number | null;
    status?: string | null;
  };
  artifact_error: string | null;
  complete: boolean;
  strict_causal: boolean;
  decision_count: number;
  anchor_count: number;
  anchor_kind: ReplayNarratorReasoningAnchor | null;
  decision_ids: string[];
  expected_window_count: number;
  loaded_window_count: number;
  status: 'ready' | 'empty' | 'error' | 'pending' | 'no_commentary' | 'complete' | 'unavailable' | 'artifact_error';
  recorded_at?: string | null;
  error: { type?: string; message?: string } | null;
  paragraphs: ReplayNarratorReasoningParagraph[];
  history_paragraphs: ReplayNarratorReasoningParagraph[];
  history_groups: ReplayNarratorReasoningGroup[];
}

export interface ReplayModelTraceSelection {
  index: number | null;
  action: string | null;
  description: string | null;
}

export interface ReplayModelTrace {
  schema: string;
  status: 'ready' | 'error';
  decision_id: string;
  game_id: string;
  replay_index: number;
  available_replay_index: number;
  source_replay_index: number;
  canonical_replay_index: number;
  alignment: 'pre_action' | 'pre_action_reordered_event' | 'post_event_reveal';
  trace_source: 'base_action_diff' | 'setup_strategy_override';
  model_id: string;
  model: string;
  model_label: string;
  player: {
    username: string | null;
    colonist_color: number;
    engine_color: string;
  };
  context_version: string | null;
  setup_strategy_version: string | null;
  setup_stage: string | null;
  generation_max_tokens: number | null;
  phase: string | null;
  forced: boolean;
  legal_action_count: number;
  selection: ReplayModelTraceSelection;
  goals: string;
  reasoning: string;
  message: string;
  reasoning_source: 'qwen_action_response' | 'qwen_self_review';
  reasoning_recorded_at: string | null;
  draft_goals: string | null;
  draft_reasoning: string | null;
  parse_warning: string | null;
  quality_warnings: string[];
  response_truncated: boolean;
  latency_ms: number | null;
  recorded_at: string | null;
  error: { type?: string; message?: string } | null;
}

export interface ReplayModelTraceWindow {
  schema: string;
  game_id: string;
  replay_index: number;
  model_id: string;
  model_label: string;
  player: {
    username: string | null;
    colonist_color: number;
    engine_color: string;
  };
  policy: {
    context_version: string | null;
    stateless_goals: boolean | null;
    allow_lookahead: boolean | null;
    execute_model_actions: boolean | null;
  };
  state_provenance: {
    archived_player_perspective: number | null;
    target_matches_archive_perspective: boolean;
    private_state_status: 'captured_player_view' | 'reconstructed_non_capture_view';
  };
  artifact_partial: boolean;
  artifact_error: string | null;
  complete: boolean;
  expected_trace_count: number;
  ready_trace_count: number;
  pending_trace_count: number;
  status: 'ready' | 'error' | 'pending' | 'no_decision' | 'complete' | 'unavailable' | 'artifact_error';
  traces: ReplayModelTrace[];
}

export interface ReplayInfo {
  game_id: string;
  event_index: number;
  total_events: number;
  colonist_players: ColonistPlayer[];
  play_order: number[];
  progress?: string;
  paired_transcript?: ReplayTranscriptWindow;
  paired_narrator_reasoning?: ReplayNarratorReasoningWindow;
  paired_model_trace?: ReplayModelTraceWindow;
}

export type NativeReasoningEffort =
  | 'off'
  | 'minimal'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh'
  | 'max';

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
  // Engine-event sequence, NOT a live-trace step index: one step emits
  // several engine events (action + speech + trade responses), so sequences
  // run higher than the step count.
  sequence: number;
  // Recorded trace step - the /api/step advance this message was spoken in.
  // Null for rows logged without a trace store, or before step stamping.
  step_index: number | null;
  player: string;
  message: string;
  model: string;
}

export interface ReplayLLMResponse extends TraceRequestMetadata {
  schema: 'agent-decision-preview-v2';
  context_version: string;
  context_id: string;
  game_id: string;
  replay_index: number;
  phase: string;
  prompt_key: string;
  player_color: string;
  requested_model: string;
  model: string;
  generation_max_tokens: number;
  game_plan: string;
  notes_update?: string | null;
  accepted?: false;
  request?: TraceRequest | null;
  action_index: number | null;
  action: string | null;
  action_description: string | null;
  parse_error: string | null;
  finish_reason: string | null;
  provider_native_finish_reason: string | null;
  provider_response_id: string | null;
  provider_request_id: string | null;
  response_truncated: boolean;
  native_reasoning: string;
  native_reasoning_details: unknown[];
  native_reasoning_returned: boolean;
  native_reasoning_missing: boolean;
  reasoning_request: Record<string, unknown>;
  reasoning_tokens: number | null;
  observation: string;
  available_actions: ReplayLLMAction[];
  raw_response: string;
  latency_ms: number | null;
  usage: Record<string, unknown>;
  model_messages: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
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
  BLACK: "#2c3e50",
  GREEN: "#27ae60",
  BRONZE: "#cd7f32",
  SILVER: "#c0c0c0",
  GOLD: "#ffd700",
  PINK: "#ec4899",
  MYSTIC_BLUE: "#c7e5fd",
};

export const BUILDING_COSTS = {
  SETTLEMENT: { WOOD: 1, BRICK: 1, SHEEP: 1, WHEAT: 1 },
  CITY: { WHEAT: 2, ORE: 3 },
  ROAD: { WOOD: 1, BRICK: 1 },
  DEVELOPMENT_CARD: { SHEEP: 1, WHEAT: 1, ORE: 1 },
} as const;
