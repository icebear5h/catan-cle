/**
 * TypeScript types for Catan game state
 * Copy this into your v0 project
 */

export type {
  AllPlayerResources,
  Building,
  Color,
  Coordinate,
  DesertTile,
  Direction,
  Edge,
  HexPosition,
  LiveColorPalette,
  Node,
  PlacedTile,
  PlayerResourceCounts,
  PortTile,
  Resource,
  ResourceTile,
  Tile,
} from './primitives';

export type {
  Action,
  GameState,
  GameStateMessage,
  PlayerState,
} from './game';

export type {
  NativeReasoningEffort,
  TraceRequest,
  TraceRequestMetadata,
} from './traces';

export type {
  ColonistPlayer,
  ReplayActivityWindow,
  ReplayInfo,
  ReplayLLMAction,
  ReplayLLMResponse,
  ReplayModelTrace,
  ReplayModelTraceSelection,
  ReplayModelTraceWindow,
  ReplayNarratorReasoningAnchor,
  ReplayNarratorReasoningGroup,
  ReplayNarratorReasoningKind,
  ReplayNarratorReasoningParagraph,
  ReplayNarratorReasoningWindow,
  ReplayTranscriptSegment,
  ReplayTranscriptWindow,
  TableTalkEntry,
} from './replay';

export type { LiveReasoningTrace } from './live';

export { BUILDING_COSTS, PLAYER_COLORS, RESOURCE_COLORS } from './constants';
