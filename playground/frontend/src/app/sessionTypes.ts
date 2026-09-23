import type { LiveStepFailure, LiveStepWarning } from '../liveStepErrors';
import type { AllPlayerDevCards } from '../playerDevCards';
import type { AllPlayerHands } from '../playerHands';
import type {
  AllPlayerResources,
  GameState,
  ReplayInfo,
} from '../types';

export interface ReplayCursor {
  gameId: string;
  eventIndex: number;
}

export interface GameLogEntry {
  type: 'dice' | 'resource' | 'building' | 'trade' | 'robber' | 'general' | 'message';
  timestamp: number;
  message: string;
  color?: string;
  step_index?: number | null;
  details?: unknown;
}

export interface ReplayStepResult {
  action?: string;
  engine_translation?: unknown;
  colonist_event?: unknown;
}

export interface LiveInferenceState {
  model: string | null;
  reasoning: {
    enabled?: boolean;
    effort?: string;
    exclude?: boolean;
  };
  max_tokens: number | null;
  max_decision_attempts: number;
}

export interface StateSnapshot {
  game: GameState | null;
  running: boolean;
  live_trace_game_id?: string | null;
  live_inference?: LiveInferenceState | null;
  last_live_step_error?: LiveStepFailure | LiveStepWarning | null;
  game_log?: GameLogEntry[];
  all_player_resources?: AllPlayerResources | null;
  all_player_dev_cards?: AllPlayerDevCards | null;
  // Spectator hand contents alongside the public totals above. Absent on
  // snapshots recorded before the field existed, so always optional.
  player_hands?: AllPlayerHands | null;
  player_types?: Record<string, string> | null;
  replay_mode?: boolean;
  replay?: ReplayInfo | null;
  last_dice_roll?: [number, number] | null;
}
