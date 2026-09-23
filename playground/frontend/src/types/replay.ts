import type { TraceRequest, TraceRequestMetadata } from './traces';

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

