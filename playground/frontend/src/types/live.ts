import type { Color } from './primitives';
import type { TraceRequest, TraceRequestMetadata } from './traces';

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

