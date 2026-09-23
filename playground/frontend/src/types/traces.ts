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


export type NativeReasoningEffort =
  | 'off'
  | 'minimal'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh'
  | 'max';

