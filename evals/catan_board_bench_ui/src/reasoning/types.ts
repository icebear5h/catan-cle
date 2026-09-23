import type { GameState } from '@playground/types'

type TraceIndexRow = {
  trace_id: string
  seed: number
  model_id: string
  prompt_sha256: string
  finish_reason: string | null
  native_reasoning_returned: boolean
  reasoning_tokens: number | null
  final_response_characters: number
  parsed_action_index: number | null
  parse_error: string | null
  latency_ms: number | null
  cost_usd: number
}

type SeedStatus = {
  seed: number
  status: 'complete' | 'partial' | 'not_started'
  captured: number
  expected: number
  missing_models: string[]
}

type ReasoningRun = {
  id: string
  title: string
  description: string
  available: boolean
  excluded_models: Record<string, string>
}

export type TraceOverview = {
  schema: string
  run_id: string
  run_title: string
  run_description: string
  available_runs: ReasoningRun[]
  excluded_models: Record<string, string>
  complete: boolean
  captured_traces: number
  planned_traces: number
  recorded_cost_usd: number
  models: string[]
  seeds: SeedStatus[]
  traces: TraceIndexRow[]
  conditions: {
    decision: string
    actor: string
    colors: string[]
    context_suite: string
    board_surface: string
    reasoning_request: Record<string, unknown>
    temperature: number
    max_tokens_omitted: boolean
    replay_input: boolean
    human_action_labels: boolean
    scheduling: string
    prompt_variant: {
      id: string
      version: string
      guidance_sha256: string
    } | null
  }
}

type PromptMessage = {
  role: string
  content: unknown
}

export type TraceDetail = {
  schema: string
  trace_id: string
  run_id: string
  input: {
    seed: number
    prompt_sha256: string
    legal_actions_sha256: string
    board_presentation: {
      board_sha256: string
      content_sha256: string
      format: string
    }
    messages: PromptMessage[]
    legal_actions: Array<{
      index: number
      action: string
      description: string
    }>
  }
  request: {
    requested_model: string
    temperature: number
    reasoning: Record<string, unknown>
    max_tokens_omitted: boolean
    provider_payload: {
      model: string
      messages: PromptMessage[]
      temperature: number
      reasoning: Record<string, unknown>
      [key: string]: unknown
    }
  }
  response: {
    served_model: string | null
    provider_response_id: string | null
    provider_request_id: string | null
    finish_reason: string | null
    provider_native_finish_reason: string | null
    latency_ms: number | null
    usage: Record<string, unknown>
    reasoning_tokens: number | null
    native_reasoning_returned: boolean
    native_reasoning: string
    native_reasoning_details: unknown[]
    final_response: string
  }
  parse: {
    error: string | null
    choice: {
      action_index: number
      action: string
      action_description: string
      game_plan: string
      parse_warning: string | null
    } | null
  }
  render_state: GameState
  render_state_provenance: {
    seed: number
    board_sha256: string
    verified_against_trace: boolean
    renderer: string
  }
}
