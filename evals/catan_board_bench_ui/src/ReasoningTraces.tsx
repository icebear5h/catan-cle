import { useEffect, useMemo, useState } from 'react'
import type { GameState } from '@playground/types'
import HexBoard from '@playground/components/HexBoard'
import './ReasoningTraces.css'

const SERVER_URL = 'http://127.0.0.1:5001'

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

type TraceOverview = {
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

type TraceDetail = {
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

function ReasoningTraces() {
  const [overview, setOverview] = useState<TraceOverview | null>(null)
  const [selectedRunId, setSelectedRunId] = useState('')
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null)
  const [selectedModel, setSelectedModel] = useState('')
  const [detail, setDetail] = useState<TraceDetail | null>(null)
  const [isLoadingOverview, setIsLoadingOverview] = useState(true)
  const [isLoadingDetail, setIsLoadingDetail] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    const loadOverview = async () => {
      try {
        setIsLoadingOverview(true)
        setError(null)
        const params = new URLSearchParams()
        if (selectedRunId) params.set('run_id', selectedRunId)
        const suffix = params.size ? '?' + params : ''
        const response = await fetch(
          SERVER_URL + '/api/catan-board-bench/reasoning-traces' + suffix,
          { signal: controller.signal },
        )
        const payload: unknown = await response.json()
        if (!response.ok) {
          throw new Error(apiError(payload, 'Failed to load reasoning traces'))
        }
        const next = payload as TraceOverview
        setOverview(next)
        setSelectedRunId(next.run_id)
        const firstStarted = next.seeds.find((seed) => seed.captured > 0)
        const seed = firstStarted?.seed ?? next.seeds[0]?.seed ?? null
        setSelectedSeed(seed)
        const capturedModels = new Set(
          next.traces
            .filter((trace) => trace.seed === seed)
            .map((trace) => trace.model_id),
        )
        setSelectedModel(next.models.find((modelId) => capturedModels.has(modelId)) || '')
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(String(loadError))
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoadingOverview(false)
        }
      }
    }
    void loadOverview()
    return () => controller.abort()
  }, [selectedRunId])

  useEffect(() => {
    if (!overview || selectedSeed === null) {
      return
    }
    const capturedModels = new Set(
      overview.traces
        .filter((trace) => trace.seed === selectedSeed)
        .map((trace) => trace.model_id),
    )
    if (!capturedModels.has(selectedModel)) {
      setSelectedModel(
        overview.models.find((modelId) => capturedModels.has(modelId)) || '',
      )
    }
  }, [overview, selectedSeed, selectedModel])

  useEffect(() => {
    const traceAvailable = overview?.traces.some(
      (trace) => trace.seed === selectedSeed && trace.model_id === selectedModel,
    )
    if (!overview || selectedSeed === null || !selectedModel || !traceAvailable) {
      setDetail(null)
      return
    }
    const controller = new AbortController()
    const loadDetail = async () => {
      try {
        setIsLoadingDetail(true)
        setError(null)
        setDetail(null)
        const params = new URLSearchParams({
          run_id: overview.run_id,
          seed: String(selectedSeed),
          model_id: selectedModel,
        })
        const response = await fetch(
          SERVER_URL + '/api/catan-board-bench/reasoning-trace?' + params,
          { signal: controller.signal },
        )
        const payload: unknown = await response.json()
        if (!response.ok) {
          throw new Error(apiError(payload, 'Failed to load reasoning trace'))
        }
        setDetail(payload as TraceDetail)
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(String(loadError))
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoadingDetail(false)
        }
      }
    }
    void loadDetail()
    return () => controller.abort()
  }, [overview, selectedSeed, selectedModel])

  const selectedSeedStatus = useMemo(
    () => overview?.seeds.find((seed) => seed.seed === selectedSeed) || null,
    [overview, selectedSeed],
  )

  const traceByModel = useMemo(() => {
    return new Map(
      (overview?.traces || [])
        .filter((trace) => trace.seed === selectedSeed)
        .map((trace) => [trace.model_id, trace]),
    )
  }, [overview, selectedSeed])

  if (isLoadingOverview && !overview) {
    return <main className="reasoning-shell"><div className="reasoning-notice">loading native traces</div></main>
  }

  return (
    <main className="reasoning-shell">
      {error && <div className="reasoning-notice error">{error}</div>}

      {overview && (
        <>
          <section className="reasoning-run-selector">
            <label htmlFor="reasoning-run">Trace condition</label>
            <select
              id="reasoning-run"
              value={overview.run_id}
              onChange={(event) => setSelectedRunId(event.target.value)}
            >
              {overview.available_runs.filter((run) => run.available).map((run) => (
                <option key={run.id} value={run.id}>{run.title}</option>
              ))}
            </select>
            <div>
              <strong>{overview.run_title}</strong>
              <span>{overview.run_description}</span>
            </div>
          </section>

          <section className="reasoning-hero">
            <div>
              <span className="reasoning-kicker">{overview.run_title} · fresh game · native channel only</span>
              <h2>Inspect the model before we call it <em>RL-fried.</em></h2>
              <p>
                Every model receives the same empty-history first-settlement prompt.
                Provider-native reasoning and the final answer stay separate; there is
                no replay, human label, or authored rationale in this view.
              </p>
            </div>
            <div className="reasoning-run-card">
              <span>Seed {selectedSeedStatus?.seed ?? '—'}</span>
              <strong>{selectedSeedStatus?.captured ?? 0}<small> / {selectedSeedStatus?.expected ?? overview.models.length}</small></strong>
              <p>{selectedSeedStatus?.status.replace('_', ' ') || 'unknown'} capture</p>
              <code>{formatCost(overview.recorded_cost_usd)} direct spend</code>
            </div>
          </section>

          <section className="reasoning-conditions" aria-label="Trace conditions">
            <Condition label="Decision" value={overview.conditions.decision} />
            <Condition label="Actor" value={overview.conditions.actor} />
            <Condition label="Suite" value={overview.conditions.context_suite} />
            <Condition label="Board" value={overview.conditions.board_surface} />
            <Condition label="Reasoning" value={formatReasoning(overview.conditions.reasoning_request)} />
            <Condition label="max_tokens" value={overview.conditions.max_tokens_omitted ? 'omitted' : 'present'} highlight />
          </section>

          {Object.keys(overview.excluded_models).length > 0 && (
            <section className="reasoning-dq" aria-label="Disqualified models">
              <div>
                <span>DQ from this rerun</span>
                <strong>{Object.keys(overview.excluded_models).length} models</strong>
              </div>
              {Object.entries(overview.excluded_models).map(([modelId, reason]) => (
                <article key={modelId}>
                  <strong>{displayModel(modelId)}</strong>
                  <small>{reason}</small>
                </article>
              ))}
            </section>
          )}

          <section className="reasoning-workspace">
            <aside className="reasoning-rail">
              <div className="reasoning-rail-heading">
                <span>Game seeds</span>
                <small>{overview.seeds.length} planned</small>
              </div>
              <div className="reasoning-seed-switch" aria-label="Fresh game seed">
                {overview.seeds.map((seed) => (
                  <button
                    key={seed.seed}
                    className={seed.seed === selectedSeed ? 'active' : ''}
                    onClick={() => setSelectedSeed(seed.seed)}
                  >
                    <strong>{seed.seed}</strong>
                    <small>{seed.captured}/{seed.expected} · {seed.status.replace('_', ' ')}</small>
                  </button>
                ))}
              </div>

              <div className="reasoning-rail-heading models">
                <span>Models</span>
                <small>{selectedSeedStatus?.captured || 0} captured</small>
              </div>
              <div className="reasoning-model-list">
                {overview.models.map((modelId) => {
                  const trace = traceByModel.get(modelId)
                  const missing = !trace
                  return (
                    <button
                      key={modelId}
                      className={`${modelId === selectedModel ? 'active' : ''} ${missing ? 'missing' : ''}`}
                      onClick={() => !missing && setSelectedModel(modelId)}
                      disabled={missing}
                    >
                      <span>
                        <strong>{displayModel(modelId)}</strong>
                        <small>{modelId}</small>
                      </span>
                      {trace ? (
                        <span className={`reasoning-finish ${trace.finish_reason === 'stop' ? 'stop' : 'length'}`}>
                          {trace.finish_reason || 'unknown'}
                        </span>
                      ) : (
                        <span className="reasoning-finish missing">not captured</span>
                      )}
                      {trace && (
                        <code>{formatTokens(trace.reasoning_tokens)} reasoning · {formatDuration(trace.latency_ms)}</code>
                      )}
                    </button>
                  )
                })}
              </div>
            </aside>

            <section className="reasoning-detail">
              {isLoadingDetail && <div className="reasoning-notice">loading exact trace</div>}
              {!isLoadingDetail && !detail && (
                <div className="reasoning-empty">
                  {selectedSeedStatus?.status === 'not_started'
                    ? 'This seed is planned but has not been run.'
                    : 'Select a captured model trace.'}
                </div>
              )}
              {detail && <TraceInspector detail={detail} />}
            </section>
          </section>
        </>
      )}
    </main>
  )
}

function TraceInspector({ detail }: { detail: TraceDetail }) {
  const usage = detail.response.usage
  const promptTokens = numericUsage(usage, 'prompt_tokens')
  const completionTokens = numericUsage(usage, 'completion_tokens')
  const cost = numericUsage(usage, 'cost')
  const providerMessages = detail.request.provider_payload.messages || detail.input.messages

  return (
    <>
      <header className="reasoning-detail-header">
        <div>
          <span>Seed {detail.input.seed}</span>
          <h3>{displayModel(detail.request.requested_model)}</h3>
          <code>{detail.request.requested_model}</code>
        </div>
        <div className="reasoning-detail-status">
          <span className={`reasoning-finish ${detail.response.finish_reason === 'stop' ? 'stop' : 'length'}`}>
            {detail.response.finish_reason || 'unknown finish'}
          </span>
          <small>{detail.response.served_model || 'served model unavailable'}</small>
        </div>
      </header>

      <div className="reasoning-context-grid">
        <section className="reasoning-board-panel">
          <header>
            <span>Exact seed board</span>
            <small>SHA verified</small>
          </header>
          <div className="reasoning-board-wrap">
            <HexBoard gameState={detail.render_state} showControls={false} />
          </div>
          <footer>
            <span>seed {detail.render_state_provenance.seed}</span>
            <code>{detail.render_state_provenance.board_sha256.slice(0, 16)}</code>
          </footer>
        </section>

        <section className="reasoning-response-overview">
          <div className="reasoning-metrics">
            <Metric label="Reasoning" value={formatTokens(detail.response.reasoning_tokens)} />
            <Metric label="Completion" value={formatTokens(completionTokens)} />
            <Metric label="Prompt" value={formatTokens(promptTokens)} />
            <Metric label="Latency" value={formatDuration(detail.response.latency_ms)} />
            <Metric label="Cost" value={formatCost(cost)} />
            <Metric
              label="Action"
              value={detail.parse.choice ? String(detail.parse.choice.action_index) : 'none'}
              warning={!detail.parse.choice}
            />
          </div>

          {(detail.parse.error || detail.response.finish_reason !== 'stop') && (
            <div className="reasoning-warning">
              <strong>{detail.response.finish_reason === 'length' ? 'Provider output ceiling reached' : 'Response warning'}</strong>
              <span>{detail.parse.error || 'The response did not stop naturally.'}</span>
            </div>
          )}

          {detail.parse.choice && (
            <section className="reasoning-selection">
              <span>Parsed convenience view</span>
              <strong>Action {detail.parse.choice.action_index}</strong>
              <p>{detail.parse.choice.action_description}</p>
            </section>
          )}

          <div className="reasoning-board-proof">
            <span>Board source</span>
            <strong>{detail.render_state_provenance.renderer}</strong>
            <small>The server reconstructed seed {detail.input.seed} without touching shared game state, then matched the immutable public-board digest.</small>
          </div>
        </section>
      </div>

      <div className="reasoning-copy-grid">
        <TraceText
          label="Native provider reasoning"
          note={`${formatTokens(detail.response.reasoning_tokens)} tokens · separate response channel`}
          text={detail.response.native_reasoning}
          className="native"
          empty="No native reasoning was returned."
        />
        <TraceText
          label="Final model response"
          note={detail.parse.choice ? `parsed action ${detail.parse.choice.action_index}` : 'no parsed action'}
          text={detail.response.final_response}
          className="final"
          empty="No final response was returned."
        />
      </div>

      <details className="reasoning-prompt">
        <summary>
          <span>
            <small>Exact provider prompt</small>
            <strong>{detail.input.prompt_sha256.slice(0, 16)}</strong>
          </span>
          <span>{providerMessages.length} messages · `max_tokens` omitted</span>
        </summary>
        <div>
          {providerMessages.map((message, index) => (
            <section key={`${message.role}-${index}`} className="reasoning-message">
              <span>{message.role}</span>
              <pre>{renderContent(message.content)}</pre>
            </section>
          ))}
        </div>
      </details>

      <details className="reasoning-provider-meta">
        <summary>Provider identity and request metadata</summary>
        <pre>{JSON.stringify({
          trace_id: detail.trace_id,
          provider_response_id: detail.response.provider_response_id,
          provider_request_id: detail.response.provider_request_id,
          provider_native_finish_reason: detail.response.provider_native_finish_reason,
          reasoning_request: detail.request.reasoning,
          temperature: detail.request.temperature,
          max_tokens_omitted: detail.request.max_tokens_omitted,
          board_sha256: detail.input.board_presentation.board_sha256,
          board_content_sha256: detail.input.board_presentation.content_sha256,
          legal_actions_sha256: detail.input.legal_actions_sha256,
          usage,
        }, null, 2)}</pre>
      </details>
    </>
  )
}

function TraceText({
  label,
  note,
  text,
  className,
  empty,
}: {
  label: string
  note: string
  text: string
  className: string
  empty: string
}) {
  return (
    <section className={`reasoning-text-panel ${className}`}>
      <header>
        <span>{label}</span>
        <small>{note}</small>
      </header>
      <pre className={!text ? 'empty' : ''}>{text || empty}</pre>
    </section>
  )
}

function Condition({
  label,
  value,
  highlight = false,
}: {
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div className={highlight ? 'highlight' : ''}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function Metric({ label, value, warning = false }: { label: string; value: string; warning?: boolean }) {
  return (
    <div className={warning ? 'warning' : ''}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function displayModel(modelId: string) {
  const name = modelId.split('/').at(-1) || modelId
  return name.replace(':free', '').replace(/-/g, ' ')
}

function formatReasoning(reasoning: Record<string, unknown>) {
  if (typeof reasoning.effort === 'string') return reasoning.effort
  if (reasoning.enabled === true) return 'enabled'
  return JSON.stringify(reasoning)
}

function numericUsage(usage: Record<string, unknown>, key: string) {
  const value = usage[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function formatTokens(value: number | null) {
  if (value === null) return '—'
  return new Intl.NumberFormat('en-US').format(value)
}

function formatDuration(value: number | null) {
  if (value === null) return '—'
  if (value < 1000) return `${Math.round(value)} ms`
  const seconds = value / 1000
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

function formatCost(value: number | null) {
  if (value === null) return '—'
  return `$${value.toFixed(value < 0.01 ? 6 : 4)}`
}

function renderContent(content: unknown) {
  return typeof content === 'string' ? content : JSON.stringify(content, null, 2)
}

function apiError(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object' && 'error' in payload) {
    return String((payload as { error: unknown }).error)
  }
  return fallback
}

export default ReasoningTraces
