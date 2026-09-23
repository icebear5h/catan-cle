import { useEffect, useMemo, useState } from 'react'
import type { TraceDetail, TraceOverview } from './types'
import TraceInspector from './TraceInspector'
import {
  apiError, displayModel, formatCost, formatDuration, formatReasoning, formatTokens,
} from './traceFormat'
import './ReasoningTraces.css'

const SERVER_URL = 'http://127.0.0.1:5001'

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

export default ReasoningTraces
