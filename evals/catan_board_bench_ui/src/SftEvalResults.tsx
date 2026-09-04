import { useEffect, useMemo, useState } from 'react'
import './SftDataExplorer.css'
import './SftEvalResults.css'

const SERVER_URL = 'http://127.0.0.1:5001'
const PAGE_SIZE = 80

type Metric = { correct: number; exact_accuracy: number; total: number }

type EvalRow = {
  index: number
  id: string
  state_id: string
  image_url: string
  prompt: string
  expected: string
  response: string
  correct: boolean
  scoring: string
  category: string
  density_bin: string
  entity_type: string
  row_kind: string
  suite: string
  relationship: string | null
  polarity: string | null
  task_type: string | null
}

type EvalSummary = {
  attempted: number
  correct: number
  exact_accuracy: number
  model_id: string
  bits: number
  batch_size: number
  generated_at: string
  by_category: Record<string, Metric>
  by_density_bin: Record<string, Metric>
  by_polarity: Record<string, Metric>
  by_relationship: Record<string, Metric>
  by_row_kind: Record<string, Metric>
  by_suite: Record<string, Metric>
  spatial_token_return: Metric
}

type EvalPayload = {
  rows: EvalRow[]
  total: number
  offset: number
  limit: number
  summary: EvalSummary
  checkpoint: { name: string; hub_url: string; modal_url: string }
  facets: {
    categories: string[]
    density_bins: string[]
    row_kinds: string[]
    suites: string[]
  }
}

type Filters = {
  category: string
  densityBin: string
  rowKind: string
  suite: string
  correctness: string
  query: string
}

const DEFAULT_FILTERS: Filters = {
  category: 'all',
  densityBin: 'all',
  rowKind: 'all',
  suite: 'all',
  correctness: 'incorrect',
  query: '',
}

function SftEvalResults() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS)
  const [offset, setOffset] = useState(0)
  const [payload, setPayload] = useState<EvalPayload | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    const timeout = window.setTimeout(async () => {
      try {
        setIsLoading(true)
        setError(null)
        const params = new URLSearchParams({
          category: filters.category,
          density_bin: filters.densityBin,
          row_kind: filters.rowKind,
          suite: filters.suite,
          correctness: filters.correctness,
          query: filters.query,
          offset: String(offset),
          limit: String(PAGE_SIZE),
        })
        const response = await fetch(`${SERVER_URL}/api/catan-board-bench/sft-eval?${params}`, {
          signal: controller.signal,
        })
        const data: unknown = await response.json()
        if (!response.ok) throw new Error(apiError(data, 'Failed to load SFT evaluation'))
        const next = data as EvalPayload
        setPayload(next)
        setSelectedId((current) => next.rows.some((row) => row.id === current)
          ? current
          : next.rows[0]?.id || '')
      } catch (loadError) {
        if (!controller.signal.aborted) setError(String(loadError))
      } finally {
        if (!controller.signal.aborted) setIsLoading(false)
      }
    }, filters.query ? 180 : 0)
    return () => {
      window.clearTimeout(timeout)
      controller.abort()
    }
  }, [filters, offset])

  const selected = useMemo(
    () => payload?.rows.find((row) => row.id === selectedId) || payload?.rows[0] || null,
    [payload, selectedId],
  )
  const selectedIndex = payload?.rows.findIndex((row) => row.id === selected?.id) ?? -1

  const setFilter = (field: keyof Filters, value: string) => {
    setFilters((current) => ({ ...current, [field]: value }))
    setOffset(0)
  }

  return (
    <main className="sft-shell eval-shell">
      {error && <div className="sft-error">{error}</div>}

      {payload && (
        <>
          <section className="eval-hero">
            <div>
              <div className="sft-overline">
                <span>final-validation-v1 · 1,080 held-out prompts</span>
                <span className="sft-live-dot">evaluation complete</span>
              </div>
              <h1>Grounding learned <em>unevenly.</em></h1>
              <p>
                The checkpoint can return port and tile tokens, but node, edge, and
                directional localization remain the clear bottleneck.
              </p>
              <div className="eval-links">
                <a href={payload.checkpoint.hub_url} target="_blank" rel="noreferrer">Hugging Face ↗</a>
                <a href={payload.checkpoint.modal_url} target="_blank" rel="noreferrer">Modal run ↗</a>
              </div>
            </div>
            <div className="eval-score">
              <span>exact match</span>
              <strong>{percent(payload.summary.exact_accuracy)}</strong>
              <p>{formatInteger(payload.summary.correct)} / {formatInteger(payload.summary.attempted)} correct</p>
            </div>
          </section>

          <section className="sft-stat-grid eval-stat-grid" aria-label="Checkpoint evaluation summary">
            <Stat label="Checkpoint" value="rank 8" note={payload.checkpoint.name} />
            <Stat label="Evaluation" value="BF16" note={`${payload.summary.batch_size} examples / batch`} />
            <Stat label="Strongest head" value={percent(payload.summary.by_category['inverse.port']?.exact_accuracy || 0)} note="inverse port localization" />
            <Stat label="Critical miss" value={percent(payload.summary.spatial_token_return.exact_accuracy)} note="spatial token return" danger />
          </section>

          <section className="eval-summary-grid">
            <PerformanceTable title="Performance by task" metrics={payload.summary.by_category} />
            <div className="eval-side-stack">
              <PerformanceBars title="Board density" metrics={payload.summary.by_density_bin} />
              <PerformanceBars title="Spatial relationship" metrics={payload.summary.by_relationship} omit={['unknown', 'presence', 'localization']} />
              <div className="eval-diagnosis">
                <span>readout</span>
                <strong>Token vocabulary works. Coordinate coverage does not.</strong>
                <p>Ports reach 98.4% and tiles 75.0%; nodes reach 18.8% and edges 14.1%.</p>
              </div>
            </div>
          </section>

          <section className="sft-controls eval-controls" aria-label="Evaluation result filters">
            <Select label="Result" value={filters.correctness} options={['all', 'incorrect', 'correct']} onChange={(value) => setFilter('correctness', value)} />
            <Select label="Task" value={filters.category} options={['all', ...payload.facets.categories]} onChange={(value) => setFilter('category', value)} />
            <Select label="Density" value={filters.densityBin} options={['all', ...payload.facets.density_bins]} onChange={(value) => setFilter('densityBin', value)} />
            <Select label="Direction" value={filters.rowKind} options={['all', ...payload.facets.row_kinds]} onChange={(value) => setFilter('rowKind', value)} />
            <Select label="Suite" value={filters.suite} options={['all', ...payload.facets.suites]} onChange={(value) => setFilter('suite', value)} />
            <label className="sft-search">
              <span>Search</span>
              <input value={filters.query} onChange={(event) => setFilter('query', event.target.value)} placeholder="token, task, response…" />
            </label>
            <button className="sft-reset" onClick={() => { setFilters(DEFAULT_FILTERS); setOffset(0) }}>Reset</button>
          </section>

          <section className="eval-workbench">
            <aside className="eval-row-rail">
              <header>
                <div><span>Matching results</span><strong>{formatInteger(payload.total)}</strong></div>
                <small>{payload.total ? offset + 1 : 0}–{Math.min(offset + payload.rows.length, payload.total)}</small>
              </header>
              <div className="eval-row-list">
                {payload.rows.map((row) => (
                  <button className={row.id === selected?.id ? 'active' : ''} key={row.id} onClick={() => setSelectedId(row.id)}>
                    <span className={`eval-result ${row.correct ? 'correct' : 'incorrect'}`}>{row.correct ? 'pass' : 'miss'}</span>
                    <strong>{row.category}</strong>
                    <small>{cleanPrompt(row.prompt)}</small>
                    <code>{row.expected} → {row.response}</code>
                  </button>
                ))}
                {!payload.rows.length && <div className="sft-empty">No evaluation rows match these filters.</div>}
              </div>
              <footer>
                <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>Previous</button>
                <button disabled={offset + PAGE_SIZE >= payload.total} onClick={() => setOffset(offset + PAGE_SIZE)}>Next</button>
              </footer>
            </aside>

            <div className="eval-detail">
              {selected ? (
                <>
                  <header>
                    <div>
                      <span className={`eval-result ${selected.correct ? 'correct' : 'incorrect'}`}>{selected.correct ? 'pass' : 'miss'}</span>
                      <span>{selected.category}</span><span>{selected.density_bin}</span><span>{selected.row_kind}</span>
                    </div>
                    <div className="sft-stepper">
                      <button disabled={selectedIndex <= 0} onClick={() => setSelectedId(payload.rows[selectedIndex - 1].id)}>←</button>
                      <code>{selected.index + 1} / {payload.summary.attempted}</code>
                      <button disabled={selectedIndex >= payload.rows.length - 1} onClick={() => setSelectedId(payload.rows[selectedIndex + 1].id)}>→</button>
                    </div>
                  </header>
                  <div className="eval-example-grid">
                    <figure className="sft-board-card">
                      <img src={SERVER_URL + selected.image_url} alt={`Held-out Catan state ${selected.state_id}`} />
                      <figcaption><span>{selected.state_id}</span><small>{selected.suite}</small></figcaption>
                    </figure>
                    <article className="eval-conversation">
                      <header><span>Held-out generation</span><code>{selected.id}</code></header>
                      <div className="sft-message user"><span>user</span><p><i>&lt;image&gt;</i>{cleanPrompt(selected.prompt)}</p></div>
                      <div className="eval-answer expected"><span>expected</span><strong>{selected.expected}</strong></div>
                      <div className={`eval-answer response ${selected.correct ? 'correct' : 'incorrect'}`}><span>model</span><strong>{selected.response || '∅'}</strong></div>
                      <dl className="sft-row-contract">
                        <div><dt>relationship</dt><dd>{selected.relationship || selected.task_type || 'n/a'}</dd></div>
                        <div><dt>polarity</dt><dd>{selected.polarity || 'n/a'}</dd></div>
                        <div><dt>scoring</dt><dd>{selected.scoring}</dd></div>
                        <div><dt>entity</dt><dd>{selected.entity_type}</dd></div>
                      </dl>
                    </article>
                  </div>
                </>
              ) : <div className="sft-detail-empty">Choose a result to inspect it.</div>}
            </div>
          </section>
        </>
      )}

      {isLoading && <div className="sft-loading">loading held-out generations</div>}
    </main>
  )
}

function PerformanceTable({ title, metrics }: { title: string; metrics: Record<string, Metric> }) {
  const entries = Object.entries(metrics).sort((left, right) => right[1].exact_accuracy - left[1].exact_accuracy)
  return (
    <article className="eval-performance-table">
      <header><h2>{title}</h2><span>exact generation match</span></header>
      <div className="eval-table-head"><span>task</span><span>correct</span><span>accuracy</span></div>
      {entries.map(([name, metric]) => (
        <div className="eval-table-row" key={name}>
          <span>{humanize(name)}</span>
          <span>{metric.correct} / {metric.total}</span>
          <strong className={metric.exact_accuracy < 0.25 ? 'danger' : ''}>{percent(metric.exact_accuracy)}</strong>
        </div>
      ))}
    </article>
  )
}

function PerformanceBars({ title, metrics, omit = [] }: { title: string; metrics: Record<string, Metric>; omit?: string[] }) {
  const entries = Object.entries(metrics).filter(([name]) => !omit.includes(name))
  return (
    <article className="eval-performance-bars">
      <header><h2>{title}</h2><span>exact match</span></header>
      {entries.map(([name, metric]) => (
        <div className="eval-bar" key={name}>
          <span>{humanize(name)}</span>
          <div><i style={{ width: `${metric.exact_accuracy * 100}%` }} /></div>
          <strong>{percent(metric.exact_accuracy)}</strong>
        </div>
      ))}
    </article>
  )
}

function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return <label className="sft-select-wrap"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option key={option} value={option}>{humanize(option)}</option>)}</select></label>
}

function Stat({ label, value, note, danger = false }: { label: string; value: string; note: string; danger?: boolean }) {
  return <div className="sft-stat"><span>{label}</span><strong className={danger ? 'eval-danger' : ''}>{value}</strong><small>{note}</small></div>
}

function percent(value: number) { return `${(value * 100).toFixed(1)}%` }
function formatInteger(value: number) { return new Intl.NumberFormat('en-US').format(value) }
function cleanPrompt(value: string) { return value.replace(/^<image>\s*/i, '') }
function humanize(value: string) { return value.replaceAll('_', ' ').replaceAll('.', ' · ') }
function apiError(payload: unknown, fallback: string) {
  return payload && typeof payload === 'object' && 'error' in payload ? String(payload.error) : fallback
}

export default SftEvalResults
