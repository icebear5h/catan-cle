import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import './SftDataExplorer.css'
import PatchInspector from './PatchInspector'

const SERVER_URL = 'http://127.0.0.1:5001'
const PAGE_SIZE = 80

const STAGE_SPLITS: Record<string, string[]> = {
  stage1: ['train', 'validation', 'test'],
  stage2: ['train', 'validation', 'test'],
  probes: ['validation', 'test'],
}

type SpatialTarget = {
  token: string
  entity_type: string
  bbox: [number, number, number, number]
  center: [number, number]
  control_bbox: [number, number, number, number]
  control_token: string
}

type SpatialRow = {
  index: number
  record_id: string
  row_id: string
  state_id: string
  image_name: string
  image_url: string
  prompt: string
  answer: string
  curriculum_stage: string
  grounding_stage: string
  task_type: string
  entity_type: string
  target_token: string | null
  queried_token: string | null
  piece: string | null
  color: string | null
  density_bin: string | null
  tokens: string[]
  marker: string | null
  marker_style: string | null
  marker_group: string[]
  relationship: string
  polarity: string
  sampling_repeat: number | null
  replay_source: string | null
  probe_style: string | null
  eval_variant: string | null
  spatial_target: SpatialTarget | null
}

type SpatialSummary = {
  row_count: number
  state_count: number
  image_count: number
  source_schema: string
  source_sha256: string
  atlas_counts: Record<string, number>
  atlas_tokens: number
  marker_groups_per_board: number
  node_edge_sampling_multiplier: number
  stage2_marker_replay_fraction: number
}

type StageCatalogRow = {
  id: string
  splits: Record<string, number>
  train_rows: number
}

type SpatialPayload = {
  dataset: string
  datasets: { id: string; label: string }[]
  stage: string
  split: string
  rows: SpatialRow[]
  total: number
  offset: number
  limit: number
  summary: SpatialSummary
  stages: StageCatalogRow[]
  distributions: {
    task_type: Record<string, number>
    entity: Record<string, number>
    relationship: Record<string, number>
    polarity: Record<string, number>
  }
  facets: {
    stages: string[]
    splits: string[]
    task_types: string[]
    entity_types: string[]
    relationships: string[]
    polarities: string[]
  }
}

type Filters = {
  dataset: string
  stage: string
  split: string
  taskType: string
  entityType: string
  relationship: string
  polarity: string
  query: string
}

const DEFAULT_FILTERS: Filters = {
  dataset: 'node_edge_readout_reweighted_v1',
  stage: 'stage1',
  split: 'train',
  taskType: 'all',
  entityType: 'all',
  relationship: 'all',
  polarity: 'all',
  query: '',
}

const STAGE_COPY: Record<string, { number: string; title: string; note: string }> = {
  stage1: {
    number: '01',
    title: 'Marked localization',
    note: 'Markers C, A, T, and N teach both marker → atlas token and atlas token → marker on empty boards.',
  },
  stage2: {
    number: '02',
    title: 'Unmarked orientation',
    note: 'Above, below, left, right, adjacent, and connected relations with hard negatives and 25% marker replay.',
  },
  probes: {
    number: '03',
    title: 'Neutral held-out probes',
    note: 'Unseen small gray dots test whether localization transfers beyond the training marker style.',
  },
}

function SftDataExplorer() {
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)
  const [offset, setOffset] = useState(0)
  const [payload, setPayload] = useState<SpatialPayload | null>(null)
  const [selectedRecordId, setSelectedRecordId] = useState('')
  const [showPatchTargets, setShowPatchTargets] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    const timeout = window.setTimeout(async () => {
      try {
        setIsLoading(true)
        setError(null)
        const params = new URLSearchParams({
          dataset: filters.dataset,
          stage: filters.stage,
          split: filters.split,
          task_type: filters.taskType,
          entity_type: filters.entityType,
          relationship: filters.relationship,
          polarity: filters.polarity,
          query: filters.query,
          offset: String(offset),
          limit: String(PAGE_SIZE),
        })
        const response = await fetch(
          SERVER_URL + '/api/catan-board-bench/spatial-localization-data?' + params,
          { signal: controller.signal },
        )
        const data: unknown = await response.json()
        if (!response.ok) throw new Error(apiError(data, 'Failed to load spatial corpus'))
        const next = data as SpatialPayload
        setPayload(next)
        setSelectedRecordId((current) => (
          next.rows.some((row) => row.record_id === current)
            ? current
            : next.rows[0]?.record_id || ''
        ))
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

  const selectedRow = useMemo(
    () => payload?.rows.find((row) => row.record_id === selectedRecordId) || payload?.rows[0] || null,
    [payload, selectedRecordId],
  )
  const selectedPageIndex = useMemo(
    () => payload?.rows.findIndex((row) => row.record_id === selectedRow?.record_id) ?? -1,
    [payload, selectedRow],
  )

  const setFilter = (field: keyof Filters, value: string) => {
    setFilters((current) => ({ ...current, [field]: value }))
    setOffset(0)
  }

  const setStage = (stage: string) => {
    setFilters((current) => ({
      ...current,
      stage,
      split: STAGE_SPLITS[stage]?.[0] || 'train',
      taskType: 'all',
      entityType: 'all',
      relationship: 'all',
      polarity: 'all',
      query: '',
    }))
    setOffset(0)
  }

  const resetFilters = () => {
    setFilters(DEFAULT_FILTERS)
    setOffset(0)
  }

  const stepRow = (delta: number) => {
    if (!payload || selectedPageIndex < 0) return
    const next = payload.rows[selectedPageIndex + delta]
    if (next) setSelectedRecordId(next.record_id)
  }

  return (
    <main className="sft-shell spatial-shell">
      <section className="sft-hero">
        <div>
          <div className="sft-overline">
            <span>{filters.dataset}</span>
            <span className="sft-live-dot">ready to inspect</span>
          </div>
          <h1>Inspect the board <em>patch by patch.</em></h1>
          <p>
            Browse the actual images, prompts, and labels from each training corpus.
            Compare 16-pixel input patches with 32-pixel merger groups on boards with pieces.
          </p>
        </div>
        {payload && (
          <div className="sft-ratio-card spatial-count-card">
            <span>selected corpus slice</span>
            <strong>{formatCompact(payload.summary.row_count)}</strong>
            <p>{payload.stage} · {payload.split} rows</p>
          </div>
        )}
      </section>

      {error && <div className="sft-error">{error}</div>}

      {payload && (
        <>
          <label className="sft-select-wrap sft-dataset-select">
            <span>Dataset</span>
            <select value={filters.dataset} onChange={(event) => {
              setFilters({ ...DEFAULT_FILTERS, dataset: event.target.value })
              setOffset(0)
            }}>
              {payload.datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.label}</option>)}
            </select>
          </label>
          {filters.dataset === 'spatial_localization_v1' && <CurriculumPanel stages={payload.stages} selectedStage={filters.stage} onSelect={setStage} />}

          <section className="sft-stat-grid" aria-label="Spatial corpus summary">
            <Stat label="Rows in slice" value={formatInteger(payload.summary.row_count)} note={`${payload.stage} / ${payload.split}`} />
            <Stat label="Unique boards" value={formatInteger(payload.summary.state_count)} note={`${formatInteger(payload.summary.image_count)} rendered images`} />
            <Stat label="Atlas positions" value={formatInteger(payload.summary.atlas_tokens)} note={atlasNote(payload.summary.atlas_counts)} />
            <Stat label="Dataset lock" value={payload.summary.source_sha256.slice(0, 10)} note="JSONL SHA-256" mono />
          </section>

          <section className="sft-controls spatial-controls" aria-label="Spatial data filters">
            <FilterSelect label="Stage" value={filters.stage} options={payload.facets.stages} onChange={setStage} includeAll={false} />
            <FilterSelect label="Split" value={filters.split} options={payload.facets.splits} onChange={(value) => setFilter('split', value)} includeAll={false} />
            <FilterSelect label="Task" value={filters.taskType} options={payload.facets.task_types} onChange={(value) => setFilter('taskType', value)} />
            <FilterSelect label="Entity" value={filters.entityType} options={payload.facets.entity_types} onChange={(value) => setFilter('entityType', value)} />
            <FilterSelect label="Relation" value={filters.relationship} options={payload.facets.relationships} onChange={(value) => setFilter('relationship', value)} />
            <FilterSelect label="Polarity" value={filters.polarity} options={payload.facets.polarities} onChange={(value) => setFilter('polarity', value)} />
            <label className="sft-search">
              <span>Search</span>
              <input value={filters.query} onChange={(event) => setFilter('query', event.target.value)} placeholder="token, prompt, state…" />
            </label>
            <button className="sft-reset" onClick={resetFilters}>Reset</button>
          </section>

          <section className="sft-workbench">
            <aside className="sft-row-rail">
              <header>
                <div><span>Rows matching filters</span><strong>{formatInteger(payload.total)}</strong></div>
                <small>{payload.total ? offset + 1 : 0}–{Math.min(offset + payload.rows.length, payload.total)}</small>
              </header>
              <div className="sft-row-list">
                {payload.rows.map((row) => (
                  <button key={row.record_id} className={row.record_id === selectedRow?.record_id ? 'active' : ''} onClick={() => setSelectedRecordId(row.record_id)}>
                    <span className={`sft-kind ${rowKind(row)}`}>{rowKind(row)}</span>
                    <span className="sft-row-token">{row.queried_token || row.target_token || row.tokens.join(' ↔ ') || row.task_type}</span>
                    <small>{cleanPrompt(row.prompt)}</small>
                  </button>
                ))}
                {!payload.rows.length && <div className="sft-empty">No rows match these filters.</div>}
              </div>
              <footer>
                <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>Previous page</button>
                <button disabled={offset + PAGE_SIZE >= payload.total} onClick={() => setOffset(offset + PAGE_SIZE)}>Next page</button>
              </footer>
            </aside>

            <div className="sft-detail">
              {selectedRow ? (
                <>
                  <div className="sft-detail-topline">
                    <div>
                      <span className={`sft-kind ${rowKind(selectedRow)}`}>{rowKind(selectedRow)}</span>
                      <span>{selectedRow.entity_type}</span>
                      <span>{humanize(selectedRow.task_type)}</span>
                      {selectedRow.relationship !== 'unknown' && <span>{humanize(selectedRow.relationship)}</span>}
                      {selectedRow.polarity !== 'unknown' && <span>{humanize(selectedRow.polarity)}</span>}
                    </div>
                    <div className="sft-stepper">
                      <button disabled={selectedPageIndex <= 0} onClick={() => stepRow(-1)}>←</button>
                      <code>{selectedRow.index + 1} / {payload.summary.row_count}</code>
                      <button disabled={selectedPageIndex >= payload.rows.length - 1} onClick={() => stepRow(1)}>→</button>
                    </div>
                  </div>

                  <div className="sft-example-grid">
                    <figure className="sft-board-card spatial-board-card">
                      <PatchInspector key={selectedRow.image_url} src={SERVER_URL + selectedRow.image_url} alt={`Board ${selectedRow.state_id}`}>
                        {showPatchTargets && selectedRow.spatial_target && (
                          <span className="sft-patch-box target" style={bboxStyle(selectedRow.spatial_target.bbox)} title={`correct patch: ${selectedRow.spatial_target.token}`} />
                        )}
                      </PatchInspector>
                      <figcaption>
                        <span>{selectedRow.image_name}</span>
                        <small>{selectedRow.state_id} · {filters.split} image</small>
                        {selectedRow.spatial_target && (
                          <label className="sft-patch-toggle">
                            <input type="checkbox" checked={showPatchTargets} onChange={(event) => setShowPatchTargets(event.target.checked)} />
                            show correct patch <i className="target" />
                          </label>
                        )}
                      </figcaption>
                    </figure>

                    <article className="sft-conversation">
                      <header><span>Exact dataset row</span><code>{selectedRow.row_id}</code></header>
                      <div className="sft-message user"><span>user</span><p><i>&lt;image&gt;</i>{cleanPrompt(selectedRow.prompt)}</p></div>
                      <div className="sft-flow-arrow" aria-hidden="true">↓</div>
                      <div className="sft-message assistant"><span>assistant</span><p>{selectedRow.answer}</p></div>
                      <dl className="sft-row-contract">
                        <Contract label="supervision" value={supervisionLabel(selectedRow)} />
                        <Contract label="queried location" value={selectedRow.queried_token || selectedRow.tokens.join(', ') || 'see prompt'} mono />
                        <Contract label="target location" value={selectedRow.target_token || 'see answer'} mono />
                        <Contract label="piece / color" value={[selectedRow.piece, selectedRow.color].filter(Boolean).join(' / ') || 'none'} />
                        <Contract label="board density" value={selectedRow.density_bin || 'not specified'} />
                        <Contract label="marker" value={markerLabel(selectedRow)} />
                        <Contract label="patch target" value={patchLabel(selectedRow.spatial_target)} mono />
                        <Contract label="control patch" value={controlLabel(selectedRow.spatial_target)} mono />
                        <Contract label="replay" value={selectedRow.replay_source || 'none'} />
                      </dl>
                    </article>
                  </div>
                </>
              ) : <div className="sft-detail-empty">Choose a row to inspect its exact image and messages.</div>}
            </div>
          </section>

          <section className="sft-distributions spatial-distributions">
            <Distribution title="Task type" counts={payload.distributions.task_type} total={payload.summary.row_count} limit={14} />
            <Distribution title="Entity family" counts={payload.distributions.entity} total={payload.summary.row_count} limit={6} />
            <Distribution title="Relationship" counts={payload.distributions.relationship} total={payload.summary.row_count} limit={8} />
            <Distribution title="Polarity" counts={payload.distributions.polarity} total={payload.summary.row_count} limit={6} />
          </section>
        </>
      )}

      {isLoading && <div className="sft-loading">loading corpus window</div>}
    </main>
  )
}

function CurriculumPanel({ stages, selectedStage, onSelect }: { stages: StageCatalogRow[]; selectedStage: string; onSelect: (stage: string) => void }) {
  return (
    <section className="spatial-curriculum" aria-label="Spatial grounding curriculum">
      <header>
        <div><span className="sft-status-chip ready">materialized</span><h2>One curriculum, <em>three observable steps.</em></h2></div>
        <p>Click a stage to inspect the actual rows. Nothing here is a projected count.</p>
      </header>
      <div className="spatial-stage-grid">
        {stages.map((stage) => {
          const copy = STAGE_COPY[stage.id]
          const displayRows = stage.train_rows || stage.splits.validation || 0
          const splitLabel = stage.train_rows ? 'train rows' : 'validation rows'
          return (
            <button key={stage.id} className={stage.id === selectedStage ? 'active' : ''} onClick={() => onSelect(stage.id)}>
              <span>{copy.number}</span>
              <div><h3>{copy.title}</h3><strong>{formatInteger(displayRows)} <small>{splitLabel}</small></strong><p>{copy.note}</p></div>
            </button>
          )
        })}
      </div>
    </section>
  )
}

function Stat({ label, value, note, mono = false }: { label: string; value: string; note: string; mono?: boolean }) {
  return <div className="sft-stat"><span>{label}</span><strong className={mono ? 'mono' : ''}>{value}</strong><small>{note}</small></div>
}

function Contract({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return <div><dt>{label}</dt><dd>{mono ? <code>{value}</code> : value}</dd></div>
}

function FilterSelect({ label, value, options, onChange, includeAll = true }: { label: string; value: string; options: string[]; onChange: (value: string) => void; includeAll?: boolean }) {
  return (
    <label className="sft-select-wrap">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {includeAll && <option value="all">all</option>}
        {options.map((option) => <option key={option} value={option}>{humanize(option)}</option>)}
      </select>
    </label>
  )
}

function Distribution({ title, counts, total, limit }: { title: string; counts: Record<string, number>; total: number; limit: number }) {
  const entries = Object.entries(counts).slice(0, limit)
  const max = Math.max(...entries.map(([, count]) => count), 1)
  return (
    <div className="sft-distribution">
      <header><span>{title}</span><small>{entries.length} shown</small></header>
      {entries.map(([label, count]) => (
        <div className="sft-bar" key={label}>
          <span title={label}>{humanize(label)}</span>
          <div><i style={{ width: `${(count / max) * 100}%` }} /></div>
          <strong>{formatInteger(count)}</strong>
          <small>{((count / total) * 100).toFixed(1)}%</small>
        </div>
      ))}
    </div>
  )
}

function bboxStyle(bbox: [number, number, number, number]): CSSProperties {
  return {
    left: `${bbox[0] * 100}%`,
    top: `${bbox[1] * 100}%`,
    width: `${(bbox[2] - bbox[0]) * 100}%`,
    height: `${(bbox[3] - bbox[1]) * 100}%`,
  }
}

function rowKind(row: SpatialRow) {
  if (row.piece && !['TILE', 'PORT'].includes(row.piece.toUpperCase())) return 'piece'
  if (['tile', 'port'].includes(row.entity_type) && !row.marker) return row.entity_type
  if (row.probe_style) return 'probe'
  if (row.marker || row.replay_source) return 'marked'
  return 'relation'
}

function supervisionLabel(row: SpatialRow) {
  if (row.piece) return humanize(row.task_type)
  if (row.task_type === 'marker_to_token') return 'visual marker → atlas token'
  if (row.task_type === 'token_to_marker') return 'atlas token → visual marker'
  if (row.task_type === 'neutral_probe_token_return') return 'novel gray dot → atlas token'
  if (row.polarity === 'token_return') return 'spatial relation → atlas token'
  return `${humanize(row.relationship)} relation → ${row.polarity === 'hard_negative' ? 'no' : 'yes'}`
}

function markerLabel(row: SpatialRow) {
  if (row.probe_style) return humanize(row.probe_style)
  if (!row.marker) return 'none'
  return `${row.marker} · ${humanize(row.marker_style || 'unknown style')}`
}

function patchLabel(target: SpatialTarget | null) {
  return target ? `${target.token} · [${target.bbox.map(formatCoord).join(', ')}]` : 'not annotated in this row'
}

function controlLabel(target: SpatialTarget | null) {
  return target?.control_bbox ? `${target.control_token} · [${target.control_bbox.map(formatCoord).join(', ')}]` : 'not annotated in this row'
}

function atlasNote(counts: Record<string, number>) {
  return Object.entries(counts).map(([key, value]) => `${value} ${key}`).join(' · ')
}

function cleanPrompt(prompt: string) { return prompt.replace(/^<image>\s*/i, '') }
function humanize(value: string) { return value.replaceAll('_', ' ') }
function formatInteger(value: number) { return new Intl.NumberFormat('en-US').format(value) }
function formatCompact(value: number) { return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value) }
function formatCoord(value: number) { return value.toFixed(3) }
function apiError(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object' && 'error' in payload) return String(payload.error)
  return fallback
}

export default SftDataExplorer
