import { useEffect, useMemo, useState } from 'react'
import type { Filters, SpatialPayload } from './explorerTypes'
import {
  apiError, atlasNote, bboxStyle, cleanPrompt, controlLabel, formatCompact, formatInteger,
  humanize, markerLabel, patchLabel, rowKind, supervisionLabel,
} from './explorerFormat'
import { Contract, CurriculumPanel, Distribution, FilterSelect, Stat } from './ExplorerPanels'
import './SftDataExplorer.css'
import PatchInspector from './PatchInspector'

const SERVER_URL = 'http://127.0.0.1:5001'
const PAGE_SIZE = 80

const STAGE_SPLITS: Record<string, string[]> = {
  stage1: ['train', 'validation', 'test'],
  stage2: ['train', 'validation', 'test'],
  probes: ['validation', 'test'],
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

export default SftDataExplorer
