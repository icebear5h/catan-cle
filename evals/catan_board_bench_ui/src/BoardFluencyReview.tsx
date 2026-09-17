import { useEffect, useMemo, useRef, useState } from 'react'
import HexBoard from '@playground/components/HexBoard'
import type { GameState } from '@playground/types'
import './SftDataExplorer.css'
import './BenchmarkVerifier.css'
import './BoardFluencyReview.css'

const REVIEW_URL = '/board-fluency-review/preview.json'
const FAMILIES = [
  'relations_joins',
  'sets_coverage',
  'aggregation_comparison',
  'connectivity_structure',
  'constraints_consequences',
] as const

type Family = typeof FAMILIES[number]
type ReviewRow = {
  row_id: string
  state_id: string
  family: Family
  operation: string
  question: string
  answer: string
  prompt: string
  source: {
    row_id: string
    line: number
    density: string
    map_id: string
    source_kind: string
  }
}

type ReviewBundle = {
  schema: 'catan_board_fluency_review/v1'
  title: string
  metadata: {
    row_count: number
    counts_by_family: Record<string, number>
    counts_by_operation: Record<string, number>
    unique_state_count: number
    [key: string]: unknown
  }
  rows: ReviewRow[]
  boards: Record<string, GameState>
}

type Filters = { family: string; operation: string; query: string }
const DEFAULT_FILTERS: Filters = { family: 'all', operation: 'all', query: '' }

function matchingRows(rows: ReviewRow[], filters: Filters) {
  const query = filters.query.trim().toLowerCase()
  return rows.filter((row) => (
    (filters.family === 'all' || row.family === filters.family)
    && (filters.operation === 'all' || row.operation === filters.operation)
    && (!query || [
      row.row_id, row.state_id, row.family, row.operation,
      row.question, row.answer, row.prompt, ...Object.values(row.source),
    ].join('\n').toLowerCase().includes(query))
  ))
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function isCounts(value: unknown): value is Record<string, number> {
  return isRecord(value) && Object.values(value).every(
    (count) => typeof count === 'number' && Number.isInteger(count) && count >= 0,
  )
}

function parseBundle(value: unknown): ReviewBundle {
  if (!isRecord(value) || value.schema !== 'catan_board_fluency_review/v1'
    || typeof value.title !== 'string' || !Array.isArray(value.rows)
    || !isRecord(value.boards) || !isRecord(value.metadata)
    || value.metadata.row_count !== value.rows.length
    || typeof value.metadata.unique_state_count !== 'number'
    || !isCounts(value.metadata.counts_by_family)
    || !isCounts(value.metadata.counts_by_operation)) {
    throw new Error('Invalid Board Fluency preview: expected the catan_board_fluency_review/v1 bundle with matching row_count.')
  }
  const ids = new Set<string>()
  for (const [index, row] of value.rows.entries()) {
    if (!isRecord(row)
      || !['row_id', 'state_id', 'family', 'operation', 'question', 'answer', 'prompt']
        .every((key) => typeof row[key] === 'string')
      || !row.row_id || ids.has(String(row.row_id))
      || !FAMILIES.some((family) => family === row.family)
      || !isRecord(row.source) || !Number.isInteger(row.source.line)
      || !['row_id', 'density', 'map_id', 'source_kind'].every((key) => (
        isRecord(row.source) && typeof row.source[key] === 'string'
      ))) {
      throw new Error(`Invalid Board Fluency preview: malformed or duplicate row at position ${index + 1}.`)
    }
    ids.add(String(row.row_id))
  }
  return value as ReviewBundle
}

function BoardFluencyReview({ selectedRowId, onSelectRow }: {
  selectedRowId: string
  onSelectRow: (rowId: string) => void
}) {
  const [bundle, setBundle] = useState<ReviewBundle | null>(null)
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadAttempt, setLoadAttempt] = useState(0)
  const rowListRef = useRef<HTMLDivElement>(null)
  const activeRowRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    const load = async () => {
      setIsLoading(true)
      setError(null)
      setBundle(null)
      try {
        const response = await fetch(REVIEW_URL, { signal: controller.signal, cache: 'no-store' })
        if (!response.headers.get('content-type')?.includes('application/json')) {
          throw new Error(`Board Fluency preview returned HTTP ${response.status} without JSON. The eval Vite server must serve ${REVIEW_URL}.`)
        }
        const data: unknown = await response.json()
        if (!response.ok) {
          throw new Error(isRecord(data) && typeof data.error === 'string'
            ? data.error : `Failed to load Board Fluency preview (HTTP ${response.status}).`)
        }
        const next = parseBundle(data)
        if (!controller.signal.aborted) setBundle(next)
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : String(loadError))
        }
      } finally {
        if (!controller.signal.aborted) setIsLoading(false)
      }
    }
    void load()
    return () => controller.abort()
  }, [loadAttempt])

  const rows = useMemo(() => matchingRows(bundle?.rows || [], filters), [bundle, filters])
  const operations = useMemo(() => {
    const counts = new Map<string, number>()
    for (const row of bundle?.rows || []) {
      if (filters.family === 'all' || row.family === filters.family) {
        counts.set(row.operation, (counts.get(row.operation) || 0) + 1)
      }
    }
    return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b))
  }, [bundle, filters.family])
  const selectedIndex = rows.findIndex((row) => row.row_id === selectedRowId)
  const selectedRow = rows[selectedIndex]
  const board = selectedRow ? bundle?.boards[selectedRow.state_id] : undefined
  const unknownRow = Boolean(bundle && selectedRowId
    && !bundle.rows.some((row) => row.row_id === selectedRowId))

  useEffect(() => {
    if (!selectedRowId && rows.length) onSelectRow(rows[0].row_id)
  }, [rows, selectedRowId, onSelectRow])

  useEffect(() => {
    // Scroll just the rail; inspecting another row must not move the whole page.
    const list = rowListRef.current
    const active = activeRowRef.current
    if (!list || !active) return
    const listBounds = list.getBoundingClientRect()
    const rowBounds = active.getBoundingClientRect()
    if (rowBounds.top < listBounds.top) list.scrollTop += rowBounds.top - listBounds.top
    else if (rowBounds.bottom > listBounds.bottom) list.scrollTop += rowBounds.bottom - listBounds.bottom
  }, [selectedRowId, rows])

  const updateFilters = (next: Filters) => {
    setFilters(next)
    const matches = matchingRows(bundle?.rows || [], next)
    if (!matches.some((row) => row.row_id === selectedRowId)) {
      onSelectRow(matches[0]?.row_id || '')
    }
  }

  const stepRow = (delta: number) => {
    const next = rows[selectedIndex + delta]
    if (next) onSelectRow(next.row_id)
  }

  return (
    <main className="sft-shell board-fluency" aria-busy={isLoading}>
      <section className="panel-surface">
        <div className="panel-heading"><span>Board Fluency</span><span>text-only dataset review</span></div>
        <h2>{bundle?.title || 'Symbolic board fluency — review v1'}</h2>
        <p className="board-fluency-note">
          Inspect the exact question, gold answer, and model input. The annotated board
          is a review aid reconstructed from the source render_state; model input is text only.
        </p>
        <nav className="board-fluency-links" aria-label="Review bundle downloads">
          <a href="/board-fluency-review/review.jsonl" download>Download review.jsonl</a>
          <a href="/board-fluency-review/metadata.json" download>Download metadata.json</a>
          <a href={REVIEW_URL} download>Download preview.json</a>
        </nav>
      </section>

      {isLoading && <div className="bench-loading" role="status">Loading Board Fluency review…</div>}
      {error && (
        <div className="sft-error" role="alert">
          <p>{error}</p>
          <button className="sft-reset" onClick={() => setLoadAttempt((attempt) => attempt + 1)}>Retry loading</button>
        </div>
      )}

      {bundle && (
        <>
          <section className="sft-stat-grid" aria-label="Review counts">
            <div className="sft-stat"><span>Dataset rows</span><strong>{bundle.rows.length}</strong><small>loaded from preview.json</small></div>
            <div className="sft-stat"><span>Unique boards</span><strong>{bundle.metadata.unique_state_count}</strong><small>source states in this review</small></div>
            <div className="sft-stat"><span>Families / operations</span><strong>{new Set(bundle.rows.map((row) => row.family)).size} / {Object.keys(bundle.metadata.counts_by_operation).length}</strong><small>filter by exact IDs below</small></div>
            <div className="sft-stat"><span>Matching rows</span><strong>{rows.length}</strong><small>{new Set(rows.map((row) => row.state_id)).size} matching boards</small></div>
          </section>

          <section className="sft-controls" aria-label="Board Fluency filters">
            <label className="sft-select-wrap">
              <span>Family</span>
              <select value={filters.family} onChange={(event) => updateFilters({ ...filters, family: event.target.value, operation: 'all' })}>
                <option value="all">All families ({bundle.rows.length})</option>
                {FAMILIES.map((family) => <option key={family} value={family}>{family} ({bundle.metadata.counts_by_family[family] || 0})</option>)}
              </select>
            </label>
            <label className="sft-select-wrap">
              <span>Operation</span>
              <select value={filters.operation} onChange={(event) => updateFilters({ ...filters, operation: event.target.value })}>
                <option value="all">All operations ({operations.reduce((total, [, count]) => total + count, 0)})</option>
                {operations.map(([operation, count]) => <option key={operation} value={operation}>{operation} ({count})</option>)}
              </select>
            </label>
            <label className="sft-search">
              <span>Search all row text</span>
              <input type="search" value={filters.query} onChange={(event) => updateFilters({ ...filters, query: event.target.value })} placeholder="Question, gold, prompt, token, source…" />
            </label>
            <button className="sft-reset" onClick={() => updateFilters(DEFAULT_FILTERS)}>Reset</button>
          </section>

          {unknownRow && <div className="sft-error" role="alert">Row <code>{selectedRowId}</code> is not present in this review bundle. Choose a row below.</div>}

          <section className="sft-workbench">
            <aside className="sft-row-rail" aria-label="Review rows">
              <header>
                <div><span>Rows matching filters</span><strong>{rows.length} / {bundle.rows.length}</strong></div>
                <small>{selectedIndex >= 0 ? selectedIndex + 1 : 0} selected</small>
              </header>
              <div className="sft-row-list" ref={rowListRef}>
                {rows.map((row) => (
                  <button
                    key={row.row_id}
                    ref={row.row_id === selectedRowId ? activeRowRef : undefined}
                    className={row.row_id === selectedRowId ? 'active' : ''}
                    aria-current={row.row_id === selectedRowId ? 'true' : undefined}
                    onClick={() => onSelectRow(row.row_id)}
                  >
                    <span className="sft-row-token" title={row.row_id}>{row.row_id}</span>
                    <small title={row.operation}>{row.operation}</small>
                    <small title={row.question}>{row.question}</small>
                  </button>
                ))}
                {!rows.length && <div className="sft-empty">{bundle.rows.length ? 'No rows match these filters.' : 'The review bundle contains no rows.'}</div>}
              </div>
              <footer>
                <button disabled={selectedIndex <= 0} onClick={() => stepRow(-1)}>Previous row</button>
                <button disabled={selectedIndex < 0 || selectedIndex >= rows.length - 1} onClick={() => stepRow(1)}>Next row</button>
              </footer>
            </aside>

            <div className="sft-detail">
              {selectedRow ? (
                <>
                  <div className="sft-detail-topline">
                    <div><code>{selectedRow.family}</code><code>{selectedRow.operation}</code></div>
                    <div className="sft-stepper">
                      <button aria-label="Previous matching row" disabled={selectedIndex <= 0} onClick={() => stepRow(-1)}>←</button>
                      <code>{selectedIndex + 1} / {rows.length} matches</code>
                      <button aria-label="Next matching row" disabled={selectedIndex >= rows.length - 1} onClick={() => stepRow(1)}>→</button>
                    </div>
                  </div>

                  <div className="sft-example-grid">
                    <article className="sft-conversation">
                      <header>
                        <span>Exact dataset row</span><code>{selectedRow.row_id}</code>
                        <a href={`?${new URLSearchParams({ tab: 'board-fluency', row: selectedRow.row_id })}`}>Row permalink</a>
                      </header>
                      <div className="sft-message user"><span>Exact question</span><pre>{selectedRow.question}</pre></div>
                      <div className="sft-message assistant"><span>Gold answer · exact</span><pre>{selectedRow.answer}</pre></div>
                    </article>

                    <figure className="board-fluency-board">
                      <figcaption className="board-fluency-note">
                        <strong>Annotated board · review aid only</strong>
                        <code>{selectedRow.state_id}</code>
                        Model input is text only; this rendering is not included in the prompt.
                      </figcaption>
                      {board && Array.isArray(board.tiles) && board.tiles.length > 0
                        && isRecord(board.nodes) && Array.isArray(board.edges) ? (
                          <div className="board-frame">
                            <div className="live-board-wrap" data-board-mode="annotated">
                              <HexBoard key={selectedRow.state_id} gameState={board} showAnnotations showControls={false} />
                            </div>
                          </div>
                        ) : <div className="sft-error" role="alert">Missing or invalid source render_state for {selectedRow.state_id} in preview.json.</div>}
                    </figure>
                  </div>

                  <details className="json-panel panel-surface board-fluency-disclosure" key={`prompt-${selectedRow.row_id}`}>
                    <summary>Exact model input · text only ({selectedRow.prompt.length.toLocaleString()} characters)</summary>
                    <pre>{selectedRow.prompt}</pre>
                  </details>
                  <details className="json-panel panel-surface board-fluency-disclosure" key={`source-${selectedRow.row_id}`}>
                    <summary>Source info · {selectedRow.source.source_kind} · line {selectedRow.source.line}</summary>
                    <dl className="sft-row-contract">
                      <div><dt>Review row</dt><dd><code>{selectedRow.row_id}</code></dd></div>
                      <div><dt>State ID</dt><dd><code>{selectedRow.state_id}</code></dd></div>
                    </dl>
                    <pre>{JSON.stringify(selectedRow.source, null, 2)}</pre>
                  </details>
                </>
              ) : <div className="sft-detail-empty">{rows.length ? 'Choose a row to inspect its question, gold answer, and source board.' : 'No matching rows. Adjust the filters or reset to browse the dataset.'}</div>}
            </div>
          </section>

          <details className="json-panel panel-surface board-fluency-disclosure">
            <summary>Dataset metadata · family and operation counts</summary>
            <pre>{JSON.stringify(bundle.metadata, null, 2)}</pre>
          </details>
        </>
      )}
    </main>
  )
}

export default BoardFluencyReview
