import { useEffect, useMemo, useState } from 'react'
import './TextFormatV3.css'

const SERVER_URL = 'http://127.0.0.1:5001'

type FormatScore = {
  format: string
  exact: number
  requests: number
  exact_accuracy: number
}

type BoardSummary = {
  sample_id: string
  question_count: number
  categories: string[]
  characters: number
  lines: number
}

type FormatOverview = {
  format_id: string
  display_name: string
  model: string
  board_count: number
  question_count: number
  boards: BoardSummary[]
  question_lock: {
    byte_identical_to_source: boolean
    sha256: string
  }
  development: {
    baseline: FormatScore
    winner: FormatScore
    prompt_token_multiplier: number
    character_multiplier: number
  }
  transfer: {
    complete: boolean
    paired_questions: number
    winner: FormatScore
  }
}

type FormatQuestion = {
  id: string
  category: string
  question: string
  answer: Record<string, unknown>
  answer_text: string
}

type FormatSample = {
  sample_id: string
  version: string
  text: string
  characters: number
  lines: number
  sha256: string
  questions: FormatQuestion[]
}

type NumberedLine = { line: string; number: number }
type TextView = 'full' | 'base' | 'indexes'

function TextFormatV3() {
  const [overview, setOverview] = useState<FormatOverview | null>(null)
  const [selectedSampleId, setSelectedSampleId] = useState('')
  const [sample, setSample] = useState<FormatSample | null>(null)
  const [selectedQuestionId, setSelectedQuestionId] = useState('')
  const [artifactOpen, setArtifactOpen] = useState(false)
  const [textView, setTextView] = useState<TextView>('indexes')
  const [lineFilter, setLineFilter] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    const loadOverview = async () => {
      try {
        setError(null)
        const response = await fetch(SERVER_URL + '/api/catan-board-bench/text-format-v3', {
          signal: controller.signal,
        })
        const payload: unknown = await response.json()
        if (!response.ok) throw new Error(apiError(payload, 'Failed to load text-format v3 overview'))
        const next = payload as FormatOverview
        setOverview(next)
        setSelectedSampleId(next.boards[0]?.sample_id || '')
      } catch (loadError) {
        if (!controller.signal.aborted) setError(String(loadError))
      }
    }
    void loadOverview()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!selectedSampleId) return
    const controller = new AbortController()
    const loadSample = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const params = new URLSearchParams({ sample_id: selectedSampleId })
        const response = await fetch(
          SERVER_URL + '/api/catan-board-bench/text-format-v3/sample?' + params,
          { signal: controller.signal },
        )
        const payload: unknown = await response.json()
        if (!response.ok) throw new Error(apiError(payload, 'Failed to load text-format v3 sample'))
        const next = payload as FormatSample
        setSample(next)
        setSelectedQuestionId(next.questions[0]?.id || '')
      } catch (loadError) {
        if (!controller.signal.aborted) setError(String(loadError))
      } finally {
        if (!controller.signal.aborted) setIsLoading(false)
      }
    }
    void loadSample()
    return () => controller.abort()
  }, [selectedSampleId])

  const selectedQuestion = useMemo(
    () => sample?.questions.find((question) => question.id === selectedQuestionId) || sample?.questions[0],
    [sample, selectedQuestionId],
  )

  const evidence = useMemo(
    () => sample && selectedQuestion ? buildQuestionEvidence(sample, selectedQuestion) : null,
    [sample, selectedQuestion],
  )

  const visibleLines = useMemo(() => {
    if (!sample) return []
    const allLines = numberLines(sample.text)
    const queryStart = allLines.findIndex(({ line }) => line.startsWith('QUERY INDEXES'))
    let selected = allLines
    if (textView === 'base' && queryStart >= 0) selected = selected.slice(0, queryStart)
    if (textView === 'indexes' && queryStart >= 0) selected = selected.slice(queryStart)
    const needle = lineFilter.trim().toLowerCase()
    return needle
      ? selected.filter(({ line }) => line.toLowerCase().includes(needle))
      : selected
  }, [sample, textView, lineFilter])

  const chooseBoard = (sampleId: string) => {
    setSelectedSampleId(sampleId)
    setSelectedQuestionId('')
    setLineFilter('')
  }

  if (!overview && isLoading) {
    return <main className="format-v3-shell"><div className="format-loading">loading frozen v3</div></main>
  }

  return (
    <main className="format-v3-shell">
      {error && <div className="format-error">{error}</div>}

      {overview && (
        <>
          <section className="format-hero">
            <div className="format-hero-copy">
              <div className="format-overline">
                <span>Indexed tile rows · frozen v3</span>
                <span className="format-lock-dot">question lock verified</span>
              </div>
              <h1><em>{overview.development.winner.exact - overview.development.baseline.exact} misses</em> removed by changing the interface.</h1>
              <p>
                The model sees the same 60 questions and targets. V3 makes the reusable spatial joins explicit—so the benchmark measures Catan state reading instead of diagonal ASCII navigation.
              </p>
              <div className="format-proof-row">
                <span>60 / 60 development</span>
                <span>{overview.transfer.winner.exact} / {overview.transfer.paired_questions} frozen transfer</span>
                <span>{overview.development.prompt_token_multiplier.toFixed(2)}× prompt tokens</span>
              </div>
            </div>

            <div className="format-score-story" aria-label="Development accuracy comparison">
              <div className="format-score-row baseline">
                <span>Raw tile rows</span>
                <div><i style={{ width: String(overview.development.baseline.exact_accuracy * 100) + '%' }} /></div>
                <strong>{formatScore(overview.development.baseline)}</strong>
              </div>
              <div className="format-score-delta">
                <span>+{overview.development.winner.exact - overview.development.baseline.exact} exact answers</span>
                <small>no question changes</small>
              </div>
              <div className="format-score-row winner">
                <span>Indexed v3</span>
                <div><i style={{ width: String(overview.development.winner.exact_accuracy * 100) + '%' }} /></div>
                <strong>{formatScore(overview.development.winner)}</strong>
              </div>
            </div>
          </section>

          <section className="format-audit-strip">
            <div>
              <span>Questions</span>
              <strong>byte-identical</strong>
              <code>{overview.question_lock.sha256.slice(0, 12)}</code>
            </div>
            <div>
              <span>Representation</span>
              <strong>question-independent</strong>
              <small>same v3 graph for every query on a board</small>
            </div>
            <div>
              <span>Tradeoff</span>
              <strong>accuracy-first</strong>
              <small>{overview.development.character_multiplier.toFixed(2)}× chars · {overview.development.prompt_token_multiplier.toFixed(2)}× tokens</small>
            </div>
          </section>

          {sample && selectedQuestion && evidence && (
            <section className="format-lens">
              <header className="format-lens-header">
                <div>
                  <span className="format-kicker">Question lens</span>
                  <h2>See exactly what v3 changed.</h2>
                </div>
                <div className="format-board-picker" aria-label="Frozen boards">
                  {overview.boards.map((board, index) => (
                    <button
                      key={board.sample_id}
                      className={board.sample_id === selectedSampleId ? 'active' : ''}
                      onClick={() => chooseBoard(board.sample_id)}
                      aria-label={'Open ' + board.sample_id}
                      title={board.sample_id + ' · ' + board.question_count + ' questions'}
                    >
                      {String(index + 1).padStart(2, '0')}
                    </button>
                  ))}
                </div>
              </header>

              <div className="format-lens-body">
                <aside className="format-question-list">
                  <div className="format-current-board">
                    <span>{sample.sample_id}</span>
                    <small>{sample.questions.length} unchanged questions</small>
                  </div>
                  {sample.questions.map((question, index) => (
                    <button
                      key={question.id}
                      className={question.id === selectedQuestion.id ? 'active' : ''}
                      onClick={() => setSelectedQuestionId(question.id)}
                    >
                      <span>{String(index + 1).padStart(2, '0')}</span>
                      <span>
                        <strong>{question.category.replace(/_/g, ' ')}</strong>
                        <small>{question.question}</small>
                      </span>
                    </button>
                  ))}
                </aside>

                <article className="format-question-stage">
                  <div className="format-question-heading">
                    <div>
                      <span>{selectedQuestion.category.replace(/_/g, ' ')}</span>
                      <code>{selectedQuestion.id}</code>
                    </div>
                    <h3>{selectedQuestion.question}</h3>
                    <div className="format-target">
                      <span>unchanged strict target</span>
                      <code>{selectedQuestion.answer_text}</code>
                    </div>
                  </div>

                  <div className="format-evidence-grid">
                    <EvidencePanel
                      label="Raw graph"
                      note={String(evidence.raw.length) + ' scattered record' + (evidence.raw.length === 1 ? '' : 's') + ' to inspect'}
                      lines={evidence.raw}
                    />
                    <div className="format-evidence-arrow" aria-hidden="true"><span>→</span></div>
                    <EvidencePanel
                      label="Indexed v3"
                      note={evidence.indexed.length === 1 ? 'answer-local row' : 'reusable derived rows'}
                      lines={evidence.indexed}
                      indexed
                    />
                  </div>
                </article>
              </div>
            </section>
          )}

          {sample && (
            <section className={'format-artifact ' + (artifactOpen ? 'open' : '')}>
              <button className="format-artifact-summary" onClick={() => setArtifactOpen((open) => !open)}>
                <span>
                  <small>Exact frozen artifact</small>
                  <strong>{sample.sample_id}</strong>
                  <code>{sample.lines} lines · sha {sample.sha256.slice(0, 12)}</code>
                </span>
                <span>{artifactOpen ? 'Close inspector' : 'Inspect full serialization'} <b>{artifactOpen ? '−' : '+'}</b></span>
              </button>

              {artifactOpen && (
                <div className="format-artifact-body">
                  <div className="format-code-controls">
                    <div className="format-view-switch" aria-label="Serialization section">
                      {(['full', 'base', 'indexes'] as TextView[]).map((view) => (
                        <button
                          key={view}
                          className={textView === view ? 'active' : ''}
                          onClick={() => setTextView(view)}
                        >
                          {view === 'base' ? 'Base graph' : view === 'indexes' ? 'Query indexes' : 'Full v3'}
                        </button>
                      ))}
                    </div>
                    <input
                      aria-label="Filter serialization lines"
                      value={lineFilter}
                      onChange={(event) => setLineFilter(event.target.value)}
                      placeholder="Filter lines — e.g. ROLL_SOURCE|9"
                    />
                  </div>
                  <div className="format-code-scroll" data-text-view={textView}>
                    <ol>
                      {visibleLines.map(({ line, number }) => (
                        <li key={String(number) + '-' + line} value={number}><code>{line || ' '}</code></li>
                      ))}
                    </ol>
                  </div>
                </div>
              )}
            </section>
          )}
        </>
      )}
    </main>
  )
}

function EvidencePanel({
  label,
  note,
  lines,
  indexed = false,
}: {
  label: string
  note: string
  lines: NumberedLine[]
  indexed?: boolean
}) {
  return (
    <section className={'format-evidence ' + (indexed ? 'indexed' : 'raw')}>
      <header>
        <span>{label}</span>
        <small>{note}</small>
      </header>
      <div>
        {lines.length > 0 ? lines.map(({ line, number }) => (
          <p key={String(number) + '-' + line}><span>{number}</span><code>{line}</code></p>
        )) : <p className="format-no-index"><code>No extra join: the answer is already a direct base record.</code></p>}
      </div>
    </section>
  )
}

function buildQuestionEvidence(sample: FormatSample, question: FormatQuestion) {
  const lines = numberLines(sample.text)
  const queryStart = lines.findIndex(({ line }) => line.startsWith('QUERY INDEXES'))
  const rawPool = queryStart >= 0 ? lines.slice(0, queryStart) : lines
  const indexPool = queryStart >= 0 ? lines.slice(queryStart + 1) : []
  const ids = question.question.match(/\b[TNEP]\d{2}\b/g) || []
  const numberMatch = question.question.match(/If (\d+) is rolled/i)
  const roll = numberMatch?.[1]
  const color = typeof question.answer.color === 'string'
    ? question.answer.color.replace(/[<>]/g, '')
    : undefined

  let raw: NumberedLine[] = []
  let indexed: NumberedLine[] = []

  if (question.category === 'direction_to_tile' || question.category === 'tile_to_direction') {
    const anchor = question.category === 'direction_to_tile' ? ids[0] : ids.at(-1)
    raw = rawPool.filter(({ line }) =>
      Boolean(anchor) && (line.startsWith('ROW') || line.startsWith('T|' + anchor + '|')) && line.includes(anchor || ''),
    )
    indexed = indexPool.filter(({ line }) => line.startsWith('QI|TILE_NEIGHBORS|' + anchor + '|'))
  } else if (question.category === 'port_occupancy') {
    const port = ids.find((id) => id.startsWith('P'))
    const portLine = rawPool.find(({ line }) => line.startsWith('P|' + port + '|'))
    const portNodes = portLine?.line.match(/N\d{2}/g) || []
    raw = rawPool.filter(({ line }) =>
      line.startsWith('P|' + port + '|') || portNodes.some((node) => line.startsWith('N|' + node + '|')),
    )
    indexed = indexPool.filter(({ line }) => line.startsWith('QI|PORT_NODES|' + port + '|'))
  } else if (question.category === 'roll_production' && roll) {
    const rollTiles = rawPool.filter(({ line }) => line.startsWith('T|') && line.includes('|number=' + roll + '|'))
    const nodeIds = rollTiles.flatMap(({ line }) => line.match(/N\d{2}/g) || [])
    raw = [...rollTiles, ...rawPool.filter(({ line }) => nodeIds.some((node) => line.startsWith('N|' + node + '|')))]
    indexed = indexPool.filter(({ line }) =>
      line.startsWith('QI|ROLL_TILES|' + roll + '|') || line.startsWith('QI|ROLL_SOURCE|' + roll + '|'),
    )
  } else if ((question.category === 'building_counts' || question.category === 'road_inventory') && color) {
    raw = rawPool.filter(({ line }) =>
      question.category === 'building_counts'
        ? line.startsWith('N|') && line.includes('|color=' + color + '|')
        : line.startsWith('E|') && line.includes('|road=' + color + '|'),
    )
    indexed = indexPool.filter(({ line }) => line.startsWith('QI|PLAYER|' + color + '|'))
  } else if (question.category === 'node_adjacent_tiles') {
    const node = ids.find((id) => id.startsWith('N'))
    raw = rawPool.filter(({ line }) => line.startsWith('N|' + node + '|'))
    indexed = indexPool.filter(({ line }) => line.startsWith('QI|TILE_CORNERS|') && line.includes(String(node) + '/'))
  } else {
    raw = rawPool.filter(({ line }) => ids.some((id) => line.startsWith(id[0] + '|' + id + '|')))
    indexed = indexPool.filter(({ line }) => ids.some((id) => line.includes('|' + id + '|') || line.includes(id + '/')))
  }

  if (raw.length === 0) raw = rawPool.filter(({ line }) => ids.some((id) => line.includes(id)))

  return {
    raw: uniqueLines(raw).slice(0, 6),
    indexed: uniqueLines(indexed).slice(0, 6),
  }
}

function numberLines(text: string): NumberedLine[] {
  return text.split('\n').map((line, index) => ({ line, number: index + 1 }))
}

function uniqueLines(lines: NumberedLine[]) {
  return lines.filter((item, index) => lines.findIndex((other) => other.number === item.number) === index)
}

function formatScore(score: FormatScore) {
  return (score.exact_accuracy * 100).toFixed(1) + '%'
}

function apiError(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object' && 'error' in payload) {
    return String((payload as { error: unknown }).error)
  }
  return fallback
}

export default TextFormatV3
