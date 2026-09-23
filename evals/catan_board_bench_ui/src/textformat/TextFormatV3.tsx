import { useEffect, useMemo, useState } from 'react'
import './TextFormatV3.css'
import type { FormatOverview, FormatSample, TextView } from './types'
import EvidencePanel from './EvidencePanel'
import { buildQuestionEvidence, numberLines } from './evidence'
import { apiError, formatScore } from './format'

const SERVER_URL = 'http://127.0.0.1:5001'

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

export default TextFormatV3
