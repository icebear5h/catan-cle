import { useCallback, useEffect, useState } from 'react'
import BenchmarkVerifier from './verifier/BenchmarkVerifier'
import DecisionSpotChecks from './decisions/DecisionSpotChecks'
import TextFormatV3 from './textformat/TextFormatV3'
import ReasoningTraces from './reasoning/ReasoningTraces'
import SftDataExplorer from './sft/SftDataExplorer'
import SftEvalResults from './sft/SftEvalResults'
import BoardFluencyReview from './fluency/BoardFluencyReview'

type EvalSuite = 'decisions' | 'board' | 'board-fluency' | 'sft-data' | 'sft-eval' | 'text-v3' | 'reasoning'

function navigationFromUrl(): { suite: EvalSuite; rowId: string } {
  const params = new URLSearchParams(window.location.search)
  const tab = params.get('tab')
  const suite = tab === 'decisions' || tab === 'board' || tab === 'board-fluency'
    || tab === 'sft-eval' || tab === 'text-v3' || tab === 'reasoning'
    ? tab : 'sft-data'
  return { suite, rowId: suite === 'board-fluency' ? params.get('row') || '' : '' }
}

function App() {
  const [navigation, setNavigation] = useState(navigationFromUrl)
  const { suite, rowId } = navigation

  useEffect(() => {
    const onPopState = () => setNavigation(navigationFromUrl())
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  const setSuite = (next: EvalSuite) => {
    if (next === suite) return
    const url = new URL(window.location.href)
    url.searchParams.set('tab', next)
    url.searchParams.delete('row')
    window.history.pushState(null, '', url)
    setNavigation({ suite: next, rowId: '' })
  }

  const selectBoardRow = useCallback((nextRowId: string) => {
    const url = new URL(window.location.href)
    url.searchParams.set('tab', 'board-fluency')
    if (nextRowId) url.searchParams.set('row', nextRowId)
    else url.searchParams.delete('row')
    // Row browsing keeps the current URL shareable without flooding Back history.
    window.history.replaceState(null, '', url)
    setNavigation((current) => ({ ...current, rowId: nextRowId }))
  }, [])

  return (
    <div className="app">
      <header>
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>Catan Eval Suite</h1>
            <p>{suiteSubtitle(suite)}</p>
          </div>
        </div>
        <nav className="mode-switch" aria-label="Evaluation suite">
          <button
            className={suite === 'decisions' ? 'active' : ''}
            onClick={() => setSuite('decisions')}
          >
            Agent Decisions
          </button>
          <button
            className={suite === 'board' ? 'active' : ''}
            onClick={() => setSuite('board')}
          >
            Board Perception
          </button>
          <button
            className={suite === 'board-fluency' ? 'active' : ''}
            onClick={() => setSuite('board-fluency')}
          >
            Board Fluency
          </button>
          <button
            className={suite === 'sft-data' ? 'active' : ''}
            onClick={() => setSuite('sft-data')}
          >
            SFT Data
          </button>
          <button
            className={suite === 'sft-eval' ? 'active' : ''}
            onClick={() => setSuite('sft-eval')}
          >
            SFT Eval
          </button>
          <button
            className={suite === 'text-v3' ? 'active' : ''}
            onClick={() => setSuite('text-v3')}
          >
            Text Format v3
          </button>
          <button
            className={suite === 'reasoning' ? 'active' : ''}
            onClick={() => setSuite('reasoning')}
          >
            Reasoning Traces
          </button>
        </nav>
      </header>
      {suite === 'decisions' && <DecisionSpotChecks />}
      {suite === 'board' && <BenchmarkVerifier />}
      {suite === 'board-fluency' && <BoardFluencyReview selectedRowId={rowId} onSelectRow={selectBoardRow} />}
      {suite === 'sft-data' && <SftDataExplorer />}
      {suite === 'sft-eval' && <SftEvalResults />}
      {suite === 'text-v3' && <TextFormatV3 />}
      {suite === 'reasoning' && <ReasoningTraces />}
    </div>
  )
}

function suiteSubtitle(suite: EvalSuite) {
  if (suite === 'decisions') {
    return 'agent decisions / bucketed spot checks'
  }
  if (suite === 'board') {
    return 'board perception / human verification'
  }
  if (suite === 'board-fluency') {
    return 'symbolic board fluency / exact dataset review'
  }
  if (suite === 'sft-data') {
    return 'training corpora / board and patch inspection'
  }
  if (suite === 'sft-eval') {
    return 'published checkpoint / held-out generation'
  }
  if (suite === 'text-v3') {
    return 'indexed tile rows / frozen v3 audit'
  }
  return 'fresh initial settlement / native reasoning'
}

export default App
