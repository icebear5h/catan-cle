import { useState } from 'react'
import BenchmarkVerifier from './BenchmarkVerifier'
import DecisionSpotChecks from './DecisionSpotChecks'
import TextFormatV3 from './TextFormatV3'
import ReasoningTraces from './ReasoningTraces'
import SftDataExplorer from './SftDataExplorer'
import SftEvalResults from './SftEvalResults'

type EvalSuite = 'decisions' | 'board' | 'sft-data' | 'sft-eval' | 'text-v3' | 'reasoning'

function App() {
  const [suite, setSuite] = useState<EvalSuite>('sft-data')

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
  if (suite === 'sft-data') {
    return 'empty-board spatial grounding / corpus audit'
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
