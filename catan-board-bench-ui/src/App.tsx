import { useState } from 'react'
import BenchmarkVerifier from './BenchmarkVerifier'
import DecisionSpotChecks from './DecisionSpotChecks'
import TextFormatV3 from './TextFormatV3'

type EvalSuite = 'decisions' | 'board' | 'text-v3'

function App() {
  const [suite, setSuite] = useState<EvalSuite>('decisions')

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
            className={suite === 'text-v3' ? 'active' : ''}
            onClick={() => setSuite('text-v3')}
          >
            Text Format v3
          </button>
        </nav>
      </header>
      {suite === 'decisions' && <DecisionSpotChecks />}
      {suite === 'board' && <BenchmarkVerifier />}
      {suite === 'text-v3' && <TextFormatV3 />}
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
  return 'indexed tile rows / frozen v3 audit'
}

export default App
