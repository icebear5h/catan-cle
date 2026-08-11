import BenchmarkVerifier from './BenchmarkVerifier'

function App() {
  return (
    <div className="app">
      <header>
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>CatanBench Verifier</h1>
            <p>board-image benchmark / human verification</p>
          </div>
        </div>
      </header>
      <BenchmarkVerifier />
    </div>
  )
}

export default App
