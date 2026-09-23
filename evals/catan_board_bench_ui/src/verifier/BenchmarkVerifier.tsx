import { useEffect, useMemo, useState } from 'react';
import HexBoard from '@playground/components/board/HexBoard';
import './BenchmarkVerifier.css';
import type {
  EvalComparisonResponse,
  ExampleListResponse,
  SampleResponse,
  Verdict,
} from './types';
import EvalComparisonPanel from './EvalComparisonPanel';
import JsonPanel from './JsonPanel';
import QuestionCard from './QuestionCard';

const SERVER_URL = 'http://127.0.0.1:5001';

function BenchmarkVerifier() {
  const [categories, setCategories] = useState<string[]>([]);
  const [samples, setSamples] = useState<string[]>([]);
  const [sampleCounts, setSampleCounts] = useState<Record<string, number>>({});
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [selectedQuestionId, setSelectedQuestionId] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [isLoadingList, setIsLoadingList] = useState(false);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const [isLoadingComparison, setIsLoadingComparison] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [boardMode, setBoardMode] = useState<'annotated' | 'source'>('annotated');
  const [comparison, setComparison] = useState<EvalComparisonResponse | null>(null);

  useEffect(() => {
    loadExamples(selectedCategory);
  }, [selectedCategory]);

  useEffect(() => {
    loadEvalComparison();
  }, []);

  useEffect(() => {
    if (selectedSampleId) {
      loadSample(selectedSampleId);
    }
  }, [selectedSampleId]);

  const selectedIndex = useMemo(() => {
    return samples.findIndex((item) => item === selectedSampleId);
  }, [samples, selectedSampleId]);

  const selectedQuestion = useMemo(() => {
    if (!sample) {
      return null;
    }
    return sample.questions.find((item) => item.id === selectedQuestionId) || sample.questions[0] || null;
  }, [sample, selectedQuestionId]);

  const loadExamples = async (category: string) => {
    try {
      setIsLoadingList(true);
      setError(null);
      const params = new URLSearchParams({ limit: '240' });
      if (category !== 'all') {
        params.set('category', category);
      }
      const response = await fetch(`${SERVER_URL}/api/catan-board-bench/examples?${params}`);
      const data: ExampleListResponse = await response.json();
      if (!response.ok) {
        throw new Error((data as any).error || 'Failed to load CatanBoardBench examples');
      }
      setCategories(data.categories);
      setSamples(data.samples);
      setSampleCounts(data.sample_counts || {});
      setSelectedSampleId((current) => {
        if (current && data.samples.includes(current)) {
          return current;
        }
        return data.samples[0] || null;
      });
    } catch (loadError) {
      setError(String(loadError));
    } finally {
      setIsLoadingList(false);
    }
  };

  const loadSample = async (sampleId: string) => {
    try {
      setIsLoadingSample(true);
      setError(null);
      const response = await fetch(`${SERVER_URL}/api/catan-board-bench/sample?sample_id=${encodeURIComponent(sampleId)}`);
      const data: SampleResponse = await response.json();
      if (!response.ok) {
        throw new Error((data as any).error || 'Failed to load CatanBoardBench sample');
      }
      setSample(data);
      setSelectedQuestionId((current) => {
        if (current && data.questions.some((item) => item.id === current)) {
          return current;
        }
        if (selectedCategory !== 'all') {
          return data.questions.find((item) => item.category === selectedCategory)?.id || data.questions[0]?.id || null;
        }
        return data.questions[0]?.id || null;
      });
    } catch (loadError) {
      setError(String(loadError));
    } finally {
      setIsLoadingSample(false);
    }
  };

  const stepSelection = (delta: number) => {
    if (!samples.length) {
      return;
    }
    const nextIndex = Math.max(0, Math.min(samples.length - 1, selectedIndex + delta));
    setSelectedSampleId(samples[nextIndex] || null);
  };

  const setVerdict = (questionId: string, verdict: Verdict) => {
    if (!questionId) {
      return;
    }
    setVerdicts((current) => ({ ...current, [questionId]: verdict }));
  };

  const loadEvalComparison = async () => {
    try {
      setIsLoadingComparison(true);
      setError(null);
      const response = await fetch(`${SERVER_URL}/api/catan-board-bench/eval-comparison`);
      const data: EvalComparisonResponse = await response.json();
      if (!response.ok) {
        throw new Error((data as any).error || 'Failed to load eval comparison');
      }
      setComparison(data);
    } catch (loadError) {
      setError(String(loadError));
    } finally {
      setIsLoadingComparison(false);
    }
  };

  return (
    <main className="bench-shell">
      <section className="bench-rail panel-surface">
        <div className="panel-heading">
          <span>CatanBoardBench-100</span>
          <span>{samples.length} boards</span>
        </div>

        <label className="field-label" htmlFor="bench-category">Category</label>
        <select
          id="bench-category"
          className="bench-select"
          value={selectedCategory}
          onChange={(event) => setSelectedCategory(event.target.value)}
        >
          <option value="all">all</option>
          {categories.map((category) => (
            <option key={category} value={category}>{category}</option>
          ))}
        </select>

        <div className="bench-nav">
          <button className="matrix-button" onClick={() => stepSelection(-1)} disabled={selectedIndex <= 0}>
            Prev
          </button>
          <span className="bench-position">
            {selectedIndex >= 0 ? selectedIndex + 1 : 0}/{samples.length}
          </span>
          <button
            className="matrix-button"
            onClick={() => stepSelection(1)}
            disabled={selectedIndex < 0 || selectedIndex >= samples.length - 1}
          >
            Next
          </button>
        </div>

        <div className="example-list">
          {samples.map((sampleId) => (
            <button
              key={sampleId}
              className={`example-row ${sampleId === selectedSampleId ? 'active' : ''}`}
              onClick={() => setSelectedSampleId(sampleId)}
            >
              <span>{sampleId}</span>
              <strong>{sampleCounts[sampleId] || 0} rows</strong>
            </button>
          ))}
        </div>
      </section>

      <section className="bench-main">
        {comparison && <EvalComparisonPanel data={comparison} />}
        {error && <div className="bench-error">{error}</div>}
        {(isLoadingList || isLoadingSample || isLoadingComparison) && <div className="bench-loading">loading benchmark board</div>}

        {sample && (
          <>
            <div className="bench-hero panel-surface">
              <div className="board-frame">
                <div className="image-mode-switch" aria-label="Board render mode">
                  <button
                    className={boardMode === 'annotated' ? 'active' : ''}
                    onClick={() => setBoardMode('annotated')}
                  >
                    Annotated
                  </button>
                  <button
                    className={boardMode === 'source' ? 'active' : ''}
                    onClick={() => setBoardMode('source')}
                  >
                    Source PNG
                  </button>
                </div>
                {boardMode === 'annotated' ? (
                  <div className="live-board-wrap" data-board-mode="annotated">
                    <HexBoard
                      gameState={sample.render_state}
                      showAnnotations
                      showControls={false}
                    />
                  </div>
                ) : (
                  <img
                    src={`${SERVER_URL}${sample.image_url}`}
                    alt={`Catan board for ${sample.sample_id}`}
                  />
                )}
              </div>

              <div className="qa-console">
                <div className="panel-heading">
                  <span>{sample.sample_id}</span>
                  <span>{sample.questions.length} questions</span>
                </div>
                <div className="question-stack">
                  {sample.questions.map((question) => (
                    <QuestionCard
                      key={question.id}
                      question={question}
                      isActive={question.id === selectedQuestion?.id}
                      verdict={verdicts[question.id]}
                      onSelect={() => setSelectedQuestionId(question.id)}
                      onVerdict={(verdict) => setVerdict(question.id, verdict)}
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className="bench-detail-grid">
              <JsonPanel title="Selected Target" value={selectedQuestion?.target || {}} />
              <JsonPanel title="Selected Context" value={selectedQuestion?.contract_context || sample.contract_context} />
            </div>
          </>
        )}
      </section>
    </main>
  );
}

export default BenchmarkVerifier;
