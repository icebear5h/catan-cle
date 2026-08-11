import { useEffect, useMemo, useState } from 'react';
import type { GameState } from '@playground/types';
import HexBoard from '@playground/components/HexBoard';
import './BenchmarkVerifier.css';

const SERVER_URL = 'http://127.0.0.1:5001';

type CompactExample = {
  id: string;
  sample_id: string;
  category: string;
  question: string;
  answer: string;
};

type QuestionDetail = CompactExample & {
  target: Record<string, unknown>;
  scoring: string | null;
  contract_context: Record<string, unknown>;
};

type SampleResponse = {
  sample_id: string;
  sample_index: number;
  sample_count: number;
  image_path: string;
  image_url: string;
  contract_path: string;
  annotations_url: string;
  render_state: GameState;
  contract_context: Record<string, unknown>;
  questions: QuestionDetail[];
};

type CategoryScore = {
  attempted: number;
  requests: number;
  errors: number;
  exact_accuracy: number;
  component_accuracy: number;
  avg_latency_ms: number;
};

type EvalModelSummary = {
  model: string;
  display_name: string;
  attempted: number;
  exact_accuracy: number;
  component_accuracy: number;
  avg_latency_ms: number;
  errors: number;
  categories: Record<string, CategoryScore>;
};

type EvalComparisonResponse = {
  requested: string[];
  categories: string[];
  models: EvalModelSummary[];
};

type ExampleListResponse = {
  examples: CompactExample[];
  total: number;
  categories: string[];
  samples: string[];
  sample_counts: Record<string, number>;
};

type Verdict = 'correct' | 'wrong' | 'unclear';

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
      const response = await fetch(`${SERVER_URL}/api/catanbench/examples?${params}`);
      const data: ExampleListResponse = await response.json();
      if (!response.ok) {
        throw new Error((data as any).error || 'Failed to load CatanBench examples');
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
      const response = await fetch(`${SERVER_URL}/api/catanbench/sample?sample_id=${encodeURIComponent(sampleId)}`);
      const data: SampleResponse = await response.json();
      if (!response.ok) {
        throw new Error((data as any).error || 'Failed to load CatanBench sample');
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
      const response = await fetch(`${SERVER_URL}/api/catanbench/eval-comparison`);
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
          <span>CatanBench 100</span>
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

function QuestionCard({
  question,
  isActive,
  verdict,
  onSelect,
  onVerdict,
}: {
  question: QuestionDetail;
  isActive: boolean;
  verdict?: Verdict;
  onSelect: () => void;
  onVerdict: (verdict: Verdict) => void;
}) {
  return (
    <article className={`question-card ${isActive ? 'active' : ''}`} onClick={onSelect}>
      <div className="question-card-top">
        <span>{question.category}</span>
        <code>{question.id}</code>
      </div>
      <p>{question.question}</p>
      <div className="question-card-answer">
        <span>Expected</span>
        <code>{question.answer}</code>
      </div>
      <div className="verdict-strip compact" aria-label={`Human verification verdict for ${question.id}`}>
        <button
          className={`verdict-button good ${verdict === 'correct' ? 'active' : ''}`}
          onClick={(event) => {
            event.stopPropagation();
            onVerdict('correct');
          }}
        >
          Correct
        </button>
        <button
          className={`verdict-button bad ${verdict === 'wrong' ? 'active' : ''}`}
          onClick={(event) => {
            event.stopPropagation();
            onVerdict('wrong');
          }}
        >
          Wrong
        </button>
        <button
          className={`verdict-button mute ${verdict === 'unclear' ? 'active' : ''}`}
          onClick={(event) => {
            event.stopPropagation();
            onVerdict('unclear');
          }}
        >
          Unclear
        </button>
      </div>
    </article>
  );
}

function EvalComparisonPanel({ data }: { data: EvalComparisonResponse }) {
  return (
    <section className="eval-comparison panel-surface">
      <div className="panel-heading">
        <span>Eval Model Comparison</span>
        <span>Qwen 3 vs 3.5 vs 3.6 Flash</span>
      </div>

      <div className="comparison-overview">
        {data.models.map((model) => (
          <article className="comparison-card" key={model.model}>
            <h4>{model.display_name}</h4>
            <div className="comparison-meta-grid">
              <span>Exact</span>
              <strong>{formatPercent(model.exact_accuracy)}</strong>
              <span>Component</span>
              <strong>{formatPercent(model.component_accuracy)}</strong>
              <span>Latency</span>
              <strong>{formatMs(model.avg_latency_ms)}</strong>
              <span>Errors</span>
              <strong>{model.errors}</strong>
            </div>
          </article>
        ))}
      </div>

      <div className="comparison-categories">
        {data.categories.map((category) => (
          <CategoryBarRow key={category} category={category} models={data.models} />
        ))}
      </div>
    </section>
  );
}

function CategoryBarRow({ category, models }: { category: string; models: EvalModelSummary[] }) {
  const label = category.replace(/_/g, ' ');

  return (
    <article className="comparison-category-row">
      <h5>{label}</h5>
      <div className="comparison-bars">
        {models.map((model, modelIndex) => {
          const categoryScore = model.categories[category]?.exact_accuracy || 0;
          return (
            <div key={`${model.model}-${category}`} className="comparison-bar-wrap">
              <div className="comparison-bar-header">
                <span>{model.display_name}</span>
                <strong>{formatPercent(categoryScore)}</strong>
              </div>
              <div className="comparison-bar-track">
                <span
                  className={`comparison-bar comparison-model-${modelIndex % 3}`}
                  style={{ width: `${Math.round(categoryScore * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  return (
    <section className="json-panel panel-surface">
      <div className="panel-heading">
        <span>{title}</span>
        <span>oracle</span>
      </div>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}

export default BenchmarkVerifier;

function formatPercent(value: number) {
  return `${Math.max(0, Math.round((value || 0) * 1000) / 10).toFixed(1)}%`;
}

function formatMs(value: number) {
  return `${Math.round(value || 0)}ms`;
}
