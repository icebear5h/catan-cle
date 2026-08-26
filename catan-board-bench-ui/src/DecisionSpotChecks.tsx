import { useEffect, useMemo, useState } from 'react';
import './DecisionSpotChecks.css';

const SERVER_URL = 'http://127.0.0.1:5001';

type StageId = 'setup' | 'early' | 'mid' | 'late' | 'unknown';
type AgreementFilter = 'all' | 'agree' | 'disagree' | 'unscored';
type Verdict = 'strong' | 'reasonable' | 'questionable' | 'blunder';

type StageRule = {
  id: Exclude<StageId, 'unknown'>;
  label: string;
  description: string;
  detection: string;
};

type BucketDefinition = {
  id: string;
  label: string;
  group: string;
  description: string;
  review_unit: string;
  target_samples: number;
  detection: string;
  rubric: string[];
};

type ReviewLabel = {
  id: string;
  label: string;
  description: string;
};

type VerdictDefinition = ReviewLabel;

type BucketCatalog = {
  id: string;
  version: string;
  schema: string;
  stage_rules: StageRule[];
  critical_rule: {
    label: string;
    description: string;
    detection: string;
  };
  buckets: BucketDefinition[];
  review_labels: ReviewLabel[];
  verdicts: VerdictDefinition[];
  sampling: {
    default_target_per_bucket: number;
    random_fraction: number;
    reasoning_disagreement_fraction: number;
    high_stakes_fraction: number;
    include_model_agreements: boolean;
  };
};

type DecisionRun = {
  id: string;
  title: string;
  description: string;
  model_id: string;
  game_id: string;
  target_player_id: number | null;
  target_engine_color: string | null;
  available: boolean;
};

type CompactRun = {
  schema: string;
  id: string;
  title: string;
  description: string;
  game_id: string;
  model_id: string;
  target_player_id: number;
  target_engine_color: string;
  decision_count: number;
  bucketed_decision_count: number;
  response_count: number;
  disagreement_count: number;
  bucket_suite: {
    id: string;
    version: string;
    schema: string;
  };
};

type BucketCount = {
  bucket_id: string;
  decision_count: number;
  episode_count: number;
  response_count: number;
  disagreement_count: number;
};

type HumanSelection = {
  action_index: number;
  action: string;
  description: string;
  normalization?: string;
} | null;

type CompactModelSelection = {
  model_id: string;
  response_present: boolean;
  action_index: number | null;
  action: string | null;
  description: string | null;
  agreement: boolean | null;
  error: unknown;
  parse_error: string | null;
};

type CompactDecision = {
  decision_id: string;
  game_id: string;
  replay_index: number;
  source_event_index: number | null;
  action_type: string;
  classification: string;
  forced: boolean;
  stage: StageId;
  critical: boolean;
  bucket_ids: string[];
  episode_ids: Record<string, string>;
  actor: Record<string, unknown>;
  human: HumanSelection;
  model: CompactModelSelection;
};

type DecisionListResponse = {
  run: CompactRun;
  bucket_counts: BucketCount[];
  stage_counts: Record<string, number>;
  total: number;
  offset: number;
  limit: number;
  decisions: CompactDecision[];
};

type DetailedModelSelection = CompactModelSelection & {
  response_source: string | null;
  recorded_at: string | null;
  game_plan: string;
  rationale: string;
  rationale_source: string | null;
  native_reasoning: string;
  native_reasoning_details: unknown[];
  reasoning_request: Record<string, unknown> | null;
  reasoning_tokens: number | null;
  native_reasoning_returned: boolean | null;
  usage: Record<string, unknown>;
  finish_reason: string | null;
  provider_native_finish_reason: string | null;
  provider_response_id: string | null;
  provider_request_id: string | null;
  latency_ms: number | null;
  context_version: string | null;
  context_prompt: string;
  system_prompt: string;
  raw_response: string;
};

type DetailedDecision = Omit<CompactDecision, 'model'> & {
  source_replay_index: number | null;
  reason: string | null;
  phase: string | null;
  bucket_evidence: Record<string, unknown>;
  state_features: Record<string, unknown> | null;
  available_actions: Array<{
    index: number;
    action: string;
    description: string;
  }>;
  model: DetailedModelSelection;
};

type DecisionDetailResponse = {
  run: CompactRun;
  decision: DetailedDecision;
};

function DecisionSpotChecks() {
  const [catalog, setCatalog] = useState<BucketCatalog | null>(null);
  const [runs, setRuns] = useState<DecisionRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [selectedBucketId, setSelectedBucketId] = useState('all');
  const [selectedStage, setSelectedStage] = useState<'all' | StageId>('all');
  const [agreementFilter, setAgreementFilter] = useState<AgreementFilter>('all');
  const [classificationFilter, setClassificationFilter] = useState('all');
  const [listResponse, setListResponse] = useState<DecisionListResponse | null>(null);
  const [selectedDecisionId, setSelectedDecisionId] = useState<string | null>(null);
  const [detail, setDetail] = useState<DetailedDecision | null>(null);
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [reviewLabels, setReviewLabels] = useState<Record<string, string[]>>({});
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(true);
  const [isLoadingList, setIsLoadingList] = useState(false);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const loadCatalog = async () => {
      try {
        setIsLoadingCatalog(true);
        setError(null);
        const [catalogResponse, runsResponse] = await Promise.all([
          fetch(`${SERVER_URL}/api/decision-evals/catalog`, { signal: controller.signal }),
          fetch(`${SERVER_URL}/api/decision-evals/runs`, { signal: controller.signal }),
        ]);
        const catalogPayload: unknown = await catalogResponse.json();
        const runsPayload: unknown = await runsResponse.json();
        if (!catalogResponse.ok) {
          throw new Error(apiError(catalogPayload, 'Failed to load decision bucket catalog'));
        }
        if (!runsResponse.ok) {
          throw new Error(apiError(runsPayload, 'Failed to load decision eval runs'));
        }
        const nextCatalog = catalogPayload as BucketCatalog;
        const nextRuns = (runsPayload as { runs: DecisionRun[] }).runs.filter((run) => run.available);
        setCatalog(nextCatalog);
        setRuns(nextRuns);
        setSelectedRunId((current) => current || nextRuns[0]?.id || '');
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(String(loadError));
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoadingCatalog(false);
        }
      }
    };
    void loadCatalog();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedRunId) {
      return;
    }
    const controller = new AbortController();
    const loadDecisions = async () => {
      try {
        setIsLoadingList(true);
        setError(null);
        const params = new URLSearchParams({
          run_id: selectedRunId,
          bucket: selectedBucketId,
          stage: selectedStage,
          agreement: agreementFilter,
          classification: classificationFilter,
          limit: '500',
        });
        const response = await fetch(
          `${SERVER_URL}/api/decision-evals/decisions?${params}`,
          { signal: controller.signal },
        );
        const payload: unknown = await response.json();
        if (!response.ok) {
          throw new Error(apiError(payload, 'Failed to load bucketed decisions'));
        }
        const next = payload as DecisionListResponse;
        setListResponse(next);
        setSelectedDecisionId((current) => {
          if (current && next.decisions.some((decision) => decision.decision_id === current)) {
            return current;
          }
          return next.decisions[0]?.decision_id || null;
        });
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(String(loadError));
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoadingList(false);
        }
      }
    };
    void loadDecisions();
    return () => controller.abort();
  }, [selectedRunId, selectedBucketId, selectedStage, agreementFilter, classificationFilter]);

  useEffect(() => {
    if (!selectedRunId || !selectedDecisionId) {
      setDetail(null);
      return;
    }
    const controller = new AbortController();
    const loadDetail = async () => {
      try {
        setIsLoadingDetail(true);
        setError(null);
        const params = new URLSearchParams({
          run_id: selectedRunId,
          decision_id: selectedDecisionId,
        });
        const response = await fetch(
          `${SERVER_URL}/api/decision-evals/decision?${params}`,
          { signal: controller.signal },
        );
        const payload: unknown = await response.json();
        if (!response.ok) {
          throw new Error(apiError(payload, 'Failed to load decision detail'));
        }
        setDetail((payload as DecisionDetailResponse).decision);
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(String(loadError));
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoadingDetail(false);
        }
      }
    };
    void loadDetail();
    return () => controller.abort();
  }, [selectedRunId, selectedDecisionId]);

  const bucketCountById = useMemo(() => {
    return new Map((listResponse?.bucket_counts || []).map((count) => [count.bucket_id, count]));
  }, [listResponse]);

  const bucketById = useMemo(() => {
    return new Map((catalog?.buckets || []).map((bucket) => [bucket.id, bucket]));
  }, [catalog]);

  const selectedBucket = selectedBucketId === 'all' ? null : bucketById.get(selectedBucketId) || null;
  const selectedVerdict = selectedDecisionId ? verdicts[selectedDecisionId] : undefined;
  const selectedLabels = selectedDecisionId ? reviewLabels[selectedDecisionId] || [] : [];

  const toggleReviewLabel = (labelId: string) => {
    if (!selectedDecisionId) {
      return;
    }
    setReviewLabels((current) => {
      const existing = current[selectedDecisionId] || [];
      const next = existing.includes(labelId)
        ? existing.filter((item) => item !== labelId)
        : [...existing, labelId];
      return { ...current, [selectedDecisionId]: next };
    });
  };

  return (
    <main className="decision-shell">
      <aside className="decision-rail panel-surface">
        <div className="panel-heading">
          <span>Decision buckets</span>
          <span>v{catalog?.version || '—'}</span>
        </div>

        <label className="field-label" htmlFor="decision-run">Run</label>
        <select
          id="decision-run"
          className="bench-select"
          value={selectedRunId}
          onChange={(event) => setSelectedRunId(event.target.value)}
        >
          {runs.map((run) => (
            <option key={run.id} value={run.id}>{run.title}</option>
          ))}
        </select>

        <div className="decision-filter-grid">
          <label>
            <span>Stage</span>
            <select value={selectedStage} onChange={(event) => setSelectedStage(event.target.value as 'all' | StageId)}>
              <option value="all">all stages</option>
              {catalog?.stage_rules.map((stage) => (
                <option key={stage.id} value={stage.id}>{stage.label}</option>
              ))}
              <option value="unknown">unknown</option>
            </select>
          </label>
          <label>
            <span>Result</span>
            <select value={agreementFilter} onChange={(event) => setAgreementFilter(event.target.value as AgreementFilter)}>
              <option value="all">all results</option>
              <option value="disagree">model differs</option>
              <option value="agree">model agrees</option>
              <option value="unscored">no model call</option>
            </select>
          </label>
          <label>
            <span>Mapping</span>
            <select value={classificationFilter} onChange={(event) => setClassificationFilter(event.target.value)}>
              <option value="all">all mappings</option>
              <option value="exact">exact</option>
              <option value="coarse">coarse</option>
              <option value="unmappable">unmappable</option>
              <option value="lifecycle">lifecycle</option>
            </select>
          </label>
        </div>

        <div className="bucket-list" aria-label="Decision spot-check buckets">
          <button
            className={`bucket-row ${selectedBucketId === 'all' ? 'active' : ''}`}
            onClick={() => setSelectedBucketId('all')}
          >
            <span><strong>All decisions</strong><small>Every tagged and untagged row</small></span>
            <code>{listResponse?.run.decision_count || 0}</code>
          </button>
          {catalog?.buckets.map((bucket) => {
            const count = bucketCountById.get(bucket.id);
            return (
              <button
                key={bucket.id}
                className={`bucket-row ${selectedBucketId === bucket.id ? 'active' : ''} ${!count?.decision_count ? 'empty' : ''}`}
                onClick={() => setSelectedBucketId(bucket.id)}
              >
                <span>
                  <strong>{bucket.label}</strong>
                  <small>{bucket.group} · {humanize(bucket.review_unit)}</small>
                </span>
                <code>{count?.episode_count || 0}/{count?.decision_count || 0}</code>
              </button>
            );
          })}
        </div>
      </aside>

      <section className="decision-main">
        {(isLoadingCatalog || isLoadingList || isLoadingDetail) && (
          <div className="bench-loading">loading decision eval</div>
        )}
        {error && <div className="bench-error">{error}</div>}

        {listResponse && (
          <DecisionRunSummary run={listResponse.run} filteredCount={listResponse.total} />
        )}

        {selectedBucket && (
          <BucketDefinitionPanel bucket={selectedBucket} count={bucketCountById.get(selectedBucket.id)} />
        )}

        <div className="decision-review-grid">
          <section className="decision-list-panel panel-surface">
            <div className="panel-heading">
              <span>Spot-check queue</span>
              <span>{listResponse?.total || 0} decisions</span>
            </div>
            <div className="decision-card-list">
              {listResponse?.decisions.map((decision) => (
                <DecisionRow
                  key={decision.decision_id}
                  decision={decision}
                  bucketById={bucketById}
                  selected={decision.decision_id === selectedDecisionId}
                  verdict={verdicts[decision.decision_id]}
                  onSelect={() => setSelectedDecisionId(decision.decision_id)}
                />
              ))}
              {listResponse && listResponse.decisions.length === 0 && (
                <div className="empty-queue">No decisions match these filters.</div>
              )}
            </div>
          </section>

          <section className="decision-detail-column">
            {detail ? (
              <DecisionDetail
                decision={detail}
                catalog={catalog}
                bucketById={bucketById}
                verdict={selectedVerdict}
                selectedLabels={selectedLabels}
                onVerdict={(verdict) => {
                  if (selectedDecisionId) {
                    setVerdicts((current) => ({ ...current, [selectedDecisionId]: verdict }));
                  }
                }}
                onToggleLabel={toggleReviewLabel}
              />
            ) : (
              <div className="panel-surface empty-detail">Select a decision to inspect it.</div>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

function DecisionRunSummary({ run, filteredCount }: { run: CompactRun; filteredCount: number }) {
  return (
    <section className="decision-run-summary panel-surface">
      <div className="panel-heading">
        <span>{run.title}</span>
        <span>{run.bucket_suite.id}@{run.bucket_suite.version}</span>
      </div>
      <div className="decision-stat-grid">
        <Stat label="Filtered" value={filteredCount} />
        <Stat label="All rows" value={run.decision_count} />
        <Stat label="Bucketed" value={run.bucketed_decision_count} />
        <Stat label="Model calls" value={run.response_count} />
        <Stat label="Differences" value={run.disagreement_count} />
        <Stat label="Replay" value={run.game_id} />
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="decision-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function BucketDefinitionPanel({ bucket, count }: { bucket: BucketDefinition; count?: BucketCount }) {
  return (
    <section className="bucket-definition panel-surface">
      <div className="bucket-definition-heading">
        <div>
          <span className="bucket-group">{bucket.group}</span>
          <h2>{bucket.label}</h2>
        </div>
        <div className="bucket-target">
          <strong>{count?.episode_count || 0}</strong>
          <span>episodes · target {bucket.target_samples}</span>
        </div>
      </div>
      <p>{bucket.description}</p>
      <div className="bucket-definition-grid">
        <div><span>Detection</span><p>{bucket.detection}</p></div>
        <div><span>Review unit</span><p>{humanize(bucket.review_unit)}</p></div>
      </div>
      <ol className="bucket-rubric">
        {bucket.rubric.map((item) => <li key={item}>{item}</li>)}
      </ol>
    </section>
  );
}

function DecisionRow({
  decision,
  bucketById,
  selected,
  verdict,
  onSelect,
}: {
  decision: CompactDecision;
  bucketById: Map<string, BucketDefinition>;
  selected: boolean;
  verdict?: Verdict;
  onSelect: () => void;
}) {
  return (
    <button className={`decision-eval-row ${selected ? 'active' : ''}`} onClick={onSelect}>
      <div className="decision-row-top">
        <span className={`stage-chip stage-${decision.stage}`}>{decision.stage}</span>
        {decision.critical && <span className="critical-chip">critical</span>}
        <code>#{decision.replay_index}</code>
        {verdict && <span className={`verdict-dot verdict-${verdict}`}>{verdict}</span>}
      </div>
      <strong>{humanize(decision.action_type)}</strong>
      <p>{decision.model.description || decision.human?.description || decision.classification}</p>
      <div className="decision-row-tags">
        {decision.bucket_ids.slice(0, 3).map((bucketId) => (
          <span key={bucketId}>{bucketById.get(bucketId)?.label || humanize(bucketId)}</span>
        ))}
        {decision.bucket_ids.length > 3 && <span>+{decision.bucket_ids.length - 3}</span>}
      </div>
      <AgreementBadge model={decision.model} />
    </button>
  );
}

function AgreementBadge({ model }: { model: CompactModelSelection }) {
  if (!model.response_present) {
    return <span className="agreement-badge unscored">unscored</span>;
  }
  if (model.error) {
    return <span className="agreement-badge error">error</span>;
  }
  if (model.agreement) {
    return <span className="agreement-badge agree">model agrees</span>;
  }
  return <span className="agreement-badge disagree">model differs</span>;
}

function DecisionDetail({
  decision,
  catalog,
  bucketById,
  verdict,
  selectedLabels,
  onVerdict,
  onToggleLabel,
}: {
  decision: DetailedDecision;
  catalog: BucketCatalog | null;
  bucketById: Map<string, BucketDefinition>;
  verdict?: Verdict;
  selectedLabels: string[];
  onVerdict: (verdict: Verdict) => void;
  onToggleLabel: (labelId: string) => void;
}) {
  const humanIndex = decision.human?.action_index;
  const modelIndex = decision.model.action_index;

  return (
    <>
      <section className="decision-detail panel-surface">
        <div className="panel-heading">
          <span>{humanize(decision.action_type)}</span>
          <span>{decision.decision_id}</span>
        </div>
        <div className="detail-chip-row">
          <span className={`stage-chip stage-${decision.stage}`}>{decision.stage}</span>
          <span>{decision.classification}</span>
          <span>{decision.forced ? 'forced' : 'choice'}</span>
          {decision.critical && <span className="critical-chip">critical pressure</span>}
        </div>
        <div className="detail-buckets">
          {decision.bucket_ids.map((bucketId) => (
            <span key={bucketId} title={decision.episode_ids[bucketId]}>
              {bucketById.get(bucketId)?.label || humanize(bucketId)}
            </span>
          ))}
        </div>

        <div className="selection-compare">
          <SelectionCard title="Recorded human" selection={decision.human} tone="human" />
          <SelectionCard
            title={decision.model.model_id}
            selection={decision.model.response_present ? {
              action_index: decision.model.action_index ?? -1,
              action: decision.model.action || '',
              description: decision.model.description || 'No parsed model selection',
            } : null}
            tone={decision.model.agreement ? 'agree' : 'model'}
          />
        </div>

        {decision.model.parse_error && (
          <div className="detail-warning">Parse warning: {decision.model.parse_error}</div>
        )}
        {decision.model.error != null && (
          <div className="detail-warning">Model error: {JSON.stringify(decision.model.error)}</div>
        )}
      </section>

      <section className="review-controls panel-surface">
        <div className="panel-heading">
          <span>Human spot check</span>
          <span>local session</span>
        </div>
        <div className="quality-verdicts">
          {catalog?.verdicts.map((item) => (
            <button
              key={item.id}
              className={`quality-button quality-${item.id} ${verdict === item.id ? 'active' : ''}`}
              onClick={() => onVerdict(item.id as Verdict)}
              title={item.description}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="failure-labels">
          {catalog?.review_labels.map((label) => (
            <button
              key={label.id}
              className={selectedLabels.includes(label.id) ? 'active' : ''}
              onClick={() => onToggleLabel(label.id)}
              title={label.description}
            >
              {label.label}
            </button>
          ))}
        </div>
      </section>

      <TextBlock
        title="Visible rationale"
        metadata={decision.model.rationale_source === 'legacy_reasoning_field' ? 'legacy visible field' : 'fact-checkable'}
        text={decision.model.rationale}
        empty="No visible rationale was recorded."
      />
      <TextBlock
        title="Native provider reasoning"
        metadata={nativeReasoningMetadata(decision.model)}
        text={decision.model.native_reasoning}
        empty="No native reasoning was recorded in this archived run."
      />
      {decision.model.game_plan && (
        <TextBlock title="Game plan" metadata="strategic memory" text={decision.model.game_plan} />
      )}

      <section className="legal-actions panel-surface">
        <div className="panel-heading">
          <span>Legal action menu</span>
          <span>{decision.available_actions.length} actions</span>
        </div>
        <div className="legal-action-list">
          {decision.available_actions.map((action) => (
            <div
              key={`${action.index}-${action.action}`}
              className={`legal-action ${action.index === humanIndex ? 'human-selected' : ''} ${action.index === modelIndex ? 'model-selected' : ''}`}
            >
              <code>{action.index}</code>
              <span>{action.description}</span>
              <small>
                {action.index === humanIndex ? 'human' : ''}
                {action.index === humanIndex && action.index === modelIndex ? ' + ' : ''}
                {action.index === modelIndex ? 'model' : ''}
              </small>
            </div>
          ))}
        </div>
      </section>

      <details className="evidence-details panel-surface">
        <summary>Bucket evidence and state features</summary>
        <pre>{JSON.stringify({ bucket_evidence: decision.bucket_evidence, state_features: decision.state_features }, null, 2)}</pre>
      </details>
      <details className="evidence-details panel-surface">
        <summary>Model context prompt</summary>
        <pre>{decision.model.context_prompt || 'No context prompt was retained.'}</pre>
      </details>
      <details className="evidence-details panel-surface">
        <summary>Generation metadata</summary>
        <pre>{JSON.stringify({
          context_version: decision.model.context_version,
          response_source: decision.model.response_source,
          recorded_at: decision.model.recorded_at,
          finish_reason: decision.model.finish_reason,
          provider_native_finish_reason: decision.model.provider_native_finish_reason,
          provider_response_id: decision.model.provider_response_id,
          provider_request_id: decision.model.provider_request_id,
          latency_ms: decision.model.latency_ms,
          reasoning_request: decision.model.reasoning_request,
          reasoning_tokens: decision.model.reasoning_tokens,
          native_reasoning_returned: decision.model.native_reasoning_returned,
          usage: decision.model.usage,
        }, null, 2)}</pre>
      </details>
    </>
  );
}

function SelectionCard({
  title,
  selection,
  tone,
}: {
  title: string;
  selection: HumanSelection;
  tone: 'human' | 'model' | 'agree';
}) {
  return (
    <article className={`selection-card selection-${tone}`}>
      <span>{title}</span>
      {selection ? (
        <>
          <strong>#{selection.action_index}</strong>
          <p>{selection.description}</p>
        </>
      ) : (
        <p>No exact indexed selection.</p>
      )}
    </article>
  );
}

function TextBlock({
  title,
  metadata,
  text,
  empty = 'No text was recorded.',
}: {
  title: string;
  metadata: string;
  text: string;
  empty?: string;
}) {
  return (
    <section className="eval-text-block panel-surface">
      <div className="panel-heading"><span>{title}</span><span>{metadata}</span></div>
      <pre className={!text ? 'empty' : ''}>{text || empty}</pre>
    </section>
  );
}

function nativeReasoningMetadata(model: DetailedModelSelection) {
  const parts: string[] = [];
  if (model.reasoning_tokens != null) {
    parts.push(`${model.reasoning_tokens.toLocaleString()} tokens`);
  }
  if (model.reasoning_request) {
    parts.push(JSON.stringify(model.reasoning_request));
  }
  return parts.join(' · ') || 'separate provider channel';
}

function apiError(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object' && 'error' in payload) {
    return String((payload as { error: unknown }).error);
  }
  return fallback;
}

function humanize(value: string) {
  return value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default DecisionSpotChecks;
