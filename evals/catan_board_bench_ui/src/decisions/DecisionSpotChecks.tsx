import { useEffect, useMemo, useState } from 'react';
import './DecisionSpotChecks.css';
import type {
  AgreementFilter, BucketCatalog, DecisionDetailResponse, DecisionListResponse, DecisionRun,
  DetailedDecision, StageId, Verdict,
} from './types';
import { apiError } from './decisionFormat';
import DecisionRail from './DecisionRail';
import { BucketDefinitionPanel, DecisionRow, DecisionRunSummary } from './DecisionPanels';
import DecisionDetail from './DecisionDetail';

const SERVER_URL = 'http://127.0.0.1:5001';

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
      <DecisionRail
        catalog={catalog}
        runs={runs}
        selectedRunId={selectedRunId}
        setSelectedRunId={setSelectedRunId}
        selectedStage={selectedStage}
        setSelectedStage={setSelectedStage}
        agreementFilter={agreementFilter}
        setAgreementFilter={setAgreementFilter}
        classificationFilter={classificationFilter}
        setClassificationFilter={setClassificationFilter}
        selectedBucketId={selectedBucketId}
        setSelectedBucketId={setSelectedBucketId}
        listResponse={listResponse}
        bucketCountById={bucketCountById}
      />

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

export default DecisionSpotChecks;
