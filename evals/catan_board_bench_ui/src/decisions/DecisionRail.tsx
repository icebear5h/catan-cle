import type {
  AgreementFilter, BucketCatalog, BucketCount, DecisionListResponse, DecisionRun, StageId,
} from './types';
import { humanize } from './decisionFormat';

export default function DecisionRail({
  catalog,
  runs,
  selectedRunId,
  setSelectedRunId,
  selectedStage,
  setSelectedStage,
  agreementFilter,
  setAgreementFilter,
  classificationFilter,
  setClassificationFilter,
  selectedBucketId,
  setSelectedBucketId,
  listResponse,
  bucketCountById,
}: {
  catalog: BucketCatalog | null;
  runs: DecisionRun[];
  selectedRunId: string;
  setSelectedRunId: (runId: string) => void;
  selectedStage: 'all' | StageId;
  setSelectedStage: (stage: 'all' | StageId) => void;
  agreementFilter: AgreementFilter;
  setAgreementFilter: (filter: AgreementFilter) => void;
  classificationFilter: string;
  setClassificationFilter: (filter: string) => void;
  selectedBucketId: string;
  setSelectedBucketId: (bucketId: string) => void;
  listResponse: DecisionListResponse | null;
  bucketCountById: Map<string, BucketCount>;
}) {
  return (
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
  );
}
