import type {
  BucketCount, BucketDefinition, CompactDecision, CompactModelSelection, CompactRun, Verdict,
} from './types';
import { humanize } from './decisionFormat';

export function DecisionRunSummary({ run, filteredCount }: { run: CompactRun; filteredCount: number }) {
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

export function BucketDefinitionPanel({ bucket, count }: { bucket: BucketDefinition; count?: BucketCount }) {
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

export function DecisionRow({
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
