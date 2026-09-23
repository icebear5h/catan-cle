import type {
  BucketCatalog, BucketDefinition, DetailedDecision, HumanSelection, Verdict,
} from './types';
import { humanize, nativeReasoningMetadata } from './decisionFormat';

export default function DecisionDetail({
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
