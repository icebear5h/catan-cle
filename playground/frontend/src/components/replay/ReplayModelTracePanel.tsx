import type { ReplayModelTraceWindow } from '../../types';
import {
  alignmentLabel,
  formatColonistColor,
  formatLatency,
  policyLabel,
} from './replayTraceFormat';

export default function ReplayModelTracePanel({ modelTrace }: { modelTrace: ReplayModelTraceWindow }) {
  return (
    <div className="replay-alignment-section qwen-trace-section">
      <div className="replay-alignment-section-heading">
        <h4>{modelTrace.model_label} trace</h4>
        <span>model · {modelTrace.player.engine_color}</span>
      </div>

      {!modelTrace.state_provenance.target_matches_archive_perspective && (
        <p className="qwen-trace-provenance">
          The archive is {formatColonistColor(modelTrace.state_provenance.archived_player_perspective)} perspective; {modelTrace.player.engine_color} hidden state is reconstructed and provisional.
        </p>
      )}

      {modelTrace.status === 'artifact_error' && (
        <div className="replay-transcript-warning" role="status">
          Qwen trace artifact unavailable: {modelTrace.artifact_error}
        </div>
      )}

      {modelTrace.status === 'no_decision' && (
        <div className="replay-transcript-empty" role="status">
          No Qwen trace is causally available at this replay row.
        </div>
      )}

      {modelTrace.status === 'pending' && (
        <div className="replay-transcript-empty" role="status">
          This {modelTrace.player.engine_color} decision has no completed Qwen trace yet.
        </div>
      )}

      {modelTrace.status === 'complete' && (
        <div className="replay-transcript-empty" role="status">
          Replay complete—there is no additional Qwen trace.
        </div>
      )}

      {modelTrace.status === 'unavailable' && (
        <div className="replay-transcript-empty" role="status">
          No Qwen trace is available at this cursor.
        </div>
      )}

      {modelTrace.pending_trace_count > 0 && modelTrace.traces.length > 0 && (
        <div className="replay-transcript-empty" role="status">
          {modelTrace.pending_trace_count} additional trace is still pending at this cursor.
        </div>
      )}

      {modelTrace.traces.map((trace) => (
        <div className="qwen-trace-card" key={trace.decision_id}>
          {alignmentLabel(trace) && (
            <div className="qwen-trace-alignment">{alignmentLabel(trace)}</div>
          )}

          {(trace.parse_warning || trace.response_truncated || trace.error) && (
            <div className="replay-transcript-warning" role="status">
              {trace.error?.message
                || trace.parse_warning
                || 'The model response reached its completion limit.'}
            </div>
          )}

          <div className="qwen-trace-meta">
            <span>{trace.phase || 'unknown phase'}</span>
            {trace.setup_strategy_version && (
              <span>{trace.setup_strategy_version}</span>
            )}
            {trace.setup_stage && (
              <span>{trace.setup_stage.replaceAll('_', ' ')}</span>
            )}
            {trace.trace_source === 'setup_strategy_override' && (
              <span>strategic setup refresh</span>
            )}
            {trace.reasoning_source === 'qwen_self_review' && (
              <span>Qwen self-review</span>
            )}
            <span>{trace.forced ? 'forced choice' : `${trace.legal_action_count} legal actions`}</span>
            <span>{formatLatency(trace.latency_ms)}</span>
          </div>

          {trace.quality_warnings.length > 0 && (
            <div className="qwen-quality-warning" role="status">
              <strong>Model-draft caveats</strong>
              <ul>
                {trace.quality_warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="qwen-selected-action">
            <span className="replay-transcript-label">Selected move</span>
            <strong>
              {trace.selection.index === null ? '—' : `${trace.selection.index}. `}
              {trace.selection.description || 'No parseable action was returned.'}
            </strong>
          </div>

          {trace.message && (
            <blockquote className="qwen-trace-message">
              <strong>{trace.player.engine_color}:</strong> {trace.message}
            </blockquote>
          )}

          <details className="qwen-trace-details">
            <summary>
              {trace.reasoning_source === 'qwen_self_review'
                ? 'Qwen self-reviewed goals'
                : 'Qwen goals'}
            </summary>
            <p>{trace.goals || 'No goals returned.'}</p>
          </details>

          <details className="qwen-trace-details">
            <summary>
              {trace.reasoning_source === 'qwen_self_review'
                ? 'Qwen self-reviewed reasoning'
                : 'Qwen reasoning'}
            </summary>
            <p>{trace.reasoning || 'No reasoning returned.'}</p>
          </details>

          {trace.draft_reasoning && (
            <details className="qwen-trace-details">
              <summary>Original Qwen action-response draft</summary>
              {trace.draft_goals && <p>{trace.draft_goals}</p>}
              <p>{trace.draft_reasoning}</p>
            </details>
          )}
        </div>
      ))}

      <p className="qwen-trace-policy">
        {policyLabel(modelTrace)}
      </p>
    </div>
  );
}
