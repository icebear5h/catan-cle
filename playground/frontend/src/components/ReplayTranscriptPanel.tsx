import { useEffect, useRef, useState } from 'react';
import type {
  ReplayModelTrace,
  ReplayModelTraceWindow,
  ReplayNarratorReasoningGroup,
  ReplayNarratorReasoningKind,
  ReplayNarratorReasoningWindow,
  ReplayTranscriptWindow,
} from '../types';
import './ReplayTranscriptPanel.css';

interface ReplayTranscriptPanelProps {
  transcript: ReplayTranscriptWindow;
  narratorReasoning?: ReplayNarratorReasoningWindow;
  modelTrace?: ReplayModelTraceWindow;
}

type YouTubeView = 'transcript' | 'reasoning';

function formatTimestamp(seconds: number | null): string {
  if (seconds === null) {
    return '--:--';
  }
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds - minutes * 60;
  return `${minutes}:${remaining.toFixed(1).padStart(4, '0')}`;
}

function formatLatency(milliseconds: number | null): string {
  if (milliseconds === null) {
    return 'unknown latency';
  }
  return `${(milliseconds / 1000).toFixed(1)}s`;
}

function formatColonistColor(color: number | null): string {
  const names: Record<number, string> = {
    1: 'RED',
    2: 'BLUE',
    3: 'ORANGE',
    4: 'GREEN',
    5: 'BLACK',
    9: 'WHITE',
  };
  if (color === null) {
    return 'unknown';
  }
  return `${names[color] || 'UNKNOWN'} / color ${color}`;
}

function alignmentLabel(trace: ReplayModelTrace): string | null {
  if (trace.alignment === 'pre_action_reordered_event') {
    return 'pre-event · reordered source rows';
  }
  if (trace.alignment === 'post_event_reveal') {
    return 'revealed after reordered event';
  }
  return null;
}

function reasoningKindLabel(kind: ReplayNarratorReasoningKind): string {
  return kind.replaceAll('_', ' ');
}

function reasoningGroupLabel(group: ReplayNarratorReasoningGroup): string {
  if (group.anchor_kind === 'decision') {
    return `decision observation · before step ${group.subject_replay_index}`;
  }
  if (group.anchor_kind === 'complete') {
    return 'replay-complete observation';
  }
  return `public observation · before step ${group.subject_replay_index}`;
}

function policyLabel(modelTrace: ReplayModelTraceWindow): string {
  const goals = modelTrace.policy.stateless_goals === true
    ? 'stateless'
    : modelTrace.policy.stateless_goals === false
      ? 'stateful goals'
      : 'goals policy unknown';
  const lookahead = modelTrace.policy.allow_lookahead === false
    ? 'no lookahead'
    : modelTrace.policy.allow_lookahead === true
      ? 'lookahead enabled'
      : 'lookahead unknown';
  const execution = modelTrace.policy.execute_model_actions === false
    ? 'model action not executed'
    : modelTrace.policy.execute_model_actions === true
      ? 'model action executed'
      : 'execution policy unknown';
  return `${goals} · ${lookahead} · ${execution}`;
}

export default function ReplayTranscriptPanel({
  transcript,
  narratorReasoning,
  modelTrace,
}: ReplayTranscriptPanelProps) {
  const [youtubeView, setYouTubeView] = useState<YouTubeView>('transcript');
  const transcriptListRef = useRef<HTMLOListElement>(null);
  const reasoningListRef = useRef<HTMLOListElement>(null);
  const transcriptHistory = transcript.history_segments || transcript.segments;
  const reasoningHistory = narratorReasoning?.history_paragraphs
    || narratorReasoning?.paragraphs
    || [];
  const reasoningGroups = narratorReasoning?.history_groups || [];
  const interval = `${formatTimestamp(transcript.window_start_s)}–${formatTimestamp(transcript.window_end_s)}`;
  const narratorColor = modelTrace?.player.engine_color || `color ${transcript.narrator.colonist_color ?? '?'}`;

  useEffect(() => {
    const activeList = youtubeView === 'transcript'
      ? transcriptListRef.current
      : reasoningListRef.current;
    if (activeList) {
      activeList.scrollTop = activeList.scrollHeight;
    }
  }, [
    narratorReasoning?.replay_index,
    reasoningGroups.length,
    reasoningHistory.length,
    transcript.replay_index,
    transcriptHistory.length,
    youtubeView,
  ]);

  return (
    <section className="replay-transcript-panel" aria-label="Narrator and Qwen replay alignment">
      <div className="replay-transcript-heading">
        <div>
          <h3>{narratorColor} Reasoning Alignment</h3>
          <span className="replay-transcript-step">before replay step {transcript.replay_index}</span>
        </div>
        <span className={`replay-transcript-status ${transcript.verified ? 'verified' : 'provisional'}`}>
          {transcript.verified ? 'verified pair' : 'provisional pair'}
        </span>
      </div>

      <p className="replay-transcript-note">
        Human captions, their board-grounded reconstruction, and same-seat Qwen traces are shown only when their source context is available at this cursor.
      </p>

      <div className="replay-transcript-meta">
        <div>
          <span className="replay-transcript-label">Human narrator</span>
          <strong>{transcript.narrator.username || 'Unknown narrator'} · {narratorColor}</strong>
        </div>
        <div>
          <span className="replay-transcript-label">Transcript interval</span>
          <strong>{interval}</strong>
        </div>
      </div>

      <div className="replay-alignment-section youtube-commentary-section">
        <div className="replay-alignment-section-heading">
          <h4>YouTube commentary</h4>
          <span>human · {narratorColor}</span>
        </div>

        <div
          className="youtube-commentary-tabs"
          role="tablist"
          aria-label="YouTube commentary view"
        >
          <button
            type="button"
            id="youtube-transcript-tab"
            role="tab"
            aria-selected={youtubeView === 'transcript'}
            aria-controls="youtube-transcript-panel"
            className={youtubeView === 'transcript' ? 'active' : ''}
            onClick={() => setYouTubeView('transcript')}
          >
            Transcript
          </button>
          <button
            type="button"
            id="youtube-reasoning-tab"
            role="tab"
            aria-selected={youtubeView === 'reasoning'}
            aria-controls="youtube-reasoning-panel"
            className={youtubeView === 'reasoning' ? 'active' : ''}
            onClick={() => setYouTubeView('reasoning')}
          >
            Reasoning
          </button>
        </div>

        {youtubeView === 'transcript' && (
          <div
            id="youtube-transcript-panel"
            role="tabpanel"
            aria-labelledby="youtube-transcript-tab"
          >
            {transcript.status === 'clock_anomaly' && (
              <div className="replay-transcript-warning" role="status">
                Source replay time moved backward; transcript text is withheld.
              </div>
            )}

            {transcript.status === 'unavailable' && (
              <div className="replay-transcript-empty" role="status">
                This cursor has no trustworthy source timestamp.
              </div>
            )}

            {transcriptHistory.length === 0
              && transcript.status !== 'clock_anomaly'
              && transcript.status !== 'unavailable' && (
              <div className="replay-transcript-empty" role="status">
                No transcript text is causally available at this cursor.
              </div>
            )}

            {transcript.segments.length === 0 && transcriptHistory.length > 0 && (
              <div className="replay-transcript-current-note" role="status">
                {transcript.status === 'complete'
                  ? 'Replay complete—showing the full transcript.'
                  : 'No new captions in this action interval—showing transcript history.'}
              </div>
            )}

            {transcriptHistory.length > 0 && (
              <ol className="replay-transcript-lines" ref={transcriptListRef}>
                {transcriptHistory.map((segment, segmentIndex) => (
                  <li key={`${segment.source_start_index}-${segment.source_end_index}-${segmentIndex}`}>
                    <time>
                      {formatTimestamp(segment.start_s)}–{formatTimestamp(segment.end_s)}
                    </time>
                    <p>{segment.text}</p>
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}

        {youtubeView === 'reasoning' && (
          <div
            id="youtube-reasoning-panel"
            role="tabpanel"
            aria-labelledby="youtube-reasoning-tab"
            className="narrator-reasoning-panel"
          >
            <p className="narrator-reasoning-note">
              {narratorReasoning?.model_label || 'GPT-5.6'} assembles captions across replay-row boundaries into strict causal decision and public-observation packets. Each packet sees only the public board and events available at its displayed cursor. This remains separate from the Qwen policy trace below.
            </p>

            {!narratorReasoning && (
              <div className="replay-transcript-empty" role="status">
                No grounded narrator-reasoning artifact is available for this replay.
              </div>
            )}

            {narratorReasoning?.status === 'artifact_error' && (
              <div className="replay-transcript-warning" role="status">
                Narrator reasoning artifact unavailable: {narratorReasoning.artifact_error}
              </div>
            )}

            {narratorReasoning?.status === 'pending' && (
              <div className="replay-transcript-empty" role="status">
                The grounded reasoning paragraph for this interval is still pending.
              </div>
            )}

            {narratorReasoning?.status === 'error' && (
              <div className="replay-transcript-warning" role="status">
                Could not reconstruct this interval: {narratorReasoning.error?.message || 'unknown generation error'}
              </div>
            )}

            {narratorReasoning?.status === 'empty' && reasoningHistory.length === 0 && (
              <div className="replay-transcript-empty" role="status">
                These captions contain no substantive Catan reasoning after filler and repetition are removed.
              </div>
            )}

            {narratorReasoning?.status === 'no_commentary' && reasoningHistory.length === 0 && (
              <div className="replay-transcript-empty" role="status">
                No narrator reasoning is causally available at this cursor.
              </div>
            )}

            {narratorReasoning
              && ['empty', 'no_commentary', 'complete'].includes(narratorReasoning.status)
              && reasoningHistory.length > 0 && (
              <div className="replay-transcript-current-note" role="status">
                {narratorReasoning.status === 'complete'
                  ? 'Replay complete—showing the full narrator-reasoning history.'
                  : 'No new reasoning in this action interval—showing earlier paragraphs.'}
              </div>
            )}

            {narratorReasoning?.status === 'unavailable' && (
              <div className="replay-transcript-empty" role="status">
                This cursor has no trustworthy narrator-reasoning interval.
              </div>
            )}

            {reasoningGroups.length > 0 && (
              <ol className="narrator-reasoning-paragraphs" ref={reasoningListRef}>
                {reasoningGroups.map((group) => (
                  <li
                    key={`${group.anchor_kind}-${group.subject_replay_index}`}
                    className="narrator-reasoning-group"
                  >
                    <div className="narrator-reasoning-group-heading">
                      <strong>{reasoningGroupLabel(group)}</strong>
                      <span>{group.paragraphs.length} assembled {group.paragraphs.length === 1 ? 'paragraph' : 'paragraphs'}</span>
                    </div>
                    {group.paragraphs.map((paragraph) => (
                      <article className="narrator-reasoning-entry" key={paragraph.paragraph_id}>
                        <div className="narrator-reasoning-entry-meta">
                          <time>
                            {formatTimestamp(paragraph.start_s)}–{formatTimestamp(paragraph.end_s)}
                          </time>
                          <span>{reasoningKindLabel(paragraph.kind)}</span>
                        </div>
                        <p>{paragraph.text}</p>
                        {paragraph.uncertainties.length > 0 && (
                          <details>
                            <summary>Unresolved context</summary>
                            <ul>
                              {paragraph.uncertainties.map((uncertainty) => (
                                <li key={uncertainty}>{uncertainty}</li>
                              ))}
                            </ul>
                          </details>
                        )}
                      </article>
                    ))}
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
      </div>

      {modelTrace && (
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
                  <span>Qwen self-reviewed rationale</span>
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
      )}

      {transcript.video_url && (
        <a
          className="replay-transcript-source"
          href={transcript.video_url}
          target="_blank"
          rel="noreferrer"
        >
          Source video · {transcript.video_id}
        </a>
      )}
    </section>
  );
}
