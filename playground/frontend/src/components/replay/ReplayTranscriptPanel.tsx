import { useEffect, useRef, useState } from 'react';
import type {
  ReplayModelTraceWindow,
  ReplayNarratorReasoningWindow,
  ReplayTranscriptWindow,
} from '../../types';
import ReplayModelTracePanel from './ReplayModelTracePanel';
import { formatTimestamp, reasoningGroupLabel, reasoningKindLabel } from './replayTraceFormat';
import './ReplayTranscriptPanel.css';

interface ReplayTranscriptPanelProps {
  transcript: ReplayTranscriptWindow;
  narratorReasoning?: ReplayNarratorReasoningWindow;
  modelTrace?: ReplayModelTraceWindow;
}

type YouTubeView = 'transcript' | 'reasoning';

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

      {modelTrace && <ReplayModelTracePanel modelTrace={modelTrace} />}

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
