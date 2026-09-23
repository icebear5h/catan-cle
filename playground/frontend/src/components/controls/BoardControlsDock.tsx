import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import type { AutoPlayRetryNotice } from '../../autoPlay';
import type { ReplayInfo } from '../../types';
import ReplayPlaybackActions from './ReplayPlaybackActions';
import './BoardControlsDock.css';

interface BoardControlsDockProps {
  children?: ReactNode;
  hasGame: boolean;
  isRunning: boolean;
  replayMode: boolean;
  replayInfo: ReplayInfo | null;
  busy: boolean;
  historyLoading?: boolean;
  error: string | null;
  isTraceBrowsing: boolean;
  liveActor: string | null;
  liveActorIsAgent: boolean;
  liveModel: string | null;
  liveReasoningEffort: string | null;
  liveMaxTokens: number | null;
  isAutoPlaying: boolean;
  autoPlayRetry?: AutoPlayRetryNotice | null;
  onStep: () => void;
  onToggleAutoPlay: () => void;
  onPrevious: () => void;
  onSeek: (step: number) => void;
}

interface BusyElapsedProps {
  label: string;
}

function BusyElapsed({ label }: BusyElapsedProps) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    const startedAt = performance.now();
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((performance.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <>
      {label}<span className="playback-elapsed" aria-hidden="true"> · {elapsedSeconds}s</span>
    </>
  );
}

function AutoPlayRetryCountdown({ attempt, resumeAt }: AutoPlayRetryNotice) {
  const [now, setNow] = useState(() => Date.now());

  // Remounted by the parent's key for each new deadline, so the initial clock is fresh.
  useEffect(() => {
    if (resumeAt === null) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, [resumeAt]);

  const remaining = resumeAt === null ? 0 : Math.max(0, Math.ceil((resumeAt - now) / 1000));
  return (
    <div className="playback-retry" role="status">
      {`Auto-play retry ${attempt}`}
      {resumeAt === null || remaining === 0 ? ' · asking again' : ` · next request in ${remaining}s`}
    </div>
  );
}

export default function BoardControlsDock({
  children,
  hasGame,
  isRunning,
  replayMode,
  replayInfo,
  busy,
  error,
  historyLoading = false,
  isTraceBrowsing,
  liveActor,
  liveActorIsAgent,
  liveModel,
  liveReasoningEffort,
  liveMaxTokens,
  isAutoPlaying,
  autoPlayRetry = null,
  onStep,
  onToggleAutoPlay,
  onPrevious,
  onSeek,
}: BoardControlsDockProps) {
  if (!hasGame) {
    return null;
  }

  const replayIndex = replayInfo?.event_index ?? 0;
  const replayTotal = replayInfo?.total_events ?? 0;
  const liveStatus = isTraceBrowsing
    ? 'History · browse only'
    : isAutoPlaying
      ? autoPlayRetry ? 'Auto-play retrying' : 'Auto-play running'
      : isRunning ? 'Live · ready' : 'Game stopped';
  const liveBusyLabel = historyLoading ? 'Loading checkpoint' : liveActorIsAgent
    ? liveActor ? `Waiting for ${liveActor}` : 'Waiting for model'
    : 'Advancing sandbox';
  const modelLabel = liveActorIsAgent ? [
    liveModel,
    liveReasoningEffort ? `${liveReasoningEffort} reasoning` : null,
    liveMaxTokens === null ? 'uncapped' : `${liveMaxTokens} max tokens`,
  ].filter(Boolean).join(' · ') : null;
  const replayStatus = replayIndex >= replayTotal && replayTotal > 0
    ? 'Replay complete'
    : 'Replay playback';

  return (
    <section
      className={`board-controls-dock ${replayMode ? 'replay' : 'live'}`}
      aria-label="Gameplay controls"
    >
      <div className="playback-bar">
        <div className="playback-status" aria-live="polite">
          <span
            className={`playback-status-dot ${isRunning ? 'running' : ''}`}
            aria-hidden="true"
          />
          <span>
            <strong>
              {replayMode
                ? `${replayStatus} · ${replayIndex}/${replayTotal}`
                : busy
                  ? <BusyElapsed label={liveBusyLabel} />
                  : liveStatus}
            </strong>
            {!replayMode && modelLabel && <small title={modelLabel}>{modelLabel}</small>}
          </span>
        </div>

        {replayMode ? (
          <ReplayPlaybackActions
            key={`${replayInfo?.game_id ?? ''}:${replayIndex}:${replayTotal}`}
            replayIndex={replayIndex}
            replayTotal={replayTotal}
            isRunning={isRunning}
            busy={busy}
            onStep={onStep}
            onPrevious={onPrevious}
            onSeek={onSeek}
          />
        ) : (
          <div className="playback-actions live-actions">
            <button
              type="button"
              className={`playback-button auto-play ${isAutoPlaying ? 'active' : ''}`}
              onClick={onToggleAutoPlay}
              disabled={
                !isAutoPlaying
                && (busy || !isRunning || isTraceBrowsing)
              }
              aria-pressed={isAutoPlaying}
              aria-label={isAutoPlaying
                ? 'Stop auto-play after the current step'
                : 'Auto-play live game'}
              title={isAutoPlaying
                ? 'Stop after the current step or retry wait finishes'
                : 'Run serial live steps until the game ends, retrying failed steps with backoff'}
            >
              {isAutoPlaying ? 'Stop auto' : 'Auto-play'}
            </button>
            <button
              type="button"
              className="playback-button primary"
              onClick={onStep}
              disabled={busy || isAutoPlaying || !isRunning || isTraceBrowsing}
            >
              Step
            </button>
          </div>
        )}

        {!replayMode && children && <div className="live-history">{children}</div>}
        {isAutoPlaying && autoPlayRetry && (
          <AutoPlayRetryCountdown
            key={`${autoPlayRetry.attempt}:${autoPlayRetry.resumeAt ?? 'in-flight'}`}
            attempt={autoPlayRetry.attempt}
            resumeAt={autoPlayRetry.resumeAt}
          />
        )}
        {error && (
          <div className="playback-error" role="alert">
            {error}
          </div>
        )}
      </div>
    </section>
  );
}
