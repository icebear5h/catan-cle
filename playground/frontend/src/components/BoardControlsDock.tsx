import { useEffect, useState } from 'react';
import type { ReplayInfo } from '../types';
import './BoardControlsDock.css';

interface BoardControlsDockProps {
  hasGame: boolean;
  isRunning: boolean;
  replayMode: boolean;
  replayInfo: ReplayInfo | null;
  busy: boolean;
  error: string | null;
  isTraceBrowsing: boolean;
  liveActor: string | null;
  liveActorIsAgent: boolean;
  liveModel: string | null;
  liveReasoningEffort: string | null;
  liveMaxTokens: number | null;
  isAutoPlaying: boolean;
  onStep: () => void;
  onToggleAutoPlay: () => void;
  onPrevious: () => void;
  onSeek: (step: number) => void;
}

interface ReplayPlaybackActionsProps {
  replayIndex: number;
  replayTotal: number;
  isRunning: boolean;
  busy: boolean;
  onStep: () => void;
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
      <span className="visually-hidden">{label}</span>
      <span aria-hidden="true">{label} · {elapsedSeconds}s</span>
    </>
  );
}

const RANGE_COMMIT_KEYS = new Set([
  'ArrowLeft',
  'ArrowRight',
  'ArrowDown',
  'ArrowUp',
  'Home',
  'End',
  'PageDown',
  'PageUp',
]);

function ReplayPlaybackActions({
  replayIndex,
  replayTotal,
  isRunning,
  busy,
  onStep,
  onPrevious,
  onSeek,
}: ReplayPlaybackActionsProps) {
  const [targetStep, setTargetStep] = useState(replayIndex);
  const clampTarget = (value: number) => (
    Math.min(replayTotal, Math.max(0, Math.trunc(value)))
  );
  const commitSeek = (value: number) => {
    const nextStep = clampTarget(value);
    setTargetStep(nextStep);
    if (!busy && nextStep !== replayIndex) {
      onSeek(nextStep);
    }
  };

  return (
    <form
      className="playback-actions replay-actions"
      onSubmit={(event) => {
        event.preventDefault();
        commitSeek(targetStep);
      }}
    >
      <button
        type="button"
        className="playback-button"
        onClick={onPrevious}
        disabled={busy || replayIndex <= 0}
        aria-label="Previous replay step"
      >
        Previous
      </button>

      <label className="playback-timeline">
        <span className="visually-hidden">Replay position from 0 to {replayTotal}</span>
        <input
          type="range"
          min={0}
          max={replayTotal}
          step={1}
          value={targetStep}
          onChange={(event) => setTargetStep(clampTarget(Number(event.target.value)))}
          onPointerUp={(event) => commitSeek(Number(event.currentTarget.value))}
          onKeyUp={(event) => {
            if (RANGE_COMMIT_KEYS.has(event.key)) {
              commitSeek(Number(event.currentTarget.value));
            }
          }}
          disabled={busy || replayTotal === 0}
          aria-label="Replay timeline"
          aria-valuetext={`Step ${targetStep} of ${replayTotal}`}
        />
      </label>

      <div className="playback-seek">
        <input
          type="number"
          min={0}
          max={replayTotal}
          step={1}
          value={targetStep}
          onChange={(event) => {
            const value = event.target.valueAsNumber;
            if (Number.isFinite(value)) {
              setTargetStep(clampTarget(value));
            }
          }}
          disabled={busy || replayTotal === 0}
          aria-label="Replay step"
        />
        <span aria-hidden="true">/ {replayTotal}</span>
        <button
          type="submit"
          className="playback-seek-button"
          disabled={busy || replayTotal === 0 || targetStep === replayIndex}
        >
          Go
        </button>
      </div>

      <button
        type="button"
        className="playback-button primary"
        onClick={onStep}
        disabled={busy || !isRunning || replayIndex >= replayTotal}
        aria-label="Step replay forward"
      >
        {busy ? 'Working…' : 'Step'}
      </button>
    </form>
  );
}

export default function BoardControlsDock({
  hasGame,
  isRunning,
  replayMode,
  replayInfo,
  busy,
  error,
  isTraceBrowsing,
  liveActor,
  liveActorIsAgent,
  liveModel,
  liveReasoningEffort,
  liveMaxTokens,
  isAutoPlaying,
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
    ? 'Browsing a saved checkpoint'
    : isAutoPlaying
      ? 'Auto-play running'
      : isRunning ? 'Ready for next step' : 'Game stopped';
  const liveBusyLabel = liveActorIsAgent
    ? [
        liveActor ? `Waiting for ${liveActor}` : 'Waiting for model',
        liveModel,
        liveReasoningEffort ? `${liveReasoningEffort} reasoning` : null,
        liveMaxTokens === null ? 'uncapped' : `${liveMaxTokens} max tokens`,
      ].filter(Boolean).join(' · ')
    : 'Advancing sandbox';
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
            <strong>{replayMode ? replayStatus : 'Live game'}</strong>
            <small title={!replayMode && busy ? liveBusyLabel : undefined}>
              {replayMode
                ? `${replayIndex} of ${replayTotal}`
                : busy
                  ? <BusyElapsed label={liveBusyLabel} />
                  : liveStatus}
            </small>
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
                ? 'Stop after the current step finishes'
                : 'Run serial live steps until the game ends or an error occurs'}
            >
              {isAutoPlaying ? 'Stop auto' : 'Auto-play'}
            </button>
            <button
              type="button"
              className="playback-button primary"
              onClick={onStep}
              disabled={busy || isAutoPlaying || !isRunning || isTraceBrowsing}
            >
              {busy
                ? liveActorIsAgent ? 'Waiting for model…' : 'Stepping sandbox…'
                : 'Step'}
            </button>
          </div>
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
