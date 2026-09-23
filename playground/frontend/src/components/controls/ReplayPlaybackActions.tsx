import { useState } from 'react';

interface ReplayPlaybackActionsProps {
  replayIndex: number;
  replayTotal: number;
  isRunning: boolean;
  busy: boolean;
  onStep: () => void;
  onPrevious: () => void;
  onSeek: (step: number) => void;
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

export default function ReplayPlaybackActions({
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
