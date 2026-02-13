import { useState } from 'react';
import './GameControls.css';

interface GameControlsProps {
  onStartGame: (useLlm: boolean) => void;
  onStep: () => void;
  onAutoPlay: (delay: number) => void;
  onStopAutoPlay: () => void;
  onReset: () => void;
  onLoadReplay: (gameId: string) => void;
  onReplayStep: () => void;
  onReplayUndo: () => void;
  onSetReplayStep?: (step: number) => void;
  onRunUntilDrift?: () => void;
  isRunning: boolean;
  hasGame: boolean;
  isLlmProcessing?: boolean;
  isCurrentPlayerLlm?: boolean;
  replayMode?: boolean;
  replayProgress?: string;
}

export default function GameControls({
  onStartGame,
  onStep,
  onAutoPlay,
  onStopAutoPlay,
  onReset,
  onLoadReplay,
  onReplayStep,
  onReplayUndo,
  onSetReplayStep,
  onRunUntilDrift,
  isRunning,
  hasGame,
  isLlmProcessing = false,
  isCurrentPlayerLlm = false,
  replayMode = false,
  replayProgress = '',
}: GameControlsProps) {
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [replayGameId, setReplayGameId] = useState('194335024');
  const [gotoStep, setGotoStep] = useState('');
  return (
    <div className="game-controls">
      <h3>Controls</h3>

      <div className="control-section">
        <button
          className="btn btn-primary"
          onClick={() => onStartGame(false)}
        >
          Start Game (Random)
        </button>

        <button
          className="btn btn-secondary"
          onClick={() => onStartGame(true)}
        >
          Start Game (LLM)
        </button>
      </div>

      <div className="control-section">
        <h4>Load Colonist Replay</h4>
        <input
          type="text"
          value={replayGameId}
          onChange={(e) => setReplayGameId(e.target.value)}
          placeholder="Game ID (e.g. 194335024)"
          style={{
            width: '100%',
            padding: '8px',
            marginBottom: '8px',
            borderRadius: '4px',
            border: '1px solid #444',
            background: '#1a1a2e',
            color: '#fff',
          }}
        />
        <button
          className="btn btn-action"
          onClick={() => onLoadReplay(replayGameId)}
          disabled={!replayGameId.trim()}
        >
          Load Replay
        </button>
      </div>

      {hasGame && (
        <div className="control-section">
          <button
            className="btn btn-danger"
            onClick={onReset}
          >
            Clear Game
          </button>
        </div>
      )}

      {replayMode && (
        <div className="control-section">
          <div style={{
            background: '#21262d',
            border: '2px solid #58a6ff',
            borderRadius: '8px',
            padding: '12px',
            marginBottom: '12px',
            textAlign: 'center',
          }}>
            <div style={{ color: '#8b949e', fontSize: '0.75rem', marginBottom: '4px' }}>
              REPLAY STEP
            </div>
            <div style={{ color: '#58a6ff', fontSize: '1.8rem', fontWeight: 'bold', fontFamily: 'monospace' }}>
              {replayProgress}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
            <button
              className="btn btn-secondary"
              onClick={onReplayUndo}
            >
              Undo
            </button>
            <button
              className="btn btn-action"
              onClick={onReplayStep}
              disabled={!isRunning}
            >
              Step
            </button>
          </div>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
            <input
              type="number"
              value={gotoStep}
              onChange={(e) => setGotoStep(e.target.value)}
              placeholder="Step #"
              style={{
                width: '80px',
                padding: '6px 8px',
                borderRadius: '4px',
                border: '1px solid #444',
                background: '#1a1a2e',
                color: '#fff',
              }}
            />
            <button
              className="btn btn-secondary"
              onClick={() => {
                const step = parseInt(gotoStep, 10);
                if (!isNaN(step) && onSetReplayStep) {
                  onSetReplayStep(step);
                  setGotoStep('');
                }
              }}
              disabled={!gotoStep.trim() || !onSetReplayStep}
            >
              Go to Step
            </button>
          </div>
          {onRunUntilDrift && (
            <button
              className="btn btn-action"
              onClick={onRunUntilDrift}
              style={{ marginBottom: '8px', width: '100%' }}
            >
              Run Until Drift
            </button>
          )}
          <button
            className="btn btn-danger"
            onClick={onReset}
          >
            Exit Replay
          </button>
        </div>
      )}

      {hasGame && !replayMode && (
        <>
          <div className="control-section">
            <button
              className="btn btn-action"
              onClick={onStep}
              disabled={!isRunning || (isCurrentPlayerLlm && isLlmProcessing)}
            >
              {(isCurrentPlayerLlm && isLlmProcessing) ? 'LLM Thinking...' : 'Step'}
            </button>

            {!isAutoPlaying ? (
              <button
                className="btn btn-action"
                onClick={() => {
                  setIsAutoPlaying(true);
                  onAutoPlay(0.5);
                }}
                disabled={!isRunning}
              >
                Auto Play
              </button>
            ) : (
              <button
                className="btn btn-danger"
                onClick={() => {
                  setIsAutoPlaying(false);
                  onStopAutoPlay();
                }}
              >
                Stop Auto Play
              </button>
            )}
          </div>

          <div className="status">
            <span className={`status-indicator ${isRunning ? 'running' : 'stopped'}`} />
            {isRunning ? 'Game Running' : 'Game Stopped'}
          </div>
        </>
      )}
    </div>
  );
}
