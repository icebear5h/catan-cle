import { useState } from 'react';
import type { NativeReasoningEffort } from '../types';
import './GameControls.css';

const REPLAY_MODEL_PRESETS: Array<{ id: string; label: string }> = [
  // Teacher tier (large / frontier)
  { id: 'thinkingmachines/inkling-small', label: 'Inkling-Small · 276B/12B · VL' },
  { id: 'deepseek/deepseek-v4-flash-0731', label: 'DeepSeek V4 Flash · 284B/13B · cheap' },
  { id: 'z-ai/glm-5.2', label: 'GLM-5.2 · 744B/40B · frontier value' },
  { id: 'qwen/qwen3.5-397b-a17b', label: 'Qwen3.5-397B · VL teacher' },
  { id: 'xiaomi/mimo-v2.5', label: 'MiMo V2.5 · 310B/15B · VL value' },
  { id: 'minimax/minimax-m2.7', label: 'MiniMax M2.7 · 230B/10B' },
  { id: 'moonshotai/kimi-k3', label: 'Kimi K3 · 2.8T/104B · frontier' },
  { id: 'thinkingmachines/inkling', label: 'Inkling · 975B/41B' },
  { id: 'qwen/qwen3.8-max', label: 'Qwen3.8-Max · 2.4T/95B' },
  { id: 'qwen/qwen3.8-27b', label: 'Qwen3.8-27B · eval baseline' },
  { id: 'qwen/qwen3.7-flash', label: 'Qwen3.7 Flash · VL · cheap' },
  { id: 'google/gemini-2.5-flash', label: 'Gemini 2.5 Flash · default' },
  // Student tier (small)
  { id: 'qwen/qwen3.6-35b-a3b', label: 'Qwen3.6 35B-A3B · student' },
  { id: 'qwen/qwen3.5-27b', label: 'Qwen3.5-27B · student' },
  { id: 'google/gemma-4-31b-it', label: 'Gemma 4 31B · student' },
  { id: 'google/gemma-4-26b-a4b-it', label: 'Gemma 4 26B-A4B · MoE student' },
  { id: 'qwen/qwen3.5-9b', label: 'Qwen3.5-9B · local SFT twin' },
  { id: 'openai/gpt-oss-20b', label: 'GPT-OSS-20B · MoE student' },
  { id: 'nvidia/nemotron-3-nano-30b-a3b', label: 'Nemotron 3 Nano · student' },
];

interface GameControlsProps {
  onStartGame: (mode: string) => void;
  onStep: () => void;
  onReset: () => void;
  onLoadReplay: (gameId: string) => void;
  onReplayStep: () => void;
  onReplayUndo: () => void;
  onSetReplayStep?: (step: number) => void;
  onRunUntilDrift?: () => void;
  onGenerateReplayResponse: () => void;
  replayModel: string;
  onReplayModelChange: (model: string) => void;
  nativeReasoningEffort: NativeReasoningEffort;
  onNativeReasoningEffortChange: (effort: NativeReasoningEffort) => void;
  isReplayLlmProcessing?: boolean;
  replayLlmError?: string | null;
  isRunning: boolean;
  hasGame: boolean;
  isLlmProcessing?: boolean;
  liveError?: string | null;
  liveTraceGameId?: string | null;
  liveTraceDatabase?: string | null;
  isTraceBrowsing?: boolean;
  replayMode?: boolean;
  replayProgress?: string;
}

export default function GameControls({
  onStartGame,
  onStep,
  onReset,
  onLoadReplay,
  onReplayStep,
  onReplayUndo,
  onSetReplayStep,
  onRunUntilDrift,
  onGenerateReplayResponse,
  replayModel,
  onReplayModelChange,
  nativeReasoningEffort,
  onNativeReasoningEffortChange,
  isReplayLlmProcessing = false,
  replayLlmError = null,
  isRunning,
  hasGame,
  isLlmProcessing = false,
  liveError = null,
  liveTraceGameId = null,
  liveTraceDatabase = null,
  isTraceBrowsing = false,
  replayMode = false,
  replayProgress = '',
}: GameControlsProps) {
  const [replayGameId, setReplayGameId] = useState('242781000');
  const [gotoStep, setGotoStep] = useState('');
  return (
    <div className="game-controls">
      <h3>Controls</h3>

      {liveError && (
        <div className="replay-llm-error" role="alert">
          {liveError}
        </div>
      )}

      <div className="control-section">
        <button
          className="btn btn-primary"
          onClick={() => onStartGame('random')}
        >
          Start Game (Random)
        </button>

        <button
          className="btn btn-secondary"
          onClick={() => onStartGame('llm_vs_random')}
        >
          LLM vs Random
        </button>

        <button
          className="btn btn-secondary"
          onClick={() => onStartGame('llm')}
        >
          Start Game (LLM)
        </button>
      </div>

      <div className="control-section">
        <h4>Native reasoning</h4>
        <label className="field-label" htmlFor="native-reasoning-effort">
          OpenRouter effort
        </label>
        <select
          id="native-reasoning-effort"
          className="replay-model-select"
          value={nativeReasoningEffort}
          onChange={(event) => onNativeReasoningEffortChange(
            event.target.value as NativeReasoningEffort,
          )}
          disabled={isReplayLlmProcessing || isLlmProcessing}
        >
          <option value="off">Off — explicit no reasoning</option>
          <option value="minimal">Minimal</option>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
          <option value="xhigh">XHigh — exploratory default</option>
          <option value="max">Max</option>
        </select>
        <p className="replay-context-hint">
          Sent explicitly to supported providers. Provider-native reasoning is
          kept separate from the model-authored &lt;rationale&gt; response.
        </p>
      </div>

      <div className="control-section">
        <h4>Load Colonist Replay</h4>
        <input
          type="text"
          value={replayGameId}
          onChange={(e) => setReplayGameId(e.target.value)}
          placeholder="Game ID (e.g. 242781000)"
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
        <p className="replay-context-hint">
          Default: paired transcript demo. Any local replay ID still works.
        </p>
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

          <div className="replay-llm-controls">
            <h4>LLM Response</h4>
            <label className="field-label" htmlFor="replay-model-id">
              OpenRouter model ID
            </label>
            <select
              className="replay-model-select"
              aria-label="Model presets"
              value={REPLAY_MODEL_PRESETS.some((preset) => preset.id === replayModel) ? replayModel : ''}
              onChange={(event) => {
                if (event.target.value) {
                  onReplayModelChange(event.target.value);
                }
              }}
              disabled={isReplayLlmProcessing}
            >
              <option value="">Presets…</option>
              {REPLAY_MODEL_PRESETS.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label} — {preset.id}
                </option>
              ))}
            </select>
            <input
              id="replay-model-id"
              className="replay-model-input"
              type="text"
              value={replayModel}
              onChange={(event) => onReplayModelChange(event.target.value)}
              placeholder="provider/model"
              spellCheck={false}
              disabled={isReplayLlmProcessing}
            />
            <p className="replay-context-hint">
              Context: shared game plan + complete visible events + current observation.
            </p>
            <button
              className="btn btn-primary"
              onClick={onGenerateReplayResponse}
              disabled={!replayModel.trim() || isReplayLlmProcessing}
            >
              {isReplayLlmProcessing ? 'Generating...' : 'Generate Response'}
            </button>
            {replayLlmError && (
              <div className="replay-llm-error" role="alert">
                {replayLlmError}
              </div>
            )}
          </div>

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
              disabled={!isRunning || isLlmProcessing || isTraceBrowsing}
            >
              {isLlmProcessing
                ? 'Stepping sandbox...'
                : isTraceBrowsing
                  ? 'Browsing saved step'
                  : 'Step'}
            </button>
            <p className="replay-context-hint">
              {isTraceBrowsing
                ? 'Browse-only checkpoint: load the latest checkpoint to resume gameplay.'
                : (
                  'Advances one complete CatanSandbox step, including any required '
                  + 'deterministic response barrier.'
                )}
            </p>
            {liveTraceGameId && (
              <p className="replay-context-hint" title={liveTraceDatabase || undefined}>
                Local trace: {liveTraceGameId}
              </p>
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
