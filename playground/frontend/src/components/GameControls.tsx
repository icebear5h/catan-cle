import { useState } from 'react';
import type { LiveColorPalette, NativeReasoningEffort } from '../types';
import './GameControls.css';

const MODEL_PRESETS: Array<{ id: string; label: string }> = [
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
  { id: 'cerebras/qwen-3.8-27b', label: 'Qwen3.8-27B · Cerebras · fast' },
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
  onReset: () => void;
  onLoadReplay: (gameId: string) => void;
  onRunUntilDrift?: () => void;
  onGenerateReplayResponse: () => void;
  liveModel: string;
  onLiveModelChange: (model: string) => void;
  replayModel: string;
  onReplayModelChange: (model: string) => void;
  nativeReasoningEffort: NativeReasoningEffort;
  onNativeReasoningEffortChange: (effort: NativeReasoningEffort) => void;
  liveColorPalette: LiveColorPalette;
  onLiveColorPaletteChange: (palette: LiveColorPalette) => void;
  isReplayLlmProcessing?: boolean;
  replayLlmError?: string | null;
  isRunning: boolean;
  hasGame: boolean;
  isLlmProcessing?: boolean;
  liveError?: string | null;
  liveTraceGameId?: string | null;
  liveTraceDatabase?: string | null;
  isPlaybackProcessing?: boolean;
  replayMode?: boolean;
}

export default function GameControls({
  onStartGame,
  onReset,
  onLoadReplay,
  onRunUntilDrift,
  onGenerateReplayResponse,
  liveModel,
  onLiveModelChange,
  replayModel,
  onReplayModelChange,
  nativeReasoningEffort,
  onNativeReasoningEffortChange,
  liveColorPalette,
  onLiveColorPaletteChange,
  isReplayLlmProcessing = false,
  replayLlmError = null,
  isRunning,
  hasGame,
  isLlmProcessing = false,
  liveError = null,
  liveTraceGameId = null,
  liveTraceDatabase = null,
  isPlaybackProcessing = false,
  replayMode = false,
}: GameControlsProps) {
  const [replayGameId, setReplayGameId] = useState('242781000');
  return (
    <div className="game-controls">
      <h3>Controls</h3>

      {hasGame && (
        <div className="control-section">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onReset}
            disabled={isLlmProcessing || isPlaybackProcessing || isReplayLlmProcessing}
          >
            New Game
          </button>
          <p className="replay-context-hint">
            Return to setup for a fresh game. Saved games stay available.
          </p>
        </div>
      )}

      {liveError && (
        <div className="replay-llm-error" role="alert">
          {liveError}
        </div>
      )}

      {!hasGame && (
        <div className="control-section">
          <label className="field-label" htmlFor="live-color-palette">
            Player colors
          </label>
          <select
            id="live-color-palette"
            className="replay-model-select"
            value={liveColorPalette}
            onChange={(event) => onLiveColorPaletteChange(
              event.target.value as LiveColorPalette,
            )}
            disabled={isRunning || isLlmProcessing}
          >
            <option value="random_all">Random four of all 11</option>
            <option value="canonical_four">RED / BLUE / WHITE / ORANGE</option>
          </select>

          <label className="field-label" htmlFor="live-model-id">
            Live LLM model
          </label>
          <select
            className="replay-model-select"
            aria-label="Live model presets"
            value={MODEL_PRESETS.some((preset) => preset.id === liveModel) ? liveModel : ''}
            onChange={(event) => {
              if (event.target.value) {
                onLiveModelChange(event.target.value);
              }
            }}
            disabled={isRunning || isLlmProcessing}
          >
            <option value="">Presets…</option>
            {MODEL_PRESETS.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.label} — {preset.id}
              </option>
            ))}
          </select>
          <input
            id="live-model-id"
            className="replay-model-input"
            type="text"
            value={liveModel}
            onChange={(event) => onLiveModelChange(event.target.value)}
            placeholder="provider/model"
            spellCheck={false}
            disabled={isRunning || isLlmProcessing}
          />

          <button
            className="btn btn-primary"
            onClick={() => onStartGame('random')}
            disabled={isLlmProcessing || isPlaybackProcessing || isReplayLlmProcessing}
          >
            Start Game (Random)
          </button>

          <button
            className="btn btn-secondary"
            onClick={() => onStartGame('llm_vs_random')}
            disabled={isLlmProcessing || isPlaybackProcessing || isReplayLlmProcessing}
          >
            LLM vs Random
          </button>

          <button
            className="btn btn-secondary"
            onClick={() => onStartGame('llm')}
            disabled={isLlmProcessing || isPlaybackProcessing || isReplayLlmProcessing}
          >
            Start Game (LLM)
          </button>
        </div>
      )}

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
          disabled={isReplayLlmProcessing || isLlmProcessing || isPlaybackProcessing}
        >
          <option value="off">Off — explicit no reasoning</option>
          <option value="minimal">Minimal</option>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High — validated default</option>
          <option value="xhigh">XHigh</option>
          <option value="max">Max</option>
        </select>
        <p className="replay-context-hint">
          Sent explicitly to supported providers. Reasoning is read only from
          the provider-native channel and is not requested in the action XML.
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
          disabled={!replayGameId.trim() || isPlaybackProcessing || isLlmProcessing}
        >
          Load Replay
        </button>
      </div>

      {hasGame && !replayMode && (
        <div className="control-section session-status-section">
          <h4>Live session</h4>
          <div className="status">
            <span className={`status-indicator ${isRunning ? 'running' : 'stopped'}`} />
            {isRunning ? 'Game Running' : 'Game Stopped'}
          </div>
          {liveTraceGameId && (
            <p className="replay-context-hint" title={liveTraceDatabase || undefined}>
              Local trace: {liveTraceGameId}
            </p>
          )}
        </div>
      )}

      {replayMode && (
        <div className="control-section replay-tools-section">
          <h4>Replay tools</h4>
          <p className="replay-context-hint">
            Previous, Step, and direct timeline navigation stay pinned beneath the board.
          </p>
          {onRunUntilDrift && (
            <button
              className="btn btn-action"
              onClick={onRunUntilDrift}
              disabled={isPlaybackProcessing || isReplayLlmProcessing}
            >
              {isPlaybackProcessing ? 'Working…' : 'Run Until Drift'}
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
              value={MODEL_PRESETS.some((preset) => preset.id === replayModel) ? replayModel : ''}
              onChange={(event) => {
                if (event.target.value) {
                  onReplayModelChange(event.target.value);
                }
              }}
              disabled={isReplayLlmProcessing || isPlaybackProcessing}
            >
              <option value="">Presets…</option>
              {MODEL_PRESETS.map((preset) => (
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
              disabled={isReplayLlmProcessing || isPlaybackProcessing}
            />
            <p className="replay-context-hint">
              Context: shared game plan + complete visible events + current observation.
            </p>
            <button
              className="btn btn-primary"
              onClick={onGenerateReplayResponse}
              disabled={
                !replayModel.trim()
                || isReplayLlmProcessing
                || isPlaybackProcessing
              }
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
            disabled={isLlmProcessing || isPlaybackProcessing || isReplayLlmProcessing}
          >
            Exit Replay
          </button>
        </div>
      )}
    </div>
  );
}
