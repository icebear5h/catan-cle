import { useCallback, useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';
import type {
  GameState,
  Color,
  Resource,
  ReplayInfo,
  ReplayLLMResponse,
  TableTalkEntry,
  NativeReasoningEffort,
  LiveReasoningTrace as LiveReasoningTraceRecord,
} from './types';
import HexBoard from './components/HexBoard';
import GameControls from './components/GameControls';
import PlayerInfo from './components/PlayerInfo';
import GameLog from './components/GameLog';
import ReplayResponseCard from './components/ReplayResponseCard';
import LiveReasoningTrace from './components/LiveReasoningTrace';
import TableTalkLog from './components/TableTalkLog';
import ReplayTranscriptPanel from './components/ReplayTranscriptPanel';
import SavedLiveGamesBar, {
  type SavedLiveGameSummary,
} from './components/SavedLiveGamesBar';
import TraceStepNavigator, {
  type TraceStepDetail,
} from './components/TraceStepNavigator';
import './App.css';

// Toggle between servers: 5001 (mixed) or 5002 (4-LLM)
const SERVER_URL = 'http://127.0.0.1:5001';  // Main server
const REPLAY_MODEL_STORAGE_KEY = 'catan-lab.replay-model';
const NATIVE_REASONING_STORAGE_KEY = 'catan-lab.native-reasoning-effort';
const DEFAULT_REPLAY_MODEL = 'qwen/qwen3.8-27b';
const DEFAULT_NATIVE_REASONING_EFFORT: NativeReasoningEffort = 'xhigh';
const NATIVE_REASONING_EFFORTS = new Set<NativeReasoningEffort>([
  'off',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
]);

function storedNativeReasoningEffort(): NativeReasoningEffort {
  const stored = window.localStorage.getItem(NATIVE_REASONING_STORAGE_KEY);
  return stored && NATIVE_REASONING_EFFORTS.has(stored as NativeReasoningEffort)
    ? stored as NativeReasoningEffort
    : DEFAULT_NATIVE_REASONING_EFFORT;
}

function nativeReasoningRequest(effort: NativeReasoningEffort): Record<string, unknown> {
  return effort === 'off'
    ? { enabled: false }
    : { effort, exclude: false };
}

interface ReplayCursor {
  gameId: string;
  eventIndex: number;
}

interface GameLogEntry {
  type: 'dice' | 'resource' | 'building' | 'trade' | 'robber' | 'general';
  timestamp: number;
  message: string;
  color?: string;
  details?: unknown;
}

interface DevCardCounts {
  in_hand: Record<string, number>;
  played: Record<string, number>;
  total_in_hand: number;
}

interface ReplayStepResult {
  action?: string;
  engine_translation?: unknown;
  colonist_event?: unknown;
}

interface StateSnapshot {
  game: GameState | null;
  running: boolean;
  game_log?: GameLogEntry[];
  all_player_resources?: Record<Color, Record<Resource, number>> | null;
  all_player_dev_cards?: Record<string, DevCardCounts> | null;
  player_types?: Record<string, string> | null;
  replay_mode?: boolean;
  replay?: ReplayInfo | null;
  last_dice_roll?: [number, number] | null;
}

function getApiError(payload: unknown, fallback: string): string {
  if (typeof payload !== 'object' || payload === null) {
    return fallback;
  }
  const record = payload as Record<string, unknown>;
  for (const key of ['details', 'message', 'error'] as const) {
    const value = record[key];
    if (typeof value === 'string') {
      return value;
    }
  }
  return fallback;
}

async function readApiObject(response: Response): Promise<Record<string, unknown>> {
  const text = await response.text();
  try {
    const payload: unknown = JSON.parse(text);
    if (typeof payload === 'object' && payload !== null && !Array.isArray(payload)) {
      return payload as Record<string, unknown>;
    }
  } catch {
    // The error below includes the HTTP status and a bounded response preview.
  }
  const contentType = response.headers.get('content-type') || 'unknown content type';
  const preview = text.trim().replace(/\s+/g, ' ').slice(0, 160);
  throw new Error(
    `API ${response.status} returned non-JSON (${contentType}): ${preview || 'empty body'}`,
  );
}

function App() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [liveTraceGameId, setLiveTraceGameId] = useState<string | null>(null);
  const [liveTraceDatabase, setLiveTraceDatabase] = useState<string | null>(null);
  const [savedLiveGames, setSavedLiveGames] = useState<SavedLiveGameSummary[]>([]);
  const [selectedSavedGameId, setSelectedSavedGameId] = useState('');
  const [savedGamesBusy, setSavedGamesBusy] = useState(false);
  const [savedGamesError, setSavedGamesError] = useState<string | null>(null);
  const [browsedTraceStep, setBrowsedTraceStep] = useState<TraceStepDetail | null>(null);
  const [traceBrowseBusy, setTraceBrowseBusy] = useState(false);
  const [traceBrowseError, setTraceBrowseError] = useState<string | null>(null);
  const [liveReasoningTraces, setLiveReasoningTraces] = useState<LiveReasoningTraceRecord[]>([]);
  const [gameLog, setGameLog] = useState<GameLogEntry[]>([]);
  const [allPlayerResources, setAllPlayerResources] = useState<Record<Color, Record<Resource, number>> | null>(null);
  const [allPlayerDevCards, setAllPlayerDevCards] = useState<Record<string, DevCardCounts> | null>(null);
  const [isLlmProcessing, setIsLlmProcessing] = useState(false);
  const [playerTypes, setPlayerTypes] = useState<Record<string, string> | null>(null);
  const [replayMode, setReplayMode] = useState(false);
  const [replayProgress, setReplayProgress] = useState('');
  const [replayInfo, setReplayInfo] = useState<ReplayInfo | null>(null);
  const [lastReplayStep, setLastReplayStep] = useState<ReplayStepResult | null>(null);
  const [lastDiceRoll, setLastDiceRoll] = useState<[number, number] | null>(null);
  const [replayModel, setReplayModel] = useState(
    () => window.localStorage.getItem(REPLAY_MODEL_STORAGE_KEY) || DEFAULT_REPLAY_MODEL,
  );
  const [nativeReasoningEffort, setNativeReasoningEffort] = useState(
    storedNativeReasoningEffort,
  );
  const [replayGamePlan, setReplayGamePlan] = useState('');
  const [replayLlmResponse, setReplayLlmResponse] = useState<ReplayLLMResponse | null>(null);
  const [tableTalkLog, setTableTalkLog] = useState<TableTalkEntry[]>([]);
  const [replayLlmError, setReplayLlmError] = useState<string | null>(null);
  const [isReplayLlmProcessing, setIsReplayLlmProcessing] = useState(false);
  const replayCursorRef = useRef<ReplayCursor | null>(null);
  const replayModelRef = useRef(replayModel);
  const savedGamesRequestRef = useRef(0);
  const traceBrowseRequestRef = useRef(0);

  const applyStateSnapshot = useCallback((data: StateSnapshot) => {
    setGameState(data.game);
    setIsRunning(data.running);
    setGameLog(data.game_log || []);
    setAllPlayerResources(data.all_player_resources || null);
    setAllPlayerDevCards(data.all_player_dev_cards || null);
    if (data.player_types !== undefined) {
      setPlayerTypes(data.player_types);
    }

    const nextReplayMode = Boolean(data.replay_mode);
    const nextReplay = data.replay || null;
    const previousCursor = replayCursorRef.current;
    if (!nextReplayMode || !nextReplay) {
      replayCursorRef.current = null;
      setReplayGamePlan('');
      setReplayLlmResponse(null);
      setReplayLlmError(null);
      setTableTalkLog([]);
    } else {
      const changedGame = previousCursor?.gameId !== nextReplay.game_id;
      const changedCursor = previousCursor?.eventIndex !== nextReplay.event_index;
      const movedBackward = (
        previousCursor?.gameId === nextReplay.game_id
        && nextReplay.event_index < previousCursor.eventIndex
      );

      if (changedGame) {
        setTableTalkLog([]);
      } else if (movedBackward) {
        setTableTalkLog((prev) => prev.filter(
          (entry) => entry.replayIndex <= nextReplay.event_index,
        ));
      }
      if (changedGame || movedBackward) {
        setReplayGamePlan('');
      }
      if (changedGame || changedCursor) {
        setReplayLlmResponse(null);
        setReplayLlmError(null);
      }
      replayCursorRef.current = {
        gameId: nextReplay.game_id,
        eventIndex: nextReplay.event_index,
      };
    }

    setReplayMode(nextReplayMode);
    setReplayProgress(nextReplay?.progress || '');
    setReplayInfo(nextReplay);
    if (data.last_dice_roll !== undefined) {
      setLastDiceRoll(data.last_dice_roll);
    }
  }, []);

  const refreshSavedGames = useCallback(async () => {
    const requestId = ++savedGamesRequestRef.current;
    try {
      setSavedGamesBusy(true);
      setSavedGamesError(null);
      const response = await fetch(`${SERVER_URL}/api/live-traces?limit=100`);
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, 'Failed to list saved live games'));
      }
      const games = Array.isArray(data.games)
        ? data.games as SavedLiveGameSummary[]
        : [];
      if (requestId !== savedGamesRequestRef.current) {
        return;
      }
      setSavedLiveGames(games);
      setSelectedSavedGameId((previous) => (
        games.some((game) => game.game_id === previous)
          ? previous
          : games[0]?.game_id || ''
      ));
    } catch (error) {
      if (requestId === savedGamesRequestRef.current) {
        const message = error instanceof Error ? error.message : String(error);
        setSavedGamesError(message);
      }
    } finally {
      if (requestId === savedGamesRequestRef.current) {
        setSavedGamesBusy(false);
      }
    }
  }, []);

  useEffect(() => {
    void refreshSavedGames();
  }, [refreshSavedGames]);

  const browseSavedStep = async (gameId: string, stepIndex: number) => {
    const requestId = ++traceBrowseRequestRef.current;
    try {
      setTraceBrowseBusy(true);
      setTraceBrowseError(null);
      const response = await fetch(
        `${SERVER_URL}/api/live-traces/${gameId}/steps/${stepIndex}`,
      );
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, 'Failed to load saved checkpoint'));
      }
      const detail = data as unknown as TraceStepDetail;
      if (requestId !== traceBrowseRequestRef.current) {
        return;
      }
      setBrowsedTraceStep(detail);
      setSelectedSavedGameId(gameId);
      setLiveReasoningTraces([]);
      applyStateSnapshot(detail.step.public_state as StateSnapshot);
    } catch (error) {
      if (requestId === traceBrowseRequestRef.current) {
        const message = error instanceof Error ? error.message : String(error);
        setTraceBrowseError(message);
      }
    } finally {
      if (requestId === traceBrowseRequestRef.current) {
        setTraceBrowseBusy(false);
      }
    }
  };

  const selectSavedGame = (gameId: string) => {
    traceBrowseRequestRef.current += 1;
    setTraceBrowseBusy(false);
    setSelectedSavedGameId(gameId);
    setBrowsedTraceStep(null);
    setTraceBrowseError(null);
    const selected = savedLiveGames.find((game) => game.game_id === gameId);
    if (selected && selected.step_count > 0) {
      void browseSavedStep(gameId, selected.step_count - 1);
    }
  };

  useEffect(() => {
    const newSocket = io(SERVER_URL);

    newSocket.on('connect', () => {
      console.log('Connected to game server');
    });

    newSocket.on('game_state', (data) => {
      console.log('='.repeat(80));
      console.log('[FRONTEND] Received game state update');
      console.log('='.repeat(80));
      console.log('Game:', data.game);
      console.log('Running:', data.running);
      console.log('Current player:', data.game?.current_color);
      console.log('Game log entries:', data.game_log?.length || 0);
      console.log('Player types:', data.player_types);
      console.log('='.repeat(80));

      applyStateSnapshot(data as StateSnapshot);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
    });

    return () => {
      newSocket.close();
    };
  }, [applyStateSnapshot]);

  const startGame = async (mode: string = 'random') => {
    traceBrowseRequestRef.current += 1;
    setTraceBrowseBusy(false);
    setLiveError(null);
    setBrowsedTraceStep(null);
    setTraceBrowseError(null);
    setLiveReasoningTraces([]);
    setLiveTraceGameId(null);
    setLiveTraceDatabase(null);
    replayCursorRef.current = null;
    setReplayGamePlan('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
    setTableTalkLog([]);
    try {
      const response = await fetch(`${SERVER_URL}/api/start-game`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          reasoning: nativeReasoningRequest(nativeReasoningEffort),
        }),
      });
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, 'Failed to start sandbox'));
      }
      if (data.state) {
        applyStateSnapshot(data.state as StateSnapshot);
      }
      const traceGameId = (
        typeof data.trace_game_id === 'string' ? data.trace_game_id : null
      );
      setLiveTraceGameId(traceGameId);
      setLiveTraceDatabase(
        typeof data.trace_database === 'string' ? data.trace_database : null,
      );
      if (traceGameId) {
        setSelectedSavedGameId(traceGameId);
      }
      void refreshSavedGames();
      console.log('Game started:', data);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLiveError(message);
      console.error('Error starting game:', error);
    }
  };

  const stepGame = async () => {
    try {
      setLiveError(null);
      const startTime = performance.now();
      console.log('[FRONTEND] Step game - sending request...');
      setIsLlmProcessing(true);

      const response = await fetch(`${SERVER_URL}/api/step`, {
        method: 'POST',
      });
      const data = await readApiObject(response);

      const duration = performance.now() - startTime;
      console.log(`[FRONTEND] Step game - response received (${(duration / 1000).toFixed(3)}s):`, data);

      if (!response.ok) {
        throw new Error(getApiError(data, 'Sandbox step failed'));
      }
      if (data.state) {
        applyStateSnapshot(data.state as StateSnapshot);
      }
      traceBrowseRequestRef.current += 1;
      setTraceBrowseBusy(false);
      setBrowsedTraceStep(null);
      setTraceBrowseError(null);
      if (Array.isArray(data.reasoning_traces)) {
        setLiveReasoningTraces((previous) => [
          ...previous,
          ...(data.reasoning_traces as LiveReasoningTraceRecord[]),
        ].slice(-12));
      }
      void refreshSavedGames();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLiveError(message);
      console.error('[FRONTEND] Error stepping game:', error);
    } finally {
      setIsLlmProcessing(false);
    }
  };

  const resetGame = async () => {
    traceBrowseRequestRef.current += 1;
    setTraceBrowseBusy(false);
    setLiveError(null);
    setBrowsedTraceStep(null);
    setTraceBrowseError(null);
    setLiveReasoningTraces([]);
    setLiveTraceGameId(null);
    setLiveTraceDatabase(null);
    replayCursorRef.current = null;
    setReplayGamePlan('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
    try {
      const response = await fetch(`${SERVER_URL}/api/reset`, {
        method: 'POST',
      });
      const data = await response.json();
      void refreshSavedGames();
      console.log('Game reset:', data);
    } catch (error) {
      console.error('Error resetting game:', error);
    }
  };

  const renameSavedGame = async (gameId: string, name: string) => {
    try {
      setSavedGamesBusy(true);
      setSavedGamesError(null);
      const response = await fetch(`${SERVER_URL}/api/live-traces/${gameId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, 'Failed to name saved live game'));
      }
      setSelectedSavedGameId(gameId);
      await refreshSavedGames();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSavedGamesError(message);
    } finally {
      setSavedGamesBusy(false);
    }
  };

  const loadSavedGame = async (gameId: string) => {
    traceBrowseRequestRef.current += 1;
    setTraceBrowseBusy(false);
    try {
      setSavedGamesBusy(true);
      setBrowsedTraceStep(null);
      setTraceBrowseError(null);
      setSavedGamesError(null);
      setLiveError(null);
      setLiveReasoningTraces([]);
      const response = await fetch(
        `${SERVER_URL}/api/live-traces/${gameId}/load`,
        { method: 'POST' },
      );
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, 'Failed to load saved live game'));
      }
      if (data.state) {
        applyStateSnapshot(data.state as StateSnapshot);
      }
      setLiveTraceGameId(gameId);
      setLiveTraceDatabase(
        typeof data.trace_database === 'string' ? data.trace_database : null,
      );
      setSelectedSavedGameId(gameId);
      await refreshSavedGames();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSavedGamesError(message);
      setLiveError(message);
    } finally {
      setSavedGamesBusy(false);
    }
  };

  const loadReplay = async (gameId: string) => {
    traceBrowseRequestRef.current += 1;
    setTraceBrowseBusy(false);
    setBrowsedTraceStep(null);
    setTraceBrowseError(null);
    setReplayGamePlan('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
    try {
      console.log('[FRONTEND] Loading replay:', gameId);
      const response = await fetch(`${SERVER_URL}/api/load-replay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
      });
      const data = await response.json();
      if (response.ok) {
        console.log('Replay loaded:', data);
      } else {
        console.error('Failed to load replay:', data.error);
        alert(`Failed to load replay: ${data.error}`);
      }
    } catch (error) {
      console.error('Error loading replay:', error);
      alert(`Error loading replay: ${error}`);
    }
  };

  const replayStep = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/replay-step`, {
        method: 'POST',
      });
      const data = await response.json();
      console.log('Replay step:', data);
      setLastReplayStep(data);
    } catch (error) {
      console.error('Error stepping replay:', error);
    }
  };

  const replayUndo = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/replay-undo`, {
        method: 'POST',
      });
      const data = await response.json();
      console.log('Replay undo:', data);
    } catch (error) {
      console.error('Error undoing replay:', error);
    }
  };

  const setReplayStep = async (step: number) => {
    try {
      const response = await fetch(`${SERVER_URL}/api/replay-goto-sequential`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step }),
      });
      const data = await response.json();
      if (!response.ok) {
        console.error('Replay goto failed:', response.status, data);
      } else {
        console.log('Replay goto:', data);
      }
    } catch (error) {
      console.error('Error setting replay step:', error);
    }
  };

  const runUntilDrift = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/replay-goto-divergence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_steps: 500 }),
      });
      const data = await response.json();
      console.log('Run until drift:', data);
    } catch (error) {
      console.error('Error running until drift:', error);
    }
  };

  const updateReplayModel = (model: string) => {
    replayModelRef.current = model;
    setReplayModel(model);
    window.localStorage.setItem(REPLAY_MODEL_STORAGE_KEY, model);
    setReplayGamePlan('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
  };

  const updateNativeReasoningEffort = (effort: NativeReasoningEffort) => {
    setNativeReasoningEffort(effort);
    window.localStorage.setItem(NATIVE_REASONING_STORAGE_KEY, effort);
    setReplayLlmResponse(null);
    setReplayLlmError(null);
  };

  const generateReplayResponse = async () => {
    const requestCursor = replayCursorRef.current;
    const requestModel = replayModel.trim();
    if (!requestCursor) {
      setReplayLlmError('Navigate to a loaded replay position first.');
      return;
    }

    try {
      setIsReplayLlmProcessing(true);
      setReplayLlmError(null);

      const response = await fetch(`${SERVER_URL}/api/replay-llm-response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: requestModel,
          game_plan: replayGamePlan,
          reasoning: nativeReasoningRequest(nativeReasoningEffort),
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(getApiError(payload, 'Failed to generate replay response'));
      }

      const generated = payload as ReplayLLMResponse;
      const currentCursor = replayCursorRef.current;
      const clientDetectedStale = (
        currentCursor === null
        || currentCursor.gameId !== requestCursor.gameId
        || currentCursor.eventIndex !== requestCursor.eventIndex
        || replayModelRef.current.trim() !== requestModel
      );
      const result = {
        ...generated,
        stale: generated.stale || clientDetectedStale,
      };

      setReplayLlmResponse(result);
      if (!result.stale && result.game_plan) {
        setReplayGamePlan(result.game_plan);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error('Error generating replay response:', error);
      setReplayLlmError(message);
    } finally {
      setIsReplayLlmProcessing(false);
    }
  };

  const selectedSavedGame = savedLiveGames.find(
    (game) => game.game_id === selectedSavedGameId,
  ) || null;
  const isTraceBrowsing = browsedTraceStep !== null;

  return (
    <div className="app">
      <header>
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>Catan Lab</h1>
            <p>engine oracle / visual grounding</p>
          </div>
        </div>
      </header>

      <SavedLiveGamesBar
        games={savedLiveGames}
        selectedGameId={selectedSavedGameId}
        activeGameId={liveTraceGameId}
        busy={savedGamesBusy || isLlmProcessing}
        error={savedGamesError}
        onSelect={selectSavedGame}
        onLoad={loadSavedGame}
        onRename={renameSavedGame}
        onRefresh={() => { void refreshSavedGames(); }}
      />

      <TraceStepNavigator
        game={selectedSavedGame}
        detail={browsedTraceStep}
        activeGameId={liveTraceGameId}
        busy={traceBrowseBusy || savedGamesBusy || isLlmProcessing}
        error={traceBrowseError}
        onNavigate={browseSavedStep}
        onLoadLatest={loadSavedGame}
      />

      <div className="main-container">
        <div className="left-panel">
          <GameControls
            onStartGame={startGame}
            onStep={stepGame}
            onReset={resetGame}
            onLoadReplay={loadReplay}
            onReplayStep={replayStep}
            onReplayUndo={replayUndo}
            onSetReplayStep={setReplayStep}
            isRunning={isRunning}
            hasGame={gameState !== null}
            isLlmProcessing={isLlmProcessing}
            liveError={liveError}
            liveTraceGameId={liveTraceGameId}
            liveTraceDatabase={liveTraceDatabase}
            isTraceBrowsing={isTraceBrowsing}
            replayMode={replayMode}
            replayProgress={replayProgress}
            onRunUntilDrift={runUntilDrift}
            onGenerateReplayResponse={generateReplayResponse}
            replayModel={replayModel}
            onReplayModelChange={updateReplayModel}
            nativeReasoningEffort={nativeReasoningEffort}
            onNativeReasoningEffortChange={updateNativeReasoningEffort}
            isReplayLlmProcessing={isReplayLlmProcessing}
            replayLlmError={replayLlmError}
          />

          {gameState && (
            <PlayerInfo
              gameState={gameState}
              allPlayerResources={allPlayerResources}
              allPlayerDevCards={allPlayerDevCards}
              playerTypes={playerTypes}
              replayInfo={replayInfo}
            />
          )}

          {replayMode && lastReplayStep && (
            <div className="replay-step-viewer" style={{
              marginTop: '10px',
              background: '#161b22',
              border: '1px solid #30363d',
              borderRadius: '6px',
              padding: '1rem',
              fontSize: '0.8rem',
            }}>
              <h3 style={{ color: '#8b949e', fontSize: '0.9rem', marginBottom: '0.5rem' }}>Last Replay Step</h3>
              <div style={{ marginBottom: '0.5rem' }}>
                <strong style={{ color: '#58a6ff' }}>Engine Action:</strong>
                <pre style={{ margin: '0.25rem 0', color: '#e6edf3', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  {lastReplayStep.action}
                </pre>
              </div>
              <div style={{ marginBottom: '0.5rem' }}>
                <strong style={{ color: '#3fb950' }}>Translation:</strong>
                <pre style={{ margin: '0.25rem 0', color: '#e6edf3', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  {JSON.stringify(lastReplayStep.engine_translation, null, 2)}
                </pre>
              </div>
              <div>
                <strong style={{ color: '#f0883e' }}>Colonist Event:</strong>
                <pre style={{ margin: '0.25rem 0', color: '#8b949e', whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '150px', overflow: 'auto' }}>
                  {JSON.stringify(lastReplayStep.colonist_event, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>

        <div className="board-container">
          {lastDiceRoll && (
            <div style={{
              position: 'absolute',
              top: '10px',
              left: '10px',
              display: 'flex',
              gap: '8px',
              zIndex: 100,
            }}>
              {lastDiceRoll.map((die, i) => (
                <div key={i} style={{
                  width: '48px',
                  height: '48px',
                  background: '#fff',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: '#1a1a1a',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                }}>
                  {die}
                </div>
              ))}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '20px',
                fontWeight: 'bold',
                color: '#fff',
                marginLeft: '4px',
              }}>
                = {lastDiceRoll[0] + lastDiceRoll[1]}
              </div>
            </div>
          )}
          {gameState ? (
            <HexBoard gameState={gameState} />
          ) : (
            <div className="no-game">
              <p>Start a game to view the board</p>
            </div>
          )}
        </div>

        {gameState && (
          gameLog.length > 0
          || liveReasoningTraces.length > 0
          || replayLlmResponse
          || tableTalkLog.length > 0
          || replayInfo?.paired_transcript
        ) && (
          <div className="right-panel">
            {replayInfo?.paired_transcript && (
              <ReplayTranscriptPanel
                transcript={replayInfo.paired_transcript}
                narratorReasoning={replayInfo.paired_narrator_reasoning}
                modelTrace={replayInfo.paired_model_trace}
              />
            )}

            {replayLlmResponse && (
              <ReplayResponseCard response={replayLlmResponse} />
            )}

            {!replayMode && liveReasoningTraces.length > 0 && (
              <LiveReasoningTrace traces={liveReasoningTraces} />
            )}

            {tableTalkLog.length > 0 && (
              <TableTalkLog entries={tableTalkLog} />
            )}

            {gameLog.length > 0 && (
              <GameLog entries={gameLog} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
