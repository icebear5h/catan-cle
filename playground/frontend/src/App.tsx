import { useCallback, useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';
import { motion } from 'motion/react';
import {
  Group,
  Panel,
  Separator,
  useDefaultLayout,
  usePanelRef,
} from 'react-resizable-panels';
import type { PanelSize } from 'react-resizable-panels';
import { runAutoPlayLoop } from './autoPlay';
import type { AutoPlayStepResult } from './autoPlay';
import { liveStepAutoPlayResult, liveStepErrorUpdate } from './liveStepErrors';
import type { LiveStepFailure, LiveStepWarning } from './liveStepErrors';
import type { AllPlayerDevCards } from './playerDevCards';
import type {
  GameState,
  AllPlayerResources,
  ReplayInfo,
  ReplayLLMResponse,
  TableTalkEntry,
  NativeReasoningEffort,
  LiveColorPalette,
  LiveReasoningTrace as LiveReasoningTraceRecord,
} from './types';
import HexBoard from './components/HexBoard';
import BoardControlsDock from './components/BoardControlsDock';
import GameControls from './components/GameControls';
import PlayerInfo from './components/PlayerInfo';
import GameLog from './components/GameLog';
import ReplayResponseCard from './components/ReplayResponseCard';
import LiveReasoningTrace from './components/LiveReasoningTrace';
import SavedStepReasoningTrace from './components/SavedStepReasoningTrace';
import RejectedLiveAttempts from './components/RejectedLiveAttempts';
import TableTalkLog from './components/TableTalkLog';
import ReplayTranscriptPanel from './components/ReplayTranscriptPanel';
import SavedLiveGamesBar, {
  type SavedLiveGameSummary,
} from './components/SavedLiveGamesBar';
import TraceStepNavigator, {
  type TraceStepDetail,
} from './components/TraceStepNavigator';
import PromptSuiteStudio from './components/PromptSuiteStudio';
import './App.css';

// Toggle between servers: 5001 (mixed) or 5002 (4-LLM)
const SERVER_URL = 'http://127.0.0.1:5001';  // Main server
const LIVE_MODEL_STORAGE_KEY = 'catan-lab.live-model';
const REPLAY_MODEL_STORAGE_KEY = 'catan-lab.replay-model';
const NATIVE_REASONING_STORAGE_KEY = 'catan-lab.native-reasoning-effort-v2';
const DEFAULT_LIVE_MODEL = 'qwen/qwen3.8-27b';
const DEFAULT_REPLAY_MODEL = 'qwen/qwen3.8-27b';
const DEFAULT_NATIVE_REASONING_EFFORT: NativeReasoningEffort = 'high';
const AUTO_PLAY_STEP_DELAY_MS = 750;
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

interface ReplayStepResult {
  action?: string;
  engine_translation?: unknown;
  colonist_event?: unknown;
}

interface LiveInferenceState {
  model: string | null;
  reasoning: {
    enabled?: boolean;
    effort?: string;
    exclude?: boolean;
  };
  max_tokens: number | null;
  max_decision_attempts: number;
}

interface StateSnapshot {
  game: GameState | null;
  running: boolean;
  live_trace_game_id?: string | null;
  live_inference?: LiveInferenceState | null;
  last_live_step_error?: LiveStepFailure | LiveStepWarning | null;
  game_log?: GameLogEntry[];
  all_player_resources?: AllPlayerResources | null;
  all_player_dev_cards?: AllPlayerDevCards | null;
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
    if (typeof value === 'string' && value.trim()) {
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
  const [workspace, setWorkspace] = useState<'game' | 'prompt-suite'>('game');
  const [promptSuiteDirty, setPromptSuiteDirty] = useState(false);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [hasActiveGame, setHasActiveGame] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [liveStepFailure, setLiveStepFailure] = useState<LiveStepFailure | null>(null);
  const [liveInference, setLiveInference] = useState<LiveInferenceState | null>(null);
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
  const [allPlayerResources, setAllPlayerResources] = useState<AllPlayerResources | null>(null);
  const [allPlayerDevCards, setAllPlayerDevCards] = useState<AllPlayerDevCards | null>(null);
  const [isLlmProcessing, setIsLlmProcessing] = useState(false);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [playerTypes, setPlayerTypes] = useState<Record<string, string> | null>(null);
  const [replayMode, setReplayMode] = useState(false);
  const [replayInfo, setReplayInfo] = useState<ReplayInfo | null>(null);
  const [lastReplayStep, setLastReplayStep] = useState<ReplayStepResult | null>(null);
  const [lastDiceRoll, setLastDiceRoll] = useState<[number, number] | null>(null);
  const [liveModel, setLiveModel] = useState(
    () => window.localStorage.getItem(LIVE_MODEL_STORAGE_KEY) || DEFAULT_LIVE_MODEL,
  );
  const [replayModel, setReplayModel] = useState(
    () => window.localStorage.getItem(REPLAY_MODEL_STORAGE_KEY) || DEFAULT_REPLAY_MODEL,
  );
  const [nativeReasoningEffort, setNativeReasoningEffort] = useState(
    storedNativeReasoningEffort,
  );
  const [liveColorPalette, setLiveColorPalette] = useState<LiveColorPalette>(
    'random_all',
  );
  const [replayGamePlan, setReplayGamePlan] = useState('');
  const [replayLlmResponse, setReplayLlmResponse] = useState<ReplayLLMResponse | null>(null);
  const [tableTalkLog, setTableTalkLog] = useState<TableTalkEntry[]>([]);
  const [replayLlmError, setReplayLlmError] = useState<string | null>(null);
  const [isReplayLlmProcessing, setIsReplayLlmProcessing] = useState(false);
  const [replayPlaybackError, setReplayPlaybackError] = useState<string | null>(null);
  const [isReplayPlaybackProcessing, setIsReplayPlaybackProcessing] = useState(false);
  const [isSessionDrawerOpen, setIsSessionDrawerOpen] = useState(true);
  const [isInspectorDrawerOpen, setIsInspectorDrawerOpen] = useState(true);
  const replayCursorRef = useRef<ReplayCursor | null>(null);
  const replayModelRef = useRef(replayModel);
  const savedGamesRequestRef = useRef(0);
  const traceBrowseRequestRef = useRef(0);
  const autoPlayRequestedRef = useRef(false);
  const stepInFlightRef = useRef(false);
  const appMountedRef = useRef(true);
  const sessionPanelRef = usePanelRef();
  const inspectorPanelRef = usePanelRef();
  const gamePanelLayout = useDefaultLayout({
    id: 'catan-lab-game-workspace-v3',
    panelIds: ['session', 'board', 'inspector'],
    storage: window.localStorage,
    onlySaveAfterUserInteractions: false,
  });

  const stopAutoPlay = useCallback(() => {
    autoPlayRequestedRef.current = false;
    setIsAutoPlaying(false);
  }, []);

  useEffect(() => {
    appMountedRef.current = true;
    return () => {
      appMountedRef.current = false;
      autoPlayRequestedRef.current = false;
    };
  }, []);

  const applyLiveStepError = useCallback((
    payload: unknown,
    source: 'runtime' | 'checkpoint' = 'runtime',
  ) => {
    const update = liveStepErrorUpdate(payload, source);
    if (update === undefined) return;
    if (update.message !== null) stopAutoPlay();
    setLiveError(update.message);
    setLiveStepFailure(update.failure);
  }, [stopAutoPlay]);

  const applyStateSnapshot = useCallback((
    data: StateSnapshot,
    source: 'runtime' | 'checkpoint' = 'runtime',
  ) => {
    setGameState(data.game);
    setIsRunning(data.running);
    setGameLog(data.game_log || []);
    if (data.live_inference !== undefined) {
      setLiveInference(data.live_inference);
    }
    applyLiveStepError(data.last_live_step_error, source);
    setAllPlayerResources(data.all_player_resources || null);
    setAllPlayerDevCards(data.all_player_dev_cards || null);
    if (data.player_types !== undefined) {
      setPlayerTypes(data.player_types);
    }

    const nextReplayMode = Boolean(data.replay_mode);
    const nextReplay = data.replay || null;
    const previousCursor = replayCursorRef.current;
    if (nextReplayMode && nextReplay) {
      traceBrowseRequestRef.current += 1;
      setTraceBrowseBusy(false);
      setBrowsedTraceStep(null);
      setSelectedSavedGameId('');
      setLiveTraceGameId(null);
      setLiveTraceDatabase(null);
    }
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
    setReplayInfo(nextReplay);
    if (data.last_dice_roll !== undefined) {
      setLastDiceRoll(data.last_dice_roll);
    }
  }, [applyLiveStepError]);

  const refreshSavedGames = useCallback(async () => {
    const requestId = ++savedGamesRequestRef.current;
    try {
      setSavedGamesBusy(true);
      setSavedGamesError(null);
      const response = await fetch(
        `${SERVER_URL}/api/live-traces?limit=100`,
        { cache: 'no-store' },
      );
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

  const browseSavedStep = useCallback(async (gameId: string, stepIndex: number) => {
    const requestId = ++traceBrowseRequestRef.current;
    try {
      setTraceBrowseBusy(true);
      setTraceBrowseError(null);
      const response = await fetch(
        `${SERVER_URL}/api/live-traces/${gameId}/steps/${stepIndex}`,
        { cache: 'no-store' },
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
      applyStateSnapshot(detail.step.public_state as StateSnapshot, 'checkpoint');
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
  }, [applyStateSnapshot]);

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
      const snapshot = data as StateSnapshot;
      const isRuntimeSnapshot = snapshot.game !== null && (
        Boolean(snapshot.replay_mode)
        || snapshot.live_trace_game_id !== undefined
      );
      setHasActiveGame(isRuntimeSnapshot);
      if (!snapshot.replay_mode && snapshot.live_trace_game_id !== undefined) {
        setLiveTraceGameId(snapshot.live_trace_game_id);
      }

      console.log('='.repeat(80));
      console.log('[FRONTEND] Received game state update');
      console.log('='.repeat(80));
      console.log('Game:', data.game);
      console.log('Running:', data.running);
      console.log('Current player:', data.game?.current_color);
      console.log('Game log entries:', data.game_log?.length || 0);
      console.log('Player types:', data.player_types);
      console.log('='.repeat(80));

      applyStateSnapshot(snapshot);
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
    applyLiveStepError(null);
    setBrowsedTraceStep(null);
    setTraceBrowseError(null);
    setLiveReasoningTraces([]);
    setLiveTraceGameId(null);
    setLiveTraceDatabase(null);
    setSelectedSavedGameId('');
    replayCursorRef.current = null;
    setReplayGamePlan('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
    setReplayPlaybackError(null);
    setTableTalkLog([]);
    try {
      const response = await fetch(`${SERVER_URL}/api/start-game`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          model: mode === 'random' ? null : liveModel.trim() || DEFAULT_LIVE_MODEL,
          palette: liveColorPalette,
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
      setHasActiveGame(true);
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

  const stepGame = useCallback(async (): Promise<AutoPlayStepResult> => {
    if (stepInFlightRef.current) {
      return { ok: false, running: false, gameOver: false };
    }

    stepInFlightRef.current = true;
    try {
      // Keep rejected diagnostics available while the retry is in flight.
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
        applyLiveStepError(data);
        return { ok: false, running: false, gameOver: false };
      }
      if (!data.state) {
        throw new Error('Live step response did not include authoritative state');
      }

      const snapshot = data.state as unknown as StateSnapshot;
      applyStateSnapshot(snapshot);
      setTraceBrowseError(null);
      const traceGameId = typeof data.trace_game_id === 'string'
        ? data.trace_game_id
        : null;
      const traceStepIndex = typeof data.trace_step_index === 'number'
        ? data.trace_step_index
        : null;
      if (traceGameId !== null && traceStepIndex !== null) {
        setSelectedSavedGameId(traceGameId);
        await browseSavedStep(traceGameId, traceStepIndex);
      } else {
        traceBrowseRequestRef.current += 1;
        setTraceBrowseBusy(false);
        setBrowsedTraceStep(null);
      }
      if (Array.isArray(data.reasoning_traces)) {
        setLiveReasoningTraces((previous) => [
          ...previous,
          ...(data.reasoning_traces as LiveReasoningTraceRecord[]),
        ].slice(-12));
      }
      void refreshSavedGames();
      return liveStepAutoPlayResult(data, snapshot);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLiveError(message);
      console.error('[FRONTEND] Error stepping game:', error);
      return { ok: false, running: false, gameOver: false };
    } finally {
      stepInFlightRef.current = false;
      if (appMountedRef.current) {
        setIsLlmProcessing(false);
      }
    }
  }, [applyLiveStepError, applyStateSnapshot, browseSavedStep, refreshSavedGames]);

  const resetGame = async () => {
    traceBrowseRequestRef.current += 1;
    setTraceBrowseBusy(false);
    applyLiveStepError(null);
    setBrowsedTraceStep(null);
    setTraceBrowseError(null);
    setLiveReasoningTraces([]);
    setLiveTraceGameId(null);
    setLiveTraceDatabase(null);
    setSelectedSavedGameId('');
    replayCursorRef.current = null;
    setReplayGamePlan('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
    setReplayPlaybackError(null);
    try {
      const response = await fetch(`${SERVER_URL}/api/reset`, {
        method: 'POST',
      });
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, 'Failed to clear game'));
      }
      setHasActiveGame(false);
      applyStateSnapshot({
        game: null,
        running: false,
        game_log: [],
        all_player_resources: null,
        all_player_dev_cards: null,
        player_types: null,
        replay_mode: false,
        replay: null,
        last_dice_roll: null,
      });
      void refreshSavedGames();
      console.log('Game reset:', data);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLiveError(message);
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
      applyLiveStepError(null);
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
      setHasActiveGame(true);
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
    setSelectedSavedGameId('');
    setReplayGamePlan('');
    setReplayLlmResponse(null);
    setReplayLlmError(null);
    setReplayPlaybackError(null);
    applyLiveStepError(null);
    setIsReplayPlaybackProcessing(true);
    try {
      console.log('[FRONTEND] Loading replay:', gameId);
      const response = await fetch(`${SERVER_URL}/api/load-replay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId }),
      });
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, 'Failed to load replay'));
      }
      setHasActiveGame(true);
      setLastReplayStep(null);
      console.log('Replay loaded:', data);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setReplayPlaybackError(message);
      setLiveError(message);
      console.error('Error loading replay:', error);
    } finally {
      setIsReplayPlaybackProcessing(false);
    }
  };

  const runReplayMutation = async (
    endpoint: string,
    init: RequestInit,
    fallbackError: string,
  ): Promise<Record<string, unknown> | null> => {
    setIsReplayPlaybackProcessing(true);
    setReplayPlaybackError(null);
    try {
      const response = await fetch(`${SERVER_URL}${endpoint}`, init);
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, fallbackError));
      }
      return data;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setReplayPlaybackError(message);
      console.error(fallbackError, error);
      return null;
    } finally {
      setIsReplayPlaybackProcessing(false);
    }
  };

  const replayStep = async () => {
    const data = await runReplayMutation(
      '/api/replay-step',
      { method: 'POST' },
      'Failed to step replay',
    );
    if (data) {
      setLastReplayStep(data as unknown as ReplayStepResult);
      console.log('Replay step:', data);
    }
  };

  const replayUndo = async () => {
    const data = await runReplayMutation(
      '/api/replay-undo',
      { method: 'POST' },
      'Failed to move to the previous replay step',
    );
    if (data) {
      setLastReplayStep(null);
      console.log('Replay previous:', data);
    }
  };

  const setReplayStep = async (step: number) => {
    const data = await runReplayMutation(
      '/api/replay-goto-sequential',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step }),
      },
      `Failed to go to replay step ${step}`,
    );
    if (data) {
      setLastReplayStep(null);
      console.log('Replay goto:', data);
    }
  };

  const runUntilDrift = async () => {
    const data = await runReplayMutation(
      '/api/replay-goto-divergence',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_steps: 500 }),
      },
      'Failed to run replay until drift',
    );
    if (data) {
      setLastReplayStep(null);
      console.log('Run until drift:', data);
    }
  };

  const updateLiveModel = (model: string) => {
    setLiveModel(model);
    window.localStorage.setItem(LIVE_MODEL_STORAGE_KEY, model);
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
  const activeSavedGame = savedLiveGames.find(
    (game) => game.game_id === liveTraceGameId,
  ) || null;
  const activeLiveReasoningEffort = liveInference?.reasoning.enabled === false
    ? 'off'
    : liveInference?.reasoning.effort || null;

  useEffect(() => {
    if (
      replayMode
      || isReplayPlaybackProcessing
      || selectedSavedGame === null
      || selectedSavedGame.step_count === 0
      || browsedTraceStep?.game_id === selectedSavedGame.game_id
    ) {
      return;
    }
    void browseSavedStep(
      selectedSavedGame.game_id,
      selectedSavedGame.step_count - 1,
    );
  }, [
    browseSavedStep,
    browsedTraceStep?.game_id,
    isReplayPlaybackProcessing,
    replayMode,
    selectedSavedGame,
  ]);

  const isTraceBrowsing = browsedTraceStep !== null && (
    browsedTraceStep.game_id !== liveTraceGameId
    || browsedTraceStep.step.step_index !== browsedTraceStep.latest_step_index
  );

  const toggleAutoPlay = useCallback(() => {
    if (autoPlayRequestedRef.current) {
      stopAutoPlay();
      return;
    }
    if (
      stepInFlightRef.current
      || !hasActiveGame
      || !isRunning
      || replayMode
      || isTraceBrowsing
      || gameState?.winning_color != null
    ) {
      return;
    }

    autoPlayRequestedRef.current = true;
    setIsAutoPlaying(true);
    void runAutoPlayLoop({
      shouldContinue: () => autoPlayRequestedRef.current,
      step: stepGame,
      pause: () => new Promise((resolve) => {
        window.setTimeout(resolve, AUTO_PLAY_STEP_DELAY_MS);
      }),
    }).finally(() => {
      autoPlayRequestedRef.current = false;
      if (appMountedRef.current) {
        setIsAutoPlaying(false);
      }
    });
  }, [
    gameState?.winning_color,
    hasActiveGame,
    isRunning,
    isTraceBrowsing,
    replayMode,
    stepGame,
    stopAutoPlay,
  ]);

  useEffect(() => {
    if (
      autoPlayRequestedRef.current
      && (
        workspace !== 'game'
        || !hasActiveGame
        || !isRunning
        || replayMode
        || isTraceBrowsing
        || gameState?.winning_color != null
      )
    ) {
      stopAutoPlay();
    }
  }, [
    gameState?.winning_color,
    hasActiveGame,
    isRunning,
    isTraceBrowsing,
    replayMode,
    stopAutoPlay,
    workspace,
  ]);

  const hasInspectorContent = gameState !== null && (
    gameLog.length > 0
    || liveStepFailure !== null
    || liveReasoningTraces.length > 0
    || browsedTraceStep !== null
    || replayLlmResponse !== null
    || lastReplayStep !== null
    || tableTalkLog.length > 0
    || Boolean(replayInfo?.paired_transcript)
  );

  const handleSessionPanelResize = useCallback((size: PanelSize) => {
    const isOpen = size.inPixels > 1;
    setIsSessionDrawerOpen((current) => current === isOpen ? current : isOpen);
  }, []);

  const handleInspectorPanelResize = useCallback((size: PanelSize) => {
    const isOpen = size.inPixels > 1;
    setIsInspectorDrawerOpen((current) => current === isOpen ? current : isOpen);
  }, []);

  const toggleSessionDrawer = () => {
    const panel = sessionPanelRef.current;
    if (!panel) return;
    if (panel.isCollapsed()) {
      panel.expand();
    } else {
      panel.collapse();
    }
  };

  const toggleInspectorDrawer = () => {
    const panel = inspectorPanelRef.current;
    if (!panel) return;
    if (panel.isCollapsed()) {
      panel.resize('34%');
    } else {
      panel.collapse();
    }
  };

  const switchWorkspace = (next: 'game' | 'prompt-suite') => {
    if (
      workspace === 'prompt-suite'
      && next !== workspace
      && promptSuiteDirty
      && !window.confirm('Leave Prompt Suite and discard unsaved changes?')
    ) {
      return;
    }
    if (workspace === 'prompt-suite' && next !== workspace) {
      setPromptSuiteDirty(false);
    }
    setWorkspace(next);
  };

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
        <nav className="mode-switch" aria-label="Catan Lab workspace">
          <button
            type="button"
            className={workspace === 'game' ? 'active' : ''}
            onClick={() => switchWorkspace('game')}
          >
            Game
          </button>
          <button
            type="button"
            className={workspace === 'prompt-suite' ? 'active' : ''}
            onClick={() => switchWorkspace('prompt-suite')}
          >
            Prompt Suite{promptSuiteDirty ? ' *' : ''}
          </button>
        </nav>
      </header>

      {workspace === 'game' ? (
        <>
      <Group
        id="catan-game-workspace"
        className="main-container"
        orientation="horizontal"
        defaultLayout={gamePanelLayout.defaultLayout}
        onLayoutChanged={gamePanelLayout.onLayoutChanged}
        resizeTargetMinimumSize={{ fine: 10, coarse: 28 }}
      >
        <Panel
          id="session"
          className="workspace-panel"
          panelRef={sessionPanelRef}
          defaultSize="20%"
          minSize="260px"
          maxSize="28%"
          collapsible
          collapsedSize="0px"
          onResize={handleSessionPanelResize}
        >
          <motion.aside
            className="left-panel"
            initial={false}
            animate={{ opacity: isSessionDrawerOpen ? 1 : 0 }}
          >
            <div className="session-sidebar-header">
              <div>
                <span>Session</span>
                <p>Setup, games, and checkpoints</p>
              </div>
              <button
                type="button"
                onClick={toggleSessionDrawer}
                aria-label="Close session drawer"
              >
                ×
              </button>
            </div>

            <GameControls
              onStartGame={startGame}
              onReset={resetGame}
              onLoadReplay={loadReplay}
              onRunUntilDrift={runUntilDrift}
              onGenerateReplayResponse={generateReplayResponse}
              liveModel={liveModel}
              onLiveModelChange={updateLiveModel}
              replayModel={replayModel}
              onReplayModelChange={updateReplayModel}
              nativeReasoningEffort={nativeReasoningEffort}
              onNativeReasoningEffortChange={updateNativeReasoningEffort}
              liveColorPalette={liveColorPalette}
              onLiveColorPaletteChange={setLiveColorPalette}
              isRunning={hasActiveGame && isRunning}
              hasGame={hasActiveGame}
              isLlmProcessing={isLlmProcessing || isAutoPlaying}
              liveError={liveError}
              liveTraceGameId={liveTraceGameId}
              liveTraceDatabase={liveTraceDatabase}
              isPlaybackProcessing={isReplayPlaybackProcessing}
              replayMode={replayMode}
              isReplayLlmProcessing={isReplayLlmProcessing}
              replayLlmError={replayLlmError}
            />

            {!replayMode && (
              <>
                <SavedLiveGamesBar
                  games={savedLiveGames}
                  selectedGameId={selectedSavedGameId}
                  activeGameId={liveTraceGameId}
                  busy={savedGamesBusy || isLlmProcessing || isAutoPlaying}
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
                  busy={
                    traceBrowseBusy
                    || savedGamesBusy
                    || isLlmProcessing
                    || isAutoPlaying
                  }
                  error={traceBrowseError}
                  onNavigate={browseSavedStep}
                  onLoadLatest={loadSavedGame}
                />
              </>
            )}
          </motion.aside>
        </Panel>

        <Separator
          id="session-board-separator"
          className="workspace-separator"
          aria-label="Resize session drawer and board"
        >
          <span aria-hidden="true" />
        </Separator>

        <Panel
          id="board"
          className="workspace-panel board-workspace-panel"
          defaultSize="46%"
          minSize="420px"
        >
          <div className="board-container">
            <div className="workspace-drawer-actions">
              <button
                type="button"
                className="session-drawer-toggle"
                onClick={toggleSessionDrawer}
                aria-expanded={isSessionDrawerOpen}
              >
                {isSessionDrawerOpen ? 'Hide session' : 'Session'}
              </button>
              <button
                type="button"
                className={`inspector-drawer-toggle ${hasInspectorContent ? 'has-content' : ''}`}
                onClick={toggleInspectorDrawer}
                aria-expanded={isInspectorDrawerOpen}
              >
                Inspect
                {hasInspectorContent && <span aria-label="Inspector has content" />}
              </button>
            </div>

            {gameState && (
              <PlayerInfo
                gameState={gameState}
                allPlayerResources={allPlayerResources}
                allPlayerDevCards={allPlayerDevCards}
                playerTypes={playerTypes}
                replayInfo={replayInfo}
                variant="overlay"
              />
            )}

          {lastDiceRoll && (
            <div style={{
              position: 'absolute',
              top: '74px',
              left: '50%',
              transform: 'translateX(-50%)',
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

            <BoardControlsDock
              hasGame={hasActiveGame}
              isRunning={hasActiveGame && isRunning}
              replayMode={replayMode}
              replayInfo={replayInfo}
              busy={
                replayMode
                  ? isReplayPlaybackProcessing || isReplayLlmProcessing
                  : isLlmProcessing
              }
              error={replayMode ? replayPlaybackError : liveError}
              isTraceBrowsing={isTraceBrowsing}
              liveActor={gameState?.current_color || null}
              liveActorIsAgent={Boolean(
                gameState
                && playerTypes?.[gameState.current_color] === 'LLM'
              )}
              liveModel={
                liveInference?.model
                || activeSavedGame?.config.model
                || liveModel
                || null
              }
              liveReasoningEffort={activeLiveReasoningEffort}
              liveMaxTokens={liveInference?.max_tokens ?? null}
              isAutoPlaying={!replayMode && isAutoPlaying}
              onStep={replayMode
                ? replayStep
                : () => { void stepGame(); }}
              onToggleAutoPlay={toggleAutoPlay}
              onPrevious={replayUndo}
              onSeek={setReplayStep}
            />
          </div>
        </Panel>

        <Separator
          id="board-inspector-separator"
          className="workspace-separator"
          aria-label="Resize board and inspector"
        >
          <span aria-hidden="true" />
        </Separator>

        <Panel
          id="inspector"
          className="workspace-panel"
          panelRef={inspectorPanelRef}
          defaultSize="34%"
          minSize="280px"
          maxSize="58%"
          collapsible
          collapsedSize="0px"
          onResize={handleInspectorPanelResize}
        >
          <motion.aside
            className="right-panel"
            initial={false}
            animate={{
              opacity: isInspectorDrawerOpen ? 1 : 0,
              x: isInspectorDrawerOpen ? 0 : 18,
            }}
            transition={{ duration: 0.16 }}
          >
            <div className="inspector-drawer-header">
              <div>
                <span>Inspector</span>
                <p>Model output and reasoning</p>
              </div>
              <button
                type="button"
                onClick={toggleInspectorDrawer}
                aria-label="Close inspector drawer"
              >
                ×
              </button>
            </div>

            <div className={`inspector-drawer-content ${hasInspectorContent || gameState ? '' : 'empty'}`}>
            {gameState && (
              <details className="inspector-player-details">
                <summary>Full player state</summary>
                <PlayerInfo
                  gameState={gameState}
                  allPlayerResources={allPlayerResources}
                  allPlayerDevCards={allPlayerDevCards}
                  playerTypes={playerTypes}
                  replayInfo={replayInfo}
                />
              </details>
            )}

            {hasInspectorContent ? (
              <>
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

            {replayMode && lastReplayStep && (
              <div className="replay-step-viewer">
                <h3>Last replay step</h3>
                <div>
                  <strong>Engine action</strong>
                  <pre>{lastReplayStep.action}</pre>
                </div>
                <div>
                  <strong>Translation</strong>
                  <pre>{JSON.stringify(lastReplayStep.engine_translation, null, 2)}</pre>
                </div>
                <div>
                  <strong>Colonist event</strong>
                  <pre>{JSON.stringify(lastReplayStep.colonist_event, null, 2)}</pre>
                </div>
              </div>
            )}

            {tableTalkLog.length > 0 && (
              <TableTalkLog entries={tableTalkLog} />
            )}

            {gameLog.length > 0 && (
              <details className="inspector-game-log">
                <summary>
                  <span>Game log</span>
                  <span>{gameLog.length} events</span>
                </summary>
                <GameLog entries={gameLog} showHeading={false} />
              </details>
            )}

            {!replayMode && liveStepFailure && (
              <RejectedLiveAttempts failure={liveStepFailure} />
            )}

            {!replayMode && browsedTraceStep && (
              <SavedStepReasoningTrace detail={browsedTraceStep} />
            )}

            {!replayMode && !browsedTraceStep && liveReasoningTraces.length > 0 && (
              <LiveReasoningTrace traces={liveReasoningTraces} />
            )}
              </>
            ) : (
              <div className="inspector-empty">
                <span>Ready when you are</span>
                <p>Game activity, model output, reasoning, and transcripts appear here.</p>
              </div>
            )}
            </div>
          </motion.aside>
        </Panel>
      </Group>
        </>
      ) : (
        <PromptSuiteStudio
          apiBaseUrl={SERVER_URL}
          hasLoadedGame={liveTraceGameId !== null || replayMode}
          onDirtyChange={setPromptSuiteDirty}
        />
      )}
    </div>
  );
}

export default App;
