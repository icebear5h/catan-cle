import { useRef, useState } from 'react';
import { useDefaultLayout, usePanelRef } from 'react-resizable-panels';
import type { AutoPlayRetryNotice } from '../autoPlay';
import type { LiveStepFailure } from '../liveStepErrors';
import type { AllPlayerDevCards } from '../playerDevCards';
import type { AllPlayerHands } from '../playerHands';
import type {
  GameState,
  Color,
  AllPlayerResources,
  ReplayInfo,
  ReplayLLMResponse,
  LiveColorPalette,
  LiveReasoningTrace as LiveReasoningTraceRecord,
} from '../types';
import type { SavedLiveGameSummary } from '../components/controls/SavedLiveGamesBar';
import type { TraceStepDetail } from '../components/traces/TraceStepNavigator';
import type { GameUsage } from '../traceUsage';
import {
  DEFAULT_LIVE_MODEL,
  DEFAULT_REPLAY_MODEL,
  LIVE_MODEL_STORAGE_KEY,
  REPLAY_MODEL_STORAGE_KEY,
  storedNativeReasoningEffort,
} from './constants';
import type {
  GameLogEntry,
  LiveInferenceState,
  ReplayCursor,
  ReplayStepResult,
} from './sessionTypes';

// Every piece of App state, declared in one place so the hook order of the
// original single component is preserved exactly.
export function useAppState() {
  const [workspace, setWorkspace] = useState<'game' | 'prompt-suite'>('game');
  const [promptSuiteDirty, setPromptSuiteDirty] = useState(false);
  const [runtimeGameState, setGameState] = useState<GameState | null>(null);
  const [hasActiveGame, setHasActiveGame] = useState(false);
  const [runtimeReady, setRuntimeReady] = useState(false);
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
  const [reasoningSelection, setReasoningSelection] = useState<{ gameId: string | null; color: 'all' | Color }>({ gameId: null, color: 'all' });
  const reasoningActorsRef = useRef(new Map<string, Map<number, string[]>>());
  const [reasoningNavigationNotice, setReasoningNavigationNotice] = useState<string | null>(null);
  const [viewingHistory, setViewingHistory] = useState(false);
  const historyRef = useRef(false);
  const [gameUsage, setGameUsage] = useState<GameUsage | null>(null);
  const [usageError, setUsageError] = useState<string | null>(null);
  const [usageRevision, setUsageRevision] = useState(0);
  const [liveReasoningTraces, setLiveReasoningTraces] = useState<LiveReasoningTraceRecord[]>([]);
  const [runtimeGameLog, setGameLog] = useState<GameLogEntry[]>([]);
  const [runtimeResources, setAllPlayerResources] = useState<AllPlayerResources | null>(null);
  const [runtimeDevCards, setAllPlayerDevCards] = useState<AllPlayerDevCards | null>(null);
  const [runtimeHands, setPlayerHands] = useState<AllPlayerHands | null>(null);
  const [isLlmProcessing, setIsLlmProcessing] = useState(false);
  const [isSessionChanging, setIsSessionChanging] = useState(false);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [autoPlayRetry, setAutoPlayRetry] = useState<AutoPlayRetryNotice | null>(null);
  const [runtimePlayerTypes, setPlayerTypes] = useState<Record<string, string> | null>(null);
  const [replayMode, setReplayMode] = useState(false);
  const [replayInfo, setReplayInfo] = useState<ReplayInfo | null>(null);
  const [lastReplayStep, setLastReplayStep] = useState<ReplayStepResult | null>(null);
  const [runtimeDiceRoll, setLastDiceRoll] = useState<[number, number] | null>(null);
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
  const [replayLlmResponse, setReplayLlmResponse] = useState<ReplayLLMResponse | null>(null);
  const [replayLlmError, setReplayLlmError] = useState<string | null>(null);
  const [isReplayLlmProcessing, setIsReplayLlmProcessing] = useState(false);
  const [replayPlaybackError, setReplayPlaybackError] = useState<string | null>(null);
  const [isReplayPlaybackProcessing, setIsReplayPlaybackProcessing] = useState(false);
  const [isSessionDrawerOpen, setIsSessionDrawerOpen] = useState(true);
  const [isInspectorDrawerOpen, setIsInspectorDrawerOpen] = useState(true);
  const replayCursorRef = useRef<ReplayCursor | null>(null);
  const lastSnapshotKeyRef = useRef<string | null>(null);
  const lastListRefreshRef = useRef(0);
  const lastListGameRef = useRef<string | null>(null);
  const replayModelRef = useRef(replayModel);
  const savedGamesRequestRef = useRef(0);
  const traceBrowseRequestRef = useRef(0);
  const autoPlayRequestedRef = useRef(false);
  const autoPlayRetryCancelRef = useRef<(() => void) | null>(null);
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

  return {
    workspace, setWorkspace, promptSuiteDirty, setPromptSuiteDirty,
    runtimeGameState, setGameState, hasActiveGame, setHasActiveGame,
    runtimeReady, setRuntimeReady, isRunning, setIsRunning,
    liveError, setLiveError, liveStepFailure, setLiveStepFailure,
    liveInference, setLiveInference, liveTraceGameId, setLiveTraceGameId,
    liveTraceDatabase, setLiveTraceDatabase, savedLiveGames, setSavedLiveGames,
    selectedSavedGameId, setSelectedSavedGameId, savedGamesBusy, setSavedGamesBusy,
    savedGamesError, setSavedGamesError, browsedTraceStep, setBrowsedTraceStep,
    traceBrowseBusy, setTraceBrowseBusy, traceBrowseError, setTraceBrowseError,
    reasoningSelection, setReasoningSelection, reasoningActorsRef,
    reasoningNavigationNotice, setReasoningNavigationNotice,
    viewingHistory, setViewingHistory, historyRef,
    gameUsage, setGameUsage, usageError, setUsageError, usageRevision, setUsageRevision,
    liveReasoningTraces, setLiveReasoningTraces, runtimeGameLog, setGameLog,
    runtimeResources, setAllPlayerResources, runtimeDevCards, setAllPlayerDevCards,
    runtimeHands, setPlayerHands, isLlmProcessing, setIsLlmProcessing,
    isSessionChanging, setIsSessionChanging, isAutoPlaying, setIsAutoPlaying,
    autoPlayRetry, setAutoPlayRetry, runtimePlayerTypes, setPlayerTypes,
    replayMode, setReplayMode, replayInfo, setReplayInfo,
    lastReplayStep, setLastReplayStep, runtimeDiceRoll, setLastDiceRoll,
    liveModel, setLiveModel, replayModel, setReplayModel,
    nativeReasoningEffort, setNativeReasoningEffort, liveColorPalette, setLiveColorPalette,
    replayLlmResponse, setReplayLlmResponse, replayLlmError, setReplayLlmError,
    isReplayLlmProcessing, setIsReplayLlmProcessing,
    replayPlaybackError, setReplayPlaybackError,
    isReplayPlaybackProcessing, setIsReplayPlaybackProcessing,
    isSessionDrawerOpen, setIsSessionDrawerOpen, isInspectorDrawerOpen, setIsInspectorDrawerOpen,
    replayCursorRef, lastSnapshotKeyRef, lastListRefreshRef, lastListGameRef,
    replayModelRef, savedGamesRequestRef, traceBrowseRequestRef,
    autoPlayRequestedRef, autoPlayRetryCancelRef, stepInFlightRef, appMountedRef,
    sessionPanelRef, inspectorPanelRef, gamePanelLayout,
  };
}

export type AppState = ReturnType<typeof useAppState>;
