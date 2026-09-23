import { useCallback } from 'react';
import type { AutoPlayStepResult } from '../autoPlay';
import { liveStepAutoPlayResult, liveStepFailureRetryable } from '../liveStepErrors';
import type { LiveReasoningTrace as LiveReasoningTraceRecord } from '../types';
import { DEFAULT_LIVE_MODEL, SERVER_URL, nativeReasoningRequest } from './constants';
import { getApiError, readApiObject } from './api';
import type { StateSnapshot } from './sessionTypes';
import type { AppState } from './useAppState';
import type { SnapshotSync } from './useSnapshotSync';
import type { TraceBrowsing } from './useTraceBrowsing';

// Starting a live sandbox game and advancing it one engine step.
export function useLiveGameActions(state: AppState, sync: SnapshotSync, browsing: TraceBrowsing) {
  const {
    isSessionChanging, hasActiveGame, liveModel, liveColorPalette, nativeReasoningEffort,
    traceBrowseBusy, savedGamesBusy,
    stepInFlightRef, historyRef, traceBrowseRequestRef, replayCursorRef, appMountedRef,
    setIsSessionChanging, setViewingHistory, setTraceBrowseBusy, setBrowsedTraceStep,
    setTraceBrowseError, setLiveReasoningTraces, setLiveTraceGameId, setLiveTraceDatabase,
    setSelectedSavedGameId, setReplayLlmResponse, setReplayLlmError, setReplayPlaybackError,
    setHasActiveGame, setLiveError, setIsLlmProcessing, setAutoPlayRetry, setUsageRevision,
  } = state;
  const { applyLiveStepError, applyStateSnapshot } = sync;
  const { browseSavedStep, refreshSavedGames } = browsing;

  const startGame = async (mode: string = 'random') => {
    if (isSessionChanging || hasActiveGame || stepInFlightRef.current) return;
    setIsSessionChanging(true);
    historyRef.current = false;
    setViewingHistory(false);
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
    setReplayLlmResponse(null);
    setReplayLlmError(null);
    setReplayPlaybackError(null);
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
    } finally {
      setIsSessionChanging(false);
    }
  };

  const stepGame = useCallback(async (): Promise<AutoPlayStepResult> => {
    if (stepInFlightRef.current || historyRef.current || traceBrowseBusy || isSessionChanging || savedGamesBusy) {
      // Busy flags clear on their own; browsing history is a deliberate pause.
      return { ok: false, running: false, gameOver: false, retryable: !historyRef.current };
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
      console.log(
        `[FRONTEND] Step game - response received (${(duration / 1000).toFixed(3)}s)`,
        'trace_step_index',
        (data as { trace_step_index?: unknown }).trace_step_index,
      );

      if (!response.ok) {
        applyLiveStepError(data);
        return {
          ok: false, running: false, gameOver: false, retryable: liveStepFailureRetryable(data),
        };
      }
      if (!data.state) {
        throw new Error('Live step response did not include authoritative state');
      }

      const snapshot = data.state as unknown as StateSnapshot;
      applyStateSnapshot(snapshot);
      setAutoPlayRetry(null);
      setTraceBrowseError(null);
      const traceGameId = typeof data.trace_game_id === 'string'
        ? data.trace_game_id
        : null;
      const traceStepIndex = typeof data.trace_step_index === 'number'
        ? data.trace_step_index
        : null;
      if (traceGameId !== null && traceStepIndex !== null) {
        setSelectedSavedGameId(traceGameId);
        await browseSavedStep(traceGameId, traceStepIndex, true);
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
      // The server or network may be back on the next attempt.
      return { ok: false, running: false, gameOver: false, retryable: true };
    } finally {
      setUsageRevision((value) => value + 1);
      stepInFlightRef.current = false;
      if (appMountedRef.current) {
        setIsLlmProcessing(false);
      }
    }
  }, [applyLiveStepError, applyStateSnapshot, browseSavedStep, refreshSavedGames,
    traceBrowseBusy, isSessionChanging, savedGamesBusy,
    stepInFlightRef, historyRef, traceBrowseRequestRef, appMountedRef,
    setLiveError, setIsLlmProcessing, setAutoPlayRetry, setTraceBrowseError,
    setSelectedSavedGameId, setTraceBrowseBusy, setBrowsedTraceStep,
    setLiveReasoningTraces, setUsageRevision]);

  return { startGame, stepGame };
}

export type LiveGameActions = ReturnType<typeof useLiveGameActions>;
