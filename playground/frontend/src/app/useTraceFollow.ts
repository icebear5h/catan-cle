import { useEffect } from 'react';
import type { GameUsage } from '../traceUsage';
import { SERVER_URL } from './constants';
import { getApiError, readApiObject } from './api';
import type { AppState } from './useAppState';
import type { TraceBrowsing } from './useTraceBrowsing';

// Selected/active saved games, their usage projection, and keeping the
// browsed checkpoint pinned to the latest saved step.
export function useTraceFollow(state: AppState, browsing: TraceBrowsing) {
  const {
    savedLiveGames, selectedSavedGameId, liveTraceGameId, liveInference, replayMode,
    usageRevision, runtimeReady, isSessionChanging, isReplayPlaybackProcessing, isLlmProcessing,
    traceBrowseError, traceBrowseBusy, browsedTraceStep, viewingHistory, historyRef,
    setUsageError, setGameUsage,
  } = state;
  const { browseSavedStep } = browsing;

  const selectedSavedGame = savedLiveGames.find(
    (game) => game.game_id === selectedSavedGameId,
  ) || null;
  const activeSavedGame = savedLiveGames.find(
    (game) => game.game_id === liveTraceGameId,
  ) || null;
  const activeLiveReasoningEffort = liveInference?.reasoning.enabled === false
    ? 'off'
    : liveInference?.reasoning.effort || null;

  const usageGameId = selectedSavedGameId || liveTraceGameId;
  useEffect(() => {
    if (!usageGameId || replayMode) return;
    const controller = new AbortController();
    setUsageError(null);
    void fetch(`${SERVER_URL}/api/live-traces/${usageGameId}?view=usage`, {
      signal: controller.signal, cache: 'no-store',
    }).then(async (response) => {
      const data = await readApiObject(response);
      if (!response.ok) throw new Error(getApiError(data, 'Failed to load usage'));
      if (!Array.isArray(data.calls) || !Array.isArray(data.failure_calls)) {
        throw new Error('Server does not support the usage projection yet');
      }
      if (!controller.signal.aborted) setGameUsage(data as unknown as GameUsage);
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setGameUsage(null);
        setUsageError(error instanceof Error ? error.message : String(error));
      }
    });
    return () => controller.abort();
  }, [usageGameId, usageRevision, replayMode, setUsageError, setGameUsage]);

  useEffect(() => {
    if (
      !runtimeReady
      || replayMode
      || isSessionChanging
      || isReplayPlaybackProcessing
      || isLlmProcessing
      || traceBrowseError !== null
      || selectedSavedGame === null
      || selectedSavedGame.step_count === 0
      || traceBrowseBusy
      || (browsedTraceStep?.game_id === selectedSavedGame.game_id && (
        viewingHistory || browsedTraceStep.step.step_index >= selectedSavedGame.step_count - 1
      ))
    ) {
      return;
    }
    void browseSavedStep(
      selectedSavedGame.game_id,
      selectedSavedGame.step_count - 1,
      !historyRef.current && selectedSavedGame.game_id === liveTraceGameId,
    );
  }, [
    browseSavedStep,
    runtimeReady,
    browsedTraceStep,
    liveTraceGameId,
    traceBrowseBusy,
    traceBrowseError,
    isLlmProcessing,
    viewingHistory,
    isReplayPlaybackProcessing,
    isSessionChanging,
    replayMode,
    selectedSavedGame,
    historyRef,
  ]);

  return { selectedSavedGame, activeSavedGame, activeLiveReasoningEffort, usageGameId };
}

export type TraceFollow = ReturnType<typeof useTraceFollow>;
