import { useCallback, useEffect } from 'react';
import type { Color } from '../types';
import type { SavedLiveGameSummary } from '../components/controls/SavedLiveGamesBar';
import type { TraceStepDetail } from '../components/traces/TraceStepNavigator';
import { matchingReasoningActors } from '../components/traces/traceModelCalls';
import { SERVER_URL } from './constants';
import { getApiError, readApiObject } from './api';
import type { AppState } from './useAppState';
import type { CheckpointView } from './useCheckpointView';
import type { SnapshotSync } from './useSnapshotSync';

// Saved-game listing and checkpoint/reasoning navigation over saved traces.
export function useTraceBrowsing(state: AppState, view: CheckpointView, sync: SnapshotSync) {
  const {
    savedGamesRequestRef, traceBrowseRequestRef, stepInFlightRef, historyRef, reasoningActorsRef,
    liveTraceGameId, savedLiveGames,
    setSavedGamesBusy, setSavedGamesError, setSavedLiveGames, setSelectedSavedGameId,
    setViewingHistory, setTraceBrowseBusy, setTraceBrowseError, setReasoningNavigationNotice,
    setBrowsedTraceStep, setReasoningSelection,
  } = state;
  const { reasoningFilter, reasoningGameRef, reasoningGameId } = view;
  const { stopAutoPlay, cacheReasoningActors } = sync;

  const refreshSavedGames = useCallback(async (selectDefault = true) => {
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
          : selectDefault ? games[0]?.game_id || '' : ''
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
  }, [savedGamesRequestRef, setSavedGamesBusy, setSavedGamesError, setSavedLiveGames,
    setSelectedSavedGameId]);

  useEffect(() => {
    void refreshSavedGames();
  }, [refreshSavedGames]);

  const browseSavedStep = useCallback(async (gameId: string, stepIndex: number, followLive = false) => {
    if (!followLive && stepInFlightRef.current) return;
    if (!followLive) stopAutoPlay();
    historyRef.current = !followLive;
    setViewingHistory(!followLive);
    const requestId = ++traceBrowseRequestRef.current;
    try {
      setTraceBrowseBusy(true);
      setTraceBrowseError(null);
      setReasoningNavigationNotice(null);
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
      cacheReasoningActors(detail);
      setBrowsedTraceStep(detail);
      setSelectedSavedGameId(gameId);
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
  }, [stopAutoPlay, cacheReasoningActors, stepInFlightRef, historyRef, traceBrowseRequestRef,
    setViewingHistory, setTraceBrowseBusy, setTraceBrowseError, setReasoningNavigationNotice,
    setBrowsedTraceStep, setSelectedSavedGameId]);

  const navigateReasoning = async (gameId: string, startIndex: number, direction: -1 | 1, stepCount: number) => {
    if (reasoningFilter === 'all') {
      void browseSavedStep(gameId, startIndex + direction);
      return;
    }
    if (stepInFlightRef.current) return;
    stopAutoPlay();
    const requestId = ++traceBrowseRequestRef.current;
    setTraceBrowseBusy(true);
    setTraceBrowseError(null);
    setReasoningNavigationNotice(null);
    try {
      for (let index = startIndex + direction; index >= 0 && index < stepCount; index += direction) {
        if (requestId !== traceBrowseRequestRef.current || reasoningGameRef.current !== gameId) return;
        const cachedActors = reasoningActorsRef.current.get(gameId)?.get(index);
        if (cachedActors && !cachedActors.includes(reasoningFilter)) continue;
        const response = await fetch(`${SERVER_URL}/api/live-traces/${gameId}/steps/${index}`, { cache: 'no-store' });
        const data = await readApiObject(response);
        if (requestId !== traceBrowseRequestRef.current || reasoningGameRef.current !== gameId) return;
        if (!response.ok) throw new Error(getApiError(data, 'Failed to search saved reasoning'));
        const detail = data as unknown as TraceStepDetail;
        cacheReasoningActors(detail);
        if (!matchingReasoningActors(detail).includes(reasoningFilter)) continue;
        historyRef.current = true;
        setViewingHistory(true);
        setBrowsedTraceStep(detail);
        setSelectedSavedGameId(gameId);
        return;
      }
      setReasoningNavigationNotice(`No ${direction === -1 ? 'earlier' : 'later'} reasoning for ${reasoningFilter}.`);
    } catch (error) {
      if (requestId === traceBrowseRequestRef.current) {
        setTraceBrowseError(error instanceof Error ? error.message : String(error));
      }
    } finally {
      if (requestId === traceBrowseRequestRef.current) setTraceBrowseBusy(false);
    }
  };

  const returnLatest = useCallback(() => {
    if (stepInFlightRef.current) return;
    traceBrowseRequestRef.current += 1;
    setTraceBrowseBusy(false);
    setTraceBrowseError(null);
    setReasoningNavigationNotice(null);
    historyRef.current = false;
    setViewingHistory(false);
    setBrowsedTraceStep(null);
    setSelectedSavedGameId(liveTraceGameId ?? '');
    void refreshSavedGames(false);
  }, [liveTraceGameId, refreshSavedGames, stepInFlightRef, traceBrowseRequestRef, historyRef,
    setTraceBrowseBusy, setTraceBrowseError, setReasoningNavigationNotice, setViewingHistory,
    setBrowsedTraceStep, setSelectedSavedGameId]);

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

  const selectReasoningFilter = (color: 'all' | Color) => {
    traceBrowseRequestRef.current += 1;
    setTraceBrowseBusy(false);
    setTraceBrowseError(null);
    setReasoningNavigationNotice(null);
    setReasoningSelection({ gameId: reasoningGameId, color });
  };

  return {
    refreshSavedGames, browseSavedStep, navigateReasoning, returnLatest,
    selectSavedGame, selectReasoningFilter,
  };
}

export type TraceBrowsing = ReturnType<typeof useTraceBrowsing>;
