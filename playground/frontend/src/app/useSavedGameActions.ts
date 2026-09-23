import { SERVER_URL } from './constants';
import { getApiError, readApiObject } from './api';
import type { StateSnapshot } from './sessionTypes';
import type { AppState } from './useAppState';
import type { SnapshotSync } from './useSnapshotSync';
import type { TraceBrowsing } from './useTraceBrowsing';

// Clearing the sandbox and naming or resuming saved live games.
export function useSavedGameActions(state: AppState, sync: SnapshotSync, browsing: TraceBrowsing) {
  const {
    isSessionChanging, isReplayPlaybackProcessing, isReplayLlmProcessing, savedGamesBusy,
    liveTraceGameId, stepInFlightRef, traceBrowseRequestRef, historyRef,
    setIsSessionChanging, setTraceBrowseBusy, setBrowsedTraceStep, setTraceBrowseError,
    setLiveReasoningTraces, setLiveTraceGameId, setLiveTraceDatabase, setSelectedSavedGameId,
    setLastReplayStep, setReplayPlaybackError, setHasActiveGame, setViewingHistory,
    setLiveError, setSavedGamesBusy, setSavedGamesError,
  } = state;
  const { stopAutoPlay, applyLiveStepError, applyStateSnapshot } = sync;
  const { refreshSavedGames, returnLatest } = browsing;

  const resetGame = async () => {
    if (isSessionChanging || stepInFlightRef.current || isReplayPlaybackProcessing
      || isReplayLlmProcessing || savedGamesBusy) return;
    setIsSessionChanging(true);
    stopAutoPlay();
    try {
      const response = await fetch(`${SERVER_URL}/api/reset`, {
        method: 'POST',
      });
      const data = await readApiObject(response);
      if (!response.ok) {
        throw new Error(getApiError(data, 'Failed to clear game'));
      }
      traceBrowseRequestRef.current += 1;
      setTraceBrowseBusy(false);
      setBrowsedTraceStep(null);
      setTraceBrowseError(null);
      setLiveReasoningTraces([]);
      setLiveTraceGameId(null);
      setLiveTraceDatabase(null);
      setSelectedSavedGameId('');
      setLastReplayStep(null);
      setReplayPlaybackError(null);
      setHasActiveGame(false);
      historyRef.current = false;
      setViewingHistory(false);
      applyStateSnapshot({
        game: null,
        running: false,
        game_log: [],
        all_player_resources: null,
        all_player_dev_cards: null,
        player_hands: null,
        player_types: null,
        replay_mode: false,
        replay: null,
        last_dice_roll: null,
        live_trace_game_id: null,
        live_inference: null,
        last_live_step_error: null,
      });
      void refreshSavedGames(false);
      console.log('Game reset:', data);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLiveError(message);
      console.error('Error resetting game:', error);
    } finally {
      setIsSessionChanging(false);
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
    if (gameId === liveTraceGameId) { returnLatest(); return; }
    if (stepInFlightRef.current) return;
    stopAutoPlay();
    historyRef.current = false;
    setViewingHistory(false);
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

  return { resetGame, renameSavedGame, loadSavedGame };
}

export type SavedGameActions = ReturnType<typeof useSavedGameActions>;
