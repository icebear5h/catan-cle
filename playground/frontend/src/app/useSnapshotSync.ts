import { useCallback, useEffect } from 'react';
import { liveStepErrorUpdate, liveStepFailureRetryable } from '../liveStepErrors';
import type { TraceStepDetail } from '../components/traces/TraceStepNavigator';
import { matchingReasoningActors } from '../components/traces/traceModelCalls';
import { snapshotKey } from './api';
import type { StateSnapshot } from './sessionTypes';
import type { AppState } from './useAppState';

// Applies authoritative server snapshots and live-step errors to App state.
export function useSnapshotSync(state: AppState) {
  const {
    reasoningActorsRef, autoPlayRequestedRef, autoPlayRetryCancelRef, appMountedRef,
    lastSnapshotKeyRef, replayCursorRef, traceBrowseRequestRef,
    setIsAutoPlaying, setLiveError, setLiveStepFailure, setGameState, setIsRunning,
    setGameLog, setLiveInference, setAllPlayerResources, setAllPlayerDevCards,
    setPlayerHands, setPlayerTypes, setTraceBrowseBusy, setBrowsedTraceStep,
    setSelectedSavedGameId, setLiveTraceGameId, setLiveTraceDatabase,
    setReplayLlmResponse, setReplayLlmError, setReplayMode, setReplayInfo, setLastDiceRoll,
  } = state;

  const cacheReasoningActors = useCallback((detail: TraceStepDetail) => {
    const cache = reasoningActorsRef.current;
    if (!cache.has(detail.game_id)) cache.set(detail.game_id, new Map());
    cache.get(detail.game_id)!.set(detail.step.step_index, matchingReasoningActors(detail));
    if (cache.size > 3) cache.delete(cache.keys().next().value!);
  }, [reasoningActorsRef]);

  const stopAutoPlay = useCallback(() => {
    autoPlayRequestedRef.current = false;
    // A pending retry backoff ends now so the loop observes the cancellation.
    autoPlayRetryCancelRef.current?.();
    setIsAutoPlaying(false);
  }, [autoPlayRequestedRef, autoPlayRetryCancelRef, setIsAutoPlaying]);

  useEffect(() => {
    appMountedRef.current = true;
    return () => {
      appMountedRef.current = false;
      autoPlayRequestedRef.current = false;
    };
  }, [appMountedRef, autoPlayRequestedRef]);

  const applyLiveStepError = useCallback((
    payload: unknown,
    source: 'runtime' | 'checkpoint' = 'runtime',
  ) => {
    const update = liveStepErrorUpdate(payload, source);
    if (update === undefined) return;
    // Auto-play retries every checkpointed failure, from this tab or another;
    // only a failure the server could not save halts it.
    if (update.message !== null && !liveStepFailureRetryable(payload)) stopAutoPlay();
    setLiveError(update.message);
    setLiveStepFailure(update.failure);
  }, [stopAutoPlay, setLiveError, setLiveStepFailure]);

  const applyStateSnapshot = useCallback((
    data: StateSnapshot,
    source: 'runtime' | 'checkpoint' = 'runtime',
  ) => {
    lastSnapshotKeyRef.current = snapshotKey(data);
    setGameState(data.game);
    setIsRunning(data.running);
    setGameLog(data.game_log || []);
    if (data.live_inference !== undefined) {
      setLiveInference(data.live_inference);
    }
    applyLiveStepError(data.last_live_step_error, source);
    setAllPlayerResources(data.all_player_resources || null);
    setAllPlayerDevCards(data.all_player_dev_cards || null);
    setPlayerHands(data.player_hands || null);
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
      setReplayLlmResponse(null);
      setReplayLlmError(null);
    } else {
      const changedGame = previousCursor?.gameId !== nextReplay.game_id;
      const changedCursor = previousCursor?.eventIndex !== nextReplay.event_index;

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
  }, [applyLiveStepError, lastSnapshotKeyRef, replayCursorRef, traceBrowseRequestRef,
    setGameState, setIsRunning, setGameLog, setLiveInference, setAllPlayerResources,
    setAllPlayerDevCards, setPlayerHands, setPlayerTypes, setTraceBrowseBusy,
    setBrowsedTraceStep, setSelectedSavedGameId, setLiveTraceGameId, setLiveTraceDatabase,
    setReplayLlmResponse, setReplayLlmError, setReplayMode, setReplayInfo, setLastDiceRoll]);

  return { cacheReasoningActors, stopAutoPlay, applyLiveStepError, applyStateSnapshot };
}

export type SnapshotSync = ReturnType<typeof useSnapshotSync>;
