import { useCallback, useEffect } from 'react';
import { autoPlayRetryDelayMs, runAutoPlayLoop } from '../autoPlay';
import { AUTO_PLAY_STEP_DELAY_MS } from './constants';
import type { AppState } from './useAppState';
import type { CheckpointView } from './useCheckpointView';
import type { SnapshotSync } from './useSnapshotSync';
import type { LiveGameActions } from './useLiveGameActions';

// The auto-play loop toggle, stopped whenever the live game is no longer steppable.
export function useAutoPlay(
  state: AppState,
  view: CheckpointView,
  sync: SnapshotSync,
  live: LiveGameActions,
) {
  const {
    workspace, viewingHistory, hasActiveGame, isRunning, replayMode,
    traceBrowseBusy, savedGamesBusy, isSessionChanging,
    autoPlayRequestedRef, stepInFlightRef, autoPlayRetryCancelRef, appMountedRef,
    setIsAutoPlaying, setAutoPlayRetry,
  } = state;
  const { gameState } = view;
  const { stopAutoPlay } = sync;
  const { stepGame } = live;

  const isTraceBrowsing = viewingHistory;

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
      || traceBrowseBusy || savedGamesBusy || isSessionChanging
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
      retryPause: (failures) => new Promise<void>((resolve) => {
        const delayMs = autoPlayRetryDelayMs(failures);
        const finish = () => {
          window.clearTimeout(timer);
          autoPlayRetryCancelRef.current = null;
          if (appMountedRef.current) {
            setAutoPlayRetry({ attempt: failures, resumeAt: null });
          }
          resolve();
        };
        const timer = window.setTimeout(finish, delayMs);
        autoPlayRetryCancelRef.current = finish;
        setAutoPlayRetry({ attempt: failures, resumeAt: Date.now() + delayMs });
      }),
    }).finally(() => {
      autoPlayRequestedRef.current = false;
      autoPlayRetryCancelRef.current = null;
      if (appMountedRef.current) {
        setIsAutoPlaying(false);
        setAutoPlayRetry(null);
      }
    });
  }, [
    gameState?.winning_color,
    hasActiveGame,
    isRunning,
    isTraceBrowsing,
    traceBrowseBusy, savedGamesBusy, isSessionChanging,
    replayMode,
    stepGame,
    stopAutoPlay,
    autoPlayRequestedRef, stepInFlightRef, autoPlayRetryCancelRef, appMountedRef,
    setIsAutoPlaying, setAutoPlayRetry,
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
    autoPlayRequestedRef,
  ]);

  return { isTraceBrowsing, toggleAutoPlay };
}

export type AutoPlayControls = ReturnType<typeof useAutoPlay>;
