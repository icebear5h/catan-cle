import { useEffect } from 'react';
import { io } from 'socket.io-client';
import { SERVER_URL } from './constants';
import { snapshotKey } from './api';
import type { StateSnapshot } from './sessionTypes';
import type { AppState } from './useAppState';
import type { SnapshotSync } from './useSnapshotSync';
import type { TraceBrowsing } from './useTraceBrowsing';

// Subscribes to server-pushed game_state snapshots for the App's lifetime.
export function useGameSocket(state: AppState, sync: SnapshotSync, browsing: TraceBrowsing) {
  const {
    lastSnapshotKeyRef, historyRef, lastListGameRef, lastListRefreshRef,
    setRuntimeReady, setHasActiveGame, setLiveTraceGameId, setSelectedSavedGameId,
    setBrowsedTraceStep, setUsageRevision,
  } = state;
  const { applyStateSnapshot } = sync;
  const { refreshSavedGames } = browsing;

  useEffect(() => {
    const newSocket = io(SERVER_URL);

    newSocket.on('connect', () => {
      console.log('Connected to game server');
    });

    newSocket.on('game_state', (data) => {
      const snapshot = data as StateSnapshot;
      setRuntimeReady(true);
      const key = snapshotKey(snapshot);
      if (key === lastSnapshotKeyRef.current) {
        return;
      }
      lastSnapshotKeyRef.current = key;
      const isRuntimeSnapshot = snapshot.game !== null && (
        Boolean(snapshot.replay_mode)
        || snapshot.live_trace_game_id !== undefined
      );
      setHasActiveGame(isRuntimeSnapshot);
      if (!snapshot.replay_mode && snapshot.live_trace_game_id !== undefined) {
        setLiveTraceGameId(snapshot.live_trace_game_id);
        if (!historyRef.current && snapshot.live_trace_game_id) {
          setSelectedSavedGameId(snapshot.live_trace_game_id);
          setBrowsedTraceStep((previous) => previous?.game_id === snapshot.live_trace_game_id ? previous : null);
        }
      }

      console.log(
        '[FRONTEND] game_state',
        snapshot.replay_mode ? snapshot.replay?.game_id : snapshot.live_trace_game_id,
        'idx',
        snapshot.game?.state_index,
        'log',
        snapshot.game_log?.length || 0,
      );

      applyStateSnapshot(snapshot);
      // The saved-games list + usage follow-ups are ~400KB of fetch/parse per
      // event. Refresh immediately on game switch, otherwise throttle.
      const now = Date.now();
      const gameId = snapshot.live_trace_game_id ?? null;
      if (gameId !== lastListGameRef.current || now - lastListRefreshRef.current >= 2500) {
        lastListGameRef.current = gameId;
        lastListRefreshRef.current = now;
        void refreshSavedGames(false);
        setUsageRevision((value) => value + 1);
      }
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
    });

    return () => {
      newSocket.close();
    };
  }, [applyStateSnapshot, refreshSavedGames, lastSnapshotKeyRef, historyRef, lastListGameRef,
    lastListRefreshRef, setRuntimeReady, setHasActiveGame, setLiveTraceGameId,
    setSelectedSavedGameId, setBrowsedTraceStep, setUsageRevision]);
}
