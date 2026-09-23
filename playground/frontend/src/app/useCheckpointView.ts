import { useEffect, useMemo, useRef } from 'react';
import { deriveTableTalk } from '../tableTalk';
import type { TableTalkEntry } from '../types';
import type { StateSnapshot } from './sessionTypes';
import type { AppState } from './useAppState';

// The board shows a browsed checkpoint while viewing history, otherwise the
// runtime snapshot; the reasoning filter resets whenever the traced game changes.
export function useCheckpointView(state: AppState) {
  const {
    viewingHistory, browsedTraceStep, runtimeGameState, runtimeGameLog,
    runtimeResources, runtimeDevCards, runtimeHands, runtimePlayerTypes, runtimeDiceRoll,
    replayMode, selectedSavedGameId, liveTraceGameId,
    reasoningSelection, setReasoningSelection, setReasoningNavigationNotice,
  } = state;
  const checkpoint = viewingHistory && browsedTraceStep
    ? browsedTraceStep.step.public_state as StateSnapshot : null;
  const gameState = checkpoint ? checkpoint.game : runtimeGameState;
  const gameLog = checkpoint ? checkpoint.game_log ?? [] : runtimeGameLog;
  // Message board: derive table talk from game-log message rows (agent speech
  // logged by _log_table_talk). The old replay-response accumulator never fires
  // (ReplayLLMResponse no longer carries a message), so derive instead.
  const tableTalkLog: TableTalkEntry[] = useMemo(
    () => deriveTableTalk(gameLog),
    [gameLog],
  );
  const allPlayerResources = checkpoint ? checkpoint.all_player_resources ?? null : runtimeResources;
  const allPlayerDevCards = checkpoint ? checkpoint.all_player_dev_cards ?? null : runtimeDevCards;
  const playerHands = checkpoint ? checkpoint.player_hands ?? null : runtimeHands;
  const playerTypes = checkpoint ? checkpoint.player_types ?? null : runtimePlayerTypes;
  const lastDiceRoll = checkpoint ? checkpoint.last_dice_roll ?? null : runtimeDiceRoll;
  const reasoningGameId = replayMode ? null : selectedSavedGameId || liveTraceGameId;
  const reasoningFilter = reasoningSelection.gameId === reasoningGameId ? reasoningSelection.color : 'all';
  if (reasoningSelection.gameId !== reasoningGameId) {
    setReasoningSelection({ gameId: reasoningGameId, color: 'all' });
    setReasoningNavigationNotice(null);
  }
  const reasoningGameRef = useRef(reasoningGameId);
  useEffect(() => {
    reasoningGameRef.current = reasoningGameId;
  }, [reasoningGameId]);

  return {
    gameState, gameLog, tableTalkLog, allPlayerResources, allPlayerDevCards,
    playerHands, playerTypes, lastDiceRoll, reasoningGameId, reasoningFilter, reasoningGameRef,
  };
}

export type CheckpointView = ReturnType<typeof useCheckpointView>;
