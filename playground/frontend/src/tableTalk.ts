import type { TableTalkEntry } from './types';

export interface TableTalkSourceEntry {
  type: string;
  message: string;
  color?: string;
  step_index?: number | null;
  details?: unknown;
}

const logIndex = (value: unknown): number | null => (
  typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null
);

/**
 * Derive the message board from the speech rows of the game log.
 *
 * Rows carry two different counters. `step_index` is the game step - one
 * /api/step advance, one recorded trace step - and is stamped onto the row
 * only after the step is recorded. `sequence` is the engine-event number,
 * which runs well ahead of the step count because one step emits an action
 * plus every speech and trade-response event inside it. Rows written without
 * a trace store, or before step stamping shipped, keep the sequence as their
 * only label.
 */
export function deriveTableTalk(
  gameLog: readonly TableTalkSourceEntry[],
): TableTalkEntry[] {
  return gameLog
    .filter((entry) => entry.type === 'message')
    .map((entry, index) => {
      const details = (entry.details ?? {}) as {
        sequence?: unknown;
        step_index?: unknown;
      };
      return {
        sequence: logIndex(details.sequence) ?? index,
        step_index: logIndex(entry.step_index) ?? logIndex(details.step_index),
        player: entry.color ?? 'UNKNOWN',
        message: entry.message,
        model: '',
      };
    });
}
