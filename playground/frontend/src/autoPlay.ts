export interface AutoPlayStepResult {
  ok: boolean;
  running: boolean;
  gameOver: boolean;
  /** Another Step may be requested after this failure: nothing irreversible blocks it. */
  retryable?: boolean;
}

export type AutoPlayStopReason =
  | 'cancelled'
  | 'error'
  | 'game_over'
  | 'stopped';

interface AutoPlayLoopOptions {
  shouldContinue: () => boolean;
  step: () => Promise<AutoPlayStepResult>;
  pause: () => Promise<void>;
  /**
   * Waits before requesting a failed step again, given the consecutive failure
   * count (from 1). Retries are unbounded; only cancellation, a non-retryable
   * failure, the game ending or the sandbox stopping end the loop. Omitted, the
   * loop stops on the first failure.
   */
  retryPause?: (failures: number) => Promise<void>;
}

export async function runAutoPlayLoop({
  shouldContinue,
  step,
  pause,
  retryPause,
}: AutoPlayLoopOptions): Promise<AutoPlayStopReason> {
  let failures = 0;
  while (shouldContinue()) {
    let result: AutoPlayStepResult;
    try {
      result = await step();
    } catch {
      return 'error';
    }

    if (result.gameOver) {
      return 'game_over';
    }
    if (!result.ok) {
      if (result.retryable !== true || retryPause === undefined) {
        return 'error';
      }
      failures += 1;
      if (!shouldContinue()) {
        return 'cancelled';
      }
      await retryPause(failures);
      continue;
    }
    failures = 0;
    if (!result.running) {
      return 'stopped';
    }
    if (!shouldContinue()) {
      return 'cancelled';
    }

    await pause();
  }

  return 'cancelled';
}

export interface AutoPlayRetryNotice {
  /** Consecutive failed steps auto-play is recovering from. */
  attempt: number;
  /** Epoch milliseconds when the next request starts; null while it is in flight. */
  resumeAt: number | null;
}

const AUTO_PLAY_RETRY_BASE_MS = 1000;
const AUTO_PLAY_RETRY_MAX_MS = 30000;

/** Exponential backoff from 1s, capped at 30s: transport hiccups clear quickly and a stuck provider is not hammered. */
export function autoPlayRetryDelayMs(failures: number): number {
  const exponent = Math.max(0, Math.min(failures - 1, 30));
  return Math.min(AUTO_PLAY_RETRY_BASE_MS * 2 ** exponent, AUTO_PLAY_RETRY_MAX_MS);
}
