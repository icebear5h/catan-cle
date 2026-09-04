export interface AutoPlayStepResult {
  ok: boolean;
  running: boolean;
  gameOver: boolean;
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
}

export async function runAutoPlayLoop({
  shouldContinue,
  step,
  pause,
}: AutoPlayLoopOptions): Promise<AutoPlayStopReason> {
  while (shouldContinue()) {
    let result: AutoPlayStepResult;
    try {
      result = await step();
    } catch {
      return 'error';
    }

    if (!result.ok) {
      return 'error';
    }
    if (result.gameOver) {
      return 'game_over';
    }
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
