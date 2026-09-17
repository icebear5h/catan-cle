export interface UsageCall {
  step_index?: number;
  call_index: number;
  call_kind: string;
  accepted: boolean | number;
  failure_id?: string;
  usage?: Record<string, unknown> | null;
}

export interface GameUsage {
  game_id: string;
  step_count: number;
  calls: UsageCall[];
  failure_calls: UsageCall[];
}

function tokenCount(usage: UsageCall['usage'], aliases: string[]): number | null {
  for (const alias of aliases) {
    const value = usage?.[alias];
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) return value;
  }
  return null;
}

export function sumUsage(calls: UsageCall[]) {
  const unique = [...new Map(calls.map((call) => [
    JSON.stringify([call.failure_id ?? null, call.step_index ?? null, call.call_kind, call.call_index]),
    call,
  ])).values()];
  const sum = (aliases: string[]) => {
    const values = unique.map((call) => tokenCount(call.usage, aliases));
    const known = values.filter((value): value is number => value !== null);
    return { total: known.length ? known.reduce((a, b) => a + b, 0) : null, covered: known.length };
  };
  // Cache and reasoning details are subsets of these provider totals, not additions.
  return {
    input: sum(['prompt_tokens', 'input_tokens']),
    output: sum(['completion_tokens', 'output_tokens']),
    calls: unique.length,
    accepted: unique.filter((call) => Boolean(call.accepted)).length,
  };
}

export function averageUsage(game: GameUsage) {
  const steps = new Map<number, UsageCall[]>();
  for (const call of game.calls) {
    if (call.step_index === undefined) continue;
    const batch = steps.get(call.step_index) ?? [];
    batch.push(call);
    steps.set(call.step_index, batch);
  }
  const totals = [...steps.values()].map(sumUsage);
  const average = (side: 'input' | 'output') => {
    const covered = totals.filter((step) => step[side].total !== null);
    return {
      total: covered.length
        ? covered.reduce((sum, step) => sum + step[side].total!, 0) / covered.length : null,
      covered: covered.length,
    };
  };
  return { input: average('input'), output: average('output'), coverage: sumUsage(game.calls) };
}
