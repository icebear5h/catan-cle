import type { StateSnapshot } from './sessionTypes';

// Identity of a state snapshot for echo-dedupe: the POST-response state and the
// socket broadcast carry the same payload, so the second delivery can be skipped.
// A broadcast that changes only the failure notice, inference settings or seat
// types (a failure in another tab, a prompt swap) must still get through.
export function snapshotKey(snapshot: StateSnapshot): string {
  const game = snapshot.game;
  return JSON.stringify([
    snapshot.replay_mode
      ? `replay:${snapshot.replay?.game_id}:${snapshot.replay?.event_index}`
      : `live:${snapshot.live_trace_game_id}`,
    game?.state_index ?? null,
    snapshot.running,
    snapshot.game_log?.length ?? 0,
    snapshot.last_dice_roll ?? null,
    snapshot.last_live_step_error ?? null,
    snapshot.live_inference ?? null,
    snapshot.player_types ?? null,
  ]);
}

export function getApiError(payload: unknown, fallback: string): string {
  if (typeof payload !== 'object' || payload === null) {
    return fallback;
  }
  const record = payload as Record<string, unknown>;
  for (const key of ['details', 'message', 'error'] as const) {
    const value = record[key];
    if (typeof value === 'string' && value.trim()) {
      return value;
    }
  }
  return fallback;
}

export async function readApiObject(response: Response): Promise<Record<string, unknown>> {
  const text = await response.text();
  try {
    const payload: unknown = JSON.parse(text);
    if (typeof payload === 'object' && payload !== null && !Array.isArray(payload)) {
      return payload as Record<string, unknown>;
    }
  } catch {
    // The error below includes the HTTP status and a bounded response preview.
  }
  const contentType = response.headers.get('content-type') || 'unknown content type';
  const preview = text.trim().replace(/\s+/g, ' ').slice(0, 160);
  throw new Error(
    `API ${response.status} returned non-JSON (${contentType}): ${preview || 'empty body'}`,
  );
}
