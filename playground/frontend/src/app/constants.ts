import type { NativeReasoningEffort } from '../types';

// Toggle between servers: 5001 (mixed) or 5002 (4-LLM)
export const SERVER_URL = 'http://127.0.0.1:5001';  // Main server
export const LIVE_MODEL_STORAGE_KEY = 'catan-lab.live-model';
export const REPLAY_MODEL_STORAGE_KEY = 'catan-lab.replay-model';
export const NATIVE_REASONING_STORAGE_KEY = 'catan-lab.native-reasoning-effort-v2';
export const DEFAULT_LIVE_MODEL = 'qwen/qwen3.8-27b';
export const DEFAULT_REPLAY_MODEL = 'qwen/qwen3.8-27b';
export const DEFAULT_NATIVE_REASONING_EFFORT: NativeReasoningEffort = 'high';
export const AUTO_PLAY_STEP_DELAY_MS = 750;
export const NATIVE_REASONING_EFFORTS = new Set<NativeReasoningEffort>([
  'off',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
]);

export function storedNativeReasoningEffort(): NativeReasoningEffort {
  const stored = window.localStorage.getItem(NATIVE_REASONING_STORAGE_KEY);
  return stored && NATIVE_REASONING_EFFORTS.has(stored as NativeReasoningEffort)
    ? stored as NativeReasoningEffort
    : DEFAULT_NATIVE_REASONING_EFFORT;
}

export function nativeReasoningRequest(effort: NativeReasoningEffort): Record<string, unknown> {
  return effort === 'off'
    ? { enabled: false }
    : { effort, exclude: false };
}
