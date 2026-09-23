import type { DetailedModelSelection } from './types';

export function nativeReasoningMetadata(model: DetailedModelSelection) {
  const parts: string[] = [];
  if (model.reasoning_tokens != null) {
    parts.push(`${model.reasoning_tokens.toLocaleString()} tokens`);
  }
  if (model.reasoning_request) {
    parts.push(JSON.stringify(model.reasoning_request));
  }
  return parts.join(' · ') || 'separate provider channel';
}

export function apiError(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object' && 'error' in payload) {
    return String((payload as { error: unknown }).error);
  }
  return fallback;
}

export function humanize(value: string) {
  return value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());
}
