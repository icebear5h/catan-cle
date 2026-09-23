import type { FormatScore } from './types'

export function formatScore(score: FormatScore) {
  return (score.exact_accuracy * 100).toFixed(1) + '%'
}

export function apiError(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object' && 'error' in payload) {
    return String((payload as { error: unknown }).error)
  }
  return fallback
}
