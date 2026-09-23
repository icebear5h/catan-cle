export function percent(value: number) { return `${(value * 100).toFixed(1)}%` }
export function formatInteger(value: number) { return new Intl.NumberFormat('en-US').format(value) }
export function cleanPrompt(value: string) { return value.replace(/^<image>\s*/i, '') }
export function humanize(value: string) { return value.replaceAll('_', ' ').replaceAll('.', ' · ') }
export function apiError(payload: unknown, fallback: string) {
  return payload && typeof payload === 'object' && 'error' in payload ? String(payload.error) : fallback
}
