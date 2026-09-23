export function displayModel(modelId: string) {
  const name = modelId.split('/').at(-1) || modelId
  return name.replace(':free', '').replace(/-/g, ' ')
}

export function formatReasoning(reasoning: Record<string, unknown>) {
  if (typeof reasoning.effort === 'string') return reasoning.effort
  if (reasoning.enabled === true) return 'enabled'
  return JSON.stringify(reasoning)
}

export function numericUsage(usage: Record<string, unknown>, key: string) {
  const value = usage[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function formatTokens(value: number | null) {
  if (value === null) return '—'
  return new Intl.NumberFormat('en-US').format(value)
}

export function formatDuration(value: number | null) {
  if (value === null) return '—'
  if (value < 1000) return `${Math.round(value)} ms`
  const seconds = value / 1000
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

export function formatCost(value: number | null) {
  if (value === null) return '—'
  return `$${value.toFixed(value < 0.01 ? 6 : 4)}`
}

export function renderContent(content: unknown) {
  return typeof content === 'string' ? content : JSON.stringify(content, null, 2)
}

export function apiError(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object' && 'error' in payload) {
    return String((payload as { error: unknown }).error)
  }
  return fallback
}
