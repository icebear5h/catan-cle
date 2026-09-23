import type { CSSProperties } from 'react'
import type { SpatialRow, SpatialTarget } from './explorerTypes'

export function bboxStyle(bbox: [number, number, number, number]): CSSProperties {
  return {
    left: `${bbox[0] * 100}%`,
    top: `${bbox[1] * 100}%`,
    width: `${(bbox[2] - bbox[0]) * 100}%`,
    height: `${(bbox[3] - bbox[1]) * 100}%`,
  }
}

export function rowKind(row: SpatialRow) {
  if (row.piece && !['TILE', 'PORT'].includes(row.piece.toUpperCase())) return 'piece'
  if (['tile', 'port'].includes(row.entity_type) && !row.marker) return row.entity_type
  if (row.probe_style) return 'probe'
  if (row.marker || row.replay_source) return 'marked'
  return 'relation'
}

export function supervisionLabel(row: SpatialRow) {
  if (row.piece) return humanize(row.task_type)
  if (row.task_type === 'marker_to_token') return 'visual marker → atlas token'
  if (row.task_type === 'token_to_marker') return 'atlas token → visual marker'
  if (row.task_type === 'neutral_probe_token_return') return 'novel gray dot → atlas token'
  if (row.polarity === 'token_return') return 'spatial relation → atlas token'
  return `${humanize(row.relationship)} relation → ${row.polarity === 'hard_negative' ? 'no' : 'yes'}`
}

export function markerLabel(row: SpatialRow) {
  if (row.probe_style) return humanize(row.probe_style)
  if (!row.marker) return 'none'
  return `${row.marker} · ${humanize(row.marker_style || 'unknown style')}`
}

export function patchLabel(target: SpatialTarget | null) {
  return target ? `${target.token} · [${target.bbox.map(formatCoord).join(', ')}]` : 'not annotated in this row'
}

export function controlLabel(target: SpatialTarget | null) {
  return target?.control_bbox ? `${target.control_token} · [${target.control_bbox.map(formatCoord).join(', ')}]` : 'not annotated in this row'
}

export function atlasNote(counts: Record<string, number>) {
  return Object.entries(counts).map(([key, value]) => `${value} ${key}`).join(' · ')
}

export function cleanPrompt(prompt: string) { return prompt.replace(/^<image>\s*/i, '') }
export function humanize(value: string) { return value.replaceAll('_', ' ') }
export function formatInteger(value: number) { return new Intl.NumberFormat('en-US').format(value) }
export function formatCompact(value: number) { return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value) }
export function formatCoord(value: number) { return value.toFixed(3) }
export function apiError(payload: unknown, fallback: string) {
  if (payload && typeof payload === 'object' && 'error' in payload) return String(payload.error)
  return fallback
}
