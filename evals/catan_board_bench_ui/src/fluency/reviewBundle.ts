import { FAMILIES } from './types'
import type { Filters, ReviewBundle, ReviewRow } from './types'

export function matchingRows(rows: ReviewRow[], filters: Filters) {
  const query = filters.query.trim().toLowerCase()
  return rows.filter((row) => (
    (filters.family === 'all' || row.family === filters.family)
    && (filters.operation === 'all' || row.operation === filters.operation)
    && (!query || [
      row.row_id, row.state_id, row.family, row.operation,
      row.question, row.answer, row.prompt, ...Object.values(row.source),
    ].join('\n').toLowerCase().includes(query))
  ))
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

export function isCounts(value: unknown): value is Record<string, number> {
  return isRecord(value) && Object.values(value).every(
    (count) => typeof count === 'number' && Number.isInteger(count) && count >= 0,
  )
}

export function parseBundle(value: unknown): ReviewBundle {
  if (!isRecord(value) || value.schema !== 'catan_board_fluency_review/v1'
    || typeof value.title !== 'string' || !Array.isArray(value.rows)
    || !isRecord(value.boards) || !isRecord(value.metadata)
    || value.metadata.row_count !== value.rows.length
    || typeof value.metadata.unique_state_count !== 'number'
    || !isCounts(value.metadata.counts_by_family)
    || !isCounts(value.metadata.counts_by_operation)) {
    throw new Error('Invalid Board Fluency preview: expected the catan_board_fluency_review/v1 bundle with matching row_count.')
  }
  const ids = new Set<string>()
  for (const [index, row] of value.rows.entries()) {
    if (!isRecord(row)
      || !['row_id', 'state_id', 'family', 'operation', 'question', 'answer', 'prompt']
        .every((key) => typeof row[key] === 'string')
      || !row.row_id || ids.has(String(row.row_id))
      || !FAMILIES.some((family) => family === row.family)
      || !isRecord(row.source) || !Number.isInteger(row.source.line)
      || !['row_id', 'density', 'map_id', 'source_kind'].every((key) => (
        isRecord(row.source) && typeof row.source[key] === 'string'
      ))) {
      throw new Error(`Invalid Board Fluency preview: malformed or duplicate row at position ${index + 1}.`)
    }
    ids.add(String(row.row_id))
  }
  return value as ReviewBundle
}
