import type { GameState } from '@playground/types'

export const FAMILIES = [
  'relations_joins',
  'sets_coverage',
  'aggregation_comparison',
  'connectivity_structure',
  'constraints_consequences',
] as const

export type Family = typeof FAMILIES[number]
export type ReviewRow = {
  row_id: string
  state_id: string
  family: Family
  operation: string
  question: string
  answer: string
  prompt: string
  source: {
    row_id: string
    line: number
    density: string
    map_id: string
    source_kind: string
  }
}

export type ReviewBundle = {
  schema: 'catan_board_fluency_review/v1'
  title: string
  metadata: {
    row_count: number
    counts_by_family: Record<string, number>
    counts_by_operation: Record<string, number>
    unique_state_count: number
    [key: string]: unknown
  }
  rows: ReviewRow[]
  boards: Record<string, GameState>
}

export type Filters = { family: string; operation: string; query: string }
export const DEFAULT_FILTERS: Filters = { family: 'all', operation: 'all', query: '' }
