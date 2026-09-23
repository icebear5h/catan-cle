export type Metric = { correct: number; exact_accuracy: number; total: number }

export type EvalRow = {
  index: number
  id: string
  state_id: string
  image_url: string
  prompt: string
  expected: string
  response: string
  correct: boolean
  scoring: string
  category: string
  density_bin: string
  entity_type: string
  row_kind: string
  suite: string
  relationship: string | null
  polarity: string | null
  task_type: string | null
}

export type EvalSummary = {
  attempted: number
  correct: number
  exact_accuracy: number
  model_id: string
  bits: number
  batch_size: number
  generated_at: string
  by_category: Record<string, Metric>
  by_density_bin: Record<string, Metric>
  by_polarity: Record<string, Metric>
  by_relationship: Record<string, Metric>
  by_row_kind: Record<string, Metric>
  by_suite: Record<string, Metric>
  spatial_token_return: Metric
}

export type EvalPayload = {
  rows: EvalRow[]
  total: number
  offset: number
  limit: number
  summary: EvalSummary
  checkpoint: { name: string; hub_url: string; modal_url: string }
  facets: {
    categories: string[]
    density_bins: string[]
    row_kinds: string[]
    suites: string[]
  }
}

export type Filters = {
  category: string
  densityBin: string
  rowKind: string
  suite: string
  correctness: string
  query: string
}
