export type SpatialTarget = {
  token: string
  entity_type: string
  bbox: [number, number, number, number]
  center: [number, number]
  control_bbox: [number, number, number, number]
  control_token: string
}

export type SpatialRow = {
  index: number
  record_id: string
  row_id: string
  state_id: string
  image_name: string
  image_url: string
  prompt: string
  answer: string
  curriculum_stage: string
  grounding_stage: string
  task_type: string
  entity_type: string
  target_token: string | null
  queried_token: string | null
  piece: string | null
  color: string | null
  density_bin: string | null
  tokens: string[]
  marker: string | null
  marker_style: string | null
  marker_group: string[]
  relationship: string
  polarity: string
  sampling_repeat: number | null
  replay_source: string | null
  probe_style: string | null
  eval_variant: string | null
  spatial_target: SpatialTarget | null
}

export type SpatialSummary = {
  row_count: number
  state_count: number
  image_count: number
  source_schema: string
  source_sha256: string
  atlas_counts: Record<string, number>
  atlas_tokens: number
  marker_groups_per_board: number
  node_edge_sampling_multiplier: number
  stage2_marker_replay_fraction: number
}

export type StageCatalogRow = {
  id: string
  splits: Record<string, number>
  train_rows: number
}

export type SpatialPayload = {
  dataset: string
  datasets: { id: string; label: string }[]
  stage: string
  split: string
  rows: SpatialRow[]
  total: number
  offset: number
  limit: number
  summary: SpatialSummary
  stages: StageCatalogRow[]
  distributions: {
    task_type: Record<string, number>
    entity: Record<string, number>
    relationship: Record<string, number>
    polarity: Record<string, number>
  }
  facets: {
    stages: string[]
    splits: string[]
    task_types: string[]
    entity_types: string[]
    relationships: string[]
    polarities: string[]
  }
}

export type Filters = {
  dataset: string
  stage: string
  split: string
  taskType: string
  entityType: string
  relationship: string
  polarity: string
  query: string
}
