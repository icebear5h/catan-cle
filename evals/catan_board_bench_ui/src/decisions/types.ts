export type StageId = 'setup' | 'early' | 'mid' | 'late' | 'unknown';
export type AgreementFilter = 'all' | 'agree' | 'disagree' | 'unscored';
export type Verdict = 'strong' | 'reasonable' | 'questionable' | 'blunder';

export type StageRule = {
  id: Exclude<StageId, 'unknown'>;
  label: string;
  description: string;
  detection: string;
};

export type BucketDefinition = {
  id: string;
  label: string;
  group: string;
  description: string;
  review_unit: string;
  target_samples: number;
  detection: string;
  rubric: string[];
};

export type ReviewLabel = {
  id: string;
  label: string;
  description: string;
};

export type VerdictDefinition = ReviewLabel;

export type BucketCatalog = {
  id: string;
  version: string;
  schema: string;
  stage_rules: StageRule[];
  critical_rule: {
    label: string;
    description: string;
    detection: string;
  };
  buckets: BucketDefinition[];
  review_labels: ReviewLabel[];
  verdicts: VerdictDefinition[];
  sampling: {
    default_target_per_bucket: number;
    random_fraction: number;
    reasoning_disagreement_fraction: number;
    high_stakes_fraction: number;
    include_model_agreements: boolean;
  };
};

export type DecisionRun = {
  id: string;
  title: string;
  description: string;
  model_id: string;
  game_id: string;
  target_player_id: number | null;
  target_engine_color: string | null;
  available: boolean;
};

export type CompactRun = {
  schema: string;
  id: string;
  title: string;
  description: string;
  game_id: string;
  model_id: string;
  target_player_id: number;
  target_engine_color: string;
  decision_count: number;
  bucketed_decision_count: number;
  response_count: number;
  disagreement_count: number;
  bucket_suite: {
    id: string;
    version: string;
    schema: string;
  };
};

export type BucketCount = {
  bucket_id: string;
  decision_count: number;
  episode_count: number;
  response_count: number;
  disagreement_count: number;
};

export type HumanSelection = {
  action_index: number;
  action: string;
  description: string;
  normalization?: string;
} | null;

export type CompactModelSelection = {
  model_id: string;
  response_present: boolean;
  action_index: number | null;
  action: string | null;
  description: string | null;
  agreement: boolean | null;
  error: unknown;
  parse_error: string | null;
};

export type CompactDecision = {
  decision_id: string;
  game_id: string;
  replay_index: number;
  source_event_index: number | null;
  action_type: string;
  classification: string;
  forced: boolean;
  stage: StageId;
  critical: boolean;
  bucket_ids: string[];
  episode_ids: Record<string, string>;
  actor: Record<string, unknown>;
  human: HumanSelection;
  model: CompactModelSelection;
};

export type DecisionListResponse = {
  run: CompactRun;
  bucket_counts: BucketCount[];
  stage_counts: Record<string, number>;
  total: number;
  offset: number;
  limit: number;
  decisions: CompactDecision[];
};

export type DetailedModelSelection = CompactModelSelection & {
  response_source: string | null;
  recorded_at: string | null;
  game_plan: string;
  rationale: string;
  rationale_source: string | null;
  native_reasoning: string;
  native_reasoning_details: unknown[];
  reasoning_request: Record<string, unknown> | null;
  reasoning_tokens: number | null;
  native_reasoning_returned: boolean | null;
  usage: Record<string, unknown>;
  finish_reason: string | null;
  provider_native_finish_reason: string | null;
  provider_response_id: string | null;
  provider_request_id: string | null;
  latency_ms: number | null;
  context_version: string | null;
  context_prompt: string;
  system_prompt: string;
  raw_response: string;
};

export type DetailedDecision = Omit<CompactDecision, 'model'> & {
  source_replay_index: number | null;
  reason: string | null;
  phase: string | null;
  bucket_evidence: Record<string, unknown>;
  state_features: Record<string, unknown> | null;
  available_actions: Array<{
    index: number;
    action: string;
    description: string;
  }>;
  model: DetailedModelSelection;
};

export type DecisionDetailResponse = {
  run: CompactRun;
  decision: DetailedDecision;
};
