import type { GameState } from '@playground/types';

export type CompactExample = {
  id: string;
  sample_id: string;
  category: string;
  question: string;
  answer: string;
};

export type QuestionDetail = CompactExample & {
  target: Record<string, unknown>;
  scoring: string | null;
  contract_context: Record<string, unknown>;
};

export type SampleResponse = {
  sample_id: string;
  sample_index: number;
  sample_count: number;
  image_path: string;
  image_url: string;
  contract_path: string;
  annotations_url: string;
  render_state: GameState;
  contract_context: Record<string, unknown>;
  questions: QuestionDetail[];
};

export type CategoryScore = {
  attempted: number;
  requests: number;
  errors: number;
  exact_accuracy: number;
  component_accuracy: number;
  avg_latency_ms: number;
};

export type EvalModelSummary = {
  model: string;
  display_name: string;
  attempted: number;
  exact_accuracy: number;
  component_accuracy: number;
  avg_latency_ms: number;
  errors: number;
  categories: Record<string, CategoryScore>;
};

export type EvalComparisonResponse = {
  requested: string[];
  categories: string[];
  models: EvalModelSummary[];
};

export type ExampleListResponse = {
  examples: CompactExample[];
  total: number;
  categories: string[];
  samples: string[];
  sample_counts: Record<string, number>;
};

export type Verdict = 'correct' | 'wrong' | 'unclear';
