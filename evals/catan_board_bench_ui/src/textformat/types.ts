export type FormatScore = {
  format: string
  exact: number
  requests: number
  exact_accuracy: number
}

export type BoardSummary = {
  sample_id: string
  question_count: number
  categories: string[]
  characters: number
  lines: number
}

export type FormatOverview = {
  format_id: string
  display_name: string
  model: string
  board_count: number
  question_count: number
  boards: BoardSummary[]
  question_lock: {
    byte_identical_to_source: boolean
    sha256: string
  }
  development: {
    baseline: FormatScore
    winner: FormatScore
    prompt_token_multiplier: number
    character_multiplier: number
  }
  transfer: {
    complete: boolean
    paired_questions: number
    winner: FormatScore
  }
}

export type FormatQuestion = {
  id: string
  category: string
  question: string
  answer: Record<string, unknown>
  answer_text: string
}

export type FormatSample = {
  sample_id: string
  version: string
  text: string
  characters: number
  lines: number
  sha256: string
  questions: FormatQuestion[]
}

export type NumberedLine = { line: string; number: number }
export type TextView = 'full' | 'base' | 'indexes'
