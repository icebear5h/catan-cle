"""Run schemas, packet limits, tool definitions, and the system prompt."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

RUN_SCHEMA = "narrator-observation-assembly-run-v1"
ATTEMPT_SCHEMA = "narrator-observation-assembly-attempt-v1"
RESULT_SCHEMA = "narrator-observation-assembly-result-v1"
PROMPT_VERSION = "narrator-observation-assembly-v1"
DEFAULT_MODEL = "openai/gpt-5.6-sol"
MAX_PACKET_UTTERANCES = 24
MAX_PACKET_SECONDS = 45.0
MAX_PARAGRAPHS = 8
MAX_PARAGRAPH_CHARS = 1_200
PARAGRAPH_KINDS = frozenset(
    {
        "decision_reasoning",
        "board_observation",
        "opponent_assessment",
        "reaction",
        "reflection",
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECISION_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
    / "action_selection_diff/qwen3_8_27b_blue_20260817"
)

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_TOOL_DEFINITIONS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_board",
            "description": (
                "Inspect the authoritative public Catan board at this causal "
                "decision or observation cursor. Hidden hands and the next recorded "
                "action are unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_location",
            "description": (
                "Resolve one number-based location phrase that occurs verbatim in "
                "the supplied caption evidence. Ambiguity is preserved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "Exact location phrase from the captions.",
                    }
                },
                "required": ["reference"],
                "additionalProperties": False,
            },
        },
    },
]

_SYSTEM_PROMPT = f"""You are an editorial assembly agent for human Catan commentary.
Turn the supplied noisy YouTube ASR into coherent first-person reasoning for one
strictly causal decision or public-observation packet. The packet may span many replay
actions because human thoughts do not stop at engine row boundaries.

Rules ({PROMPT_VERSION}):
- Call inspect_board before answering. Use inspect_location only for a number phrase
  that appears in the supplied evidence.
- Use only this packet's captions, already-visible public events, and public board.
  The action at this cursor, future actions, and all hidden hands are unavailable.
- Assemble related sentences across replay boundaries. Remove rolling-caption
  repetition, filler, greetings, false starts, and verbal tics.
- Preserve uncertainty, alternatives, mistakes, and changes of mind. Do not improve
  the strategy or infer a chosen move, resource hand, route, visual pointer, or result.
- `decision_reasoning` is allowed only when the captions explicitly deliberate the
  narrator's current decision opportunity. Opponent comments and general table reads
  are observations, not narrator action rationale.
- Every evidence ID must appear exactly once: cite it in one paragraph, or put it in
  `omitted_evidence_ids` when it is filler, repetition, unrelated chatter, or too
  unclear to reconstruct safely.
- Return prose, not bullets. Return at most {MAX_PARAGRAPHS} paragraphs.

Allowed paragraph kinds: {", ".join(sorted(PARAGRAPH_KINDS))}.

Return only this JSON shape:
{{
  "paragraphs": [
    {{
      "kind": "decision_reasoning",
      "text": "one coherent first-person paragraph",
      "evidence_ids": ["e0001", "e0002"],
      "uncertainties": ["brief unresolved point, if any"]
    }}
  ],
  "omitted_evidence_ids": ["e0003"]
}}"""

