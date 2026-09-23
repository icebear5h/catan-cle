"""Run schemas, prompt version, tool definitions, and the system prompt."""

from __future__ import annotations

import re
from typing import Dict, List

RUN_SCHEMA = "narrator-reasoning-run-v1"
ATTEMPT_SCHEMA = "narrator-reasoning-attempt-v1"
RESULT_SCHEMA = "narrator-reasoning-result-v1"
PROMPT_VERSION = "narrator-reasoning-v2"
DEFAULT_MODEL = "openai/gpt-5.6-sol"
MAX_PARAGRAPHS = 3
MAX_PARAGRAPH_CHARS = 1_200

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_TOOL_DEFINITIONS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_board",
            "description": (
                "Inspect the authoritative public Catan board and public event context "
                "at this exact pre-action transcript cursor. Hidden hands and the next "
                "recorded action are unavailable."
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
                "Resolve one number-based location phrase that appears verbatim in the "
                "caption evidence, such as '8 4 10'. Ambiguity is preserved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "Exact location phrase from the supplied captions.",
                    }
                },
                "required": ["reference"],
                "additionalProperties": False,
            },
        },
    },
]

_SYSTEM_PROMPT = f"""You are an editorial reconstruction agent for human Catan commentary.
Your job is to turn noisy YouTube ASR captions into faithful, coherent first-person
reasoning paragraphs. You are reconstructing what the narrator expressed, not writing
a stronger strategy and not predicting the next replay action.

Rules ({PROMPT_VERSION}):
- Call inspect_board before answering. Use inspect_location for number-based board
  references when its result can clarify the wording.
- Use only supplied captions and tool facts from this exact cursor. The next recorded
  action and all future events are unavailable.
- Remove greetings, filler, repeated rolling-caption text, false starts, and verbal
  tics. Repair grammar and sentence boundaries.
- Preserve the narrator's uncertainty, alternatives, commitments, and mistakes. Do
  not silently improve the strategy or invent motives, resources, routes, players,
  visual pointers, or outcomes.
- Resolve phrases such as "this spot" only when the captions plus tool evidence make
  the referent unique. Otherwise keep the uncertainty explicit or omit the unsupported
  referent.
- Do not turn opponent speculation into fact. Use engine colors only when identity is
  supported; otherwise retain a neutral reference such as "another player."
- Return zero paragraphs when the captions contain only greetings, filler, reactions,
  or unrelated chatter.
- Each paragraph must be prose, not bullets, and must cite the caption evidence IDs it
  actually reconstructs. Return at most {MAX_PARAGRAPHS} paragraphs.

Return only this JSON shape:
{{
  "paragraphs": [
    {{
      "text": "one coherent first-person paragraph",
      "evidence_ids": ["u0", "u1"],
      "uncertainties": ["brief unresolved point, if any"]
    }}
  ]
}}"""

