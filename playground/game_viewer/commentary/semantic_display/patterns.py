"""The lexical evidence: role regexes, stop words, titles, and size limits."""

import re

__all__ = [
    "BOARD",
    "COMMITMENT",
    "MAX_TRACE_CHARS",
    "MAX_VISIBLE_TRACES",
    "MIN_TRACE_CHARS",
    "NEGOTIATION",
    "NUMBER_REFERENCE",
    "OPPONENT",
    "OPTION",
    "PLAN",
    "RECENT_CONTEXT_SECONDS",
    "ROLE_TITLES",
    "STOP_WORDS",
    "STRONG_ROLES",
    "TARGET_TRACE_CHARS",
    "UPDATE",
    "WORD",
]

NUMBER_REFERENCE = re.compile(
    r"(?<!\d)(?:[2-9]|1[0-2])(?:\s*(?:-|/|,|\s)\s*(?:[2-9]|1[0-2])){1,2}(?!\d)",
    re.IGNORECASE,
)
WORD = re.compile(r"[a-z0-9]+")

COMMITMENT = re.compile(
    r"\b(?:i think (?:it(?:'s| is)|we)|let(?:'s| us)|trust my gut|"
    r"i(?:'m| am) (?:going|gonna)|we(?:'re| are) (?:going|gonna)|"
    r"we(?:'ll| will)|i(?:'ll| will)|i (?:want|like) to)\b",
    re.IGNORECASE,
)
PLAN = re.compile(
    r"\b(?:best case|worst case|the plan|our plan|we (?:can|could|need|want)|"
    r"play off|build|settle|road|port|expand|route|production)\b",
    re.IGNORECASE,
)
OPTION = re.compile(
    r"\b(?:versus|instead|alternative|another option|one option|either|or|"
    r"could (?:go|play|take|be)|maybe|interesting|not bad|what if)\b",
    re.IGNORECASE,
)
OPPONENT = re.compile(
    r"\b(?:opponent|other player|smart (?:man|play)|stream snip|"
    r"he(?:'s| is| can| could| has| wants)|she(?:'s| is| can| could| has)|"
    r"they(?:'re| are| can| could| have)|his |her |their )",
    re.IGNORECASE,
)
NEGOTIATION = re.compile(
    r"\b(?:trade|offer|counter(?:offer)?|deal|give you|give me|"
    r"for your|for my|bank trade|maritime)\b",
    re.IGNORECASE,
)
UPDATE = re.compile(
    r"^(?:okay|all right|alright|now|so now|well|unfortunately|fortunately|"
    r"nice|great|perfect|interesting)\b",
    re.IGNORECASE,
)
BOARD = re.compile(
    r"\b(?:wood|brick|sheep|wheat|ore|robber|settlement|city|road|port|"
    r"dev card|knight|resource|pip|board|roll|seven|six|eight)\b",
    re.IGNORECASE,
)

STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "all",
        "also",
        "am",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "but",
        "by",
        "can",
        "could",
        "do",
        "for",
        "from",
        "get",
        "go",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "him",
        "his",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "just",
        "like",
        "me",
        "my",
        "of",
        "on",
        "one",
        "or",
        "our",
        "really",
        "she",
        "so",
        "some",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "they",
        "this",
        "to",
        "up",
        "us",
        "was",
        "we",
        "what",
        "when",
        "which",
        "with",
        "would",
        "you",
        "your",
    }
)

ROLE_TITLES: dict[str, str] = {
    "orientation": "Table context",
    "board_assessment": "Board assessment",
    "option_comparison": "Options and tradeoffs",
    "commitment_plan": "Commitment and plan",
    "opponent_assessment": "Opponent read",
    "negotiation": "Trade discussion",
    "reaction_update": "Plan update",
}

STRONG_ROLES: frozenset[str] = frozenset(
    {"commitment_plan", "opponent_assessment", "negotiation", "reaction_update"}
)
MIN_TRACE_CHARS = 150
TARGET_TRACE_CHARS = 420
MAX_TRACE_CHARS = 760
RECENT_CONTEXT_SECONDS = 45.0
MAX_VISIBLE_TRACES = 6
