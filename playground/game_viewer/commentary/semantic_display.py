"""Compact semantic grouping for causally available transcript evidence."""

import hashlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


_NUMBER_REFERENCE = re.compile(
    r"(?<!\d)(?:[2-9]|1[0-2])(?:\s*(?:-|/|,|\s)\s*(?:[2-9]|1[0-2])){1,2}(?!\d)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")

_COMMITMENT = re.compile(
    r"\b(?:i think (?:it(?:'s| is)|we)|let(?:'s| us)|trust my gut|"
    r"i(?:'m| am) (?:going|gonna)|we(?:'re| are) (?:going|gonna)|"
    r"we(?:'ll| will)|i(?:'ll| will)|i (?:want|like) to)\b",
    re.IGNORECASE,
)
_PLAN = re.compile(
    r"\b(?:best case|worst case|the plan|our plan|we (?:can|could|need|want)|"
    r"play off|build|settle|road|port|expand|route|production)\b",
    re.IGNORECASE,
)
_OPTION = re.compile(
    r"\b(?:versus|instead|alternative|another option|one option|either|or|"
    r"could (?:go|play|take|be)|maybe|interesting|not bad|what if)\b",
    re.IGNORECASE,
)
_OPPONENT = re.compile(
    r"\b(?:opponent|other player|smart (?:man|play)|stream snip|"
    r"he(?:'s| is| can| could| has| wants)|she(?:'s| is| can| could| has)|"
    r"they(?:'re| are| can| could| have)|his |her |their )",
    re.IGNORECASE,
)
_NEGOTIATION = re.compile(
    r"\b(?:trade|offer|counter(?:offer)?|deal|give you|give me|"
    r"for your|for my|bank trade|maritime)\b",
    re.IGNORECASE,
)
_UPDATE = re.compile(
    r"^(?:okay|all right|alright|now|so now|well|unfortunately|fortunately|"
    r"nice|great|perfect|interesting)\b",
    re.IGNORECASE,
)
_BOARD = re.compile(
    r"\b(?:wood|brick|sheep|wheat|ore|robber|settlement|city|road|port|"
    r"dev card|knight|resource|pip|board|roll|seven|six|eight)\b",
    re.IGNORECASE,
)

_STOP_WORDS = frozenset(
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

_ROLE_TITLES = {
    "orientation": "Table context",
    "board_assessment": "Board assessment",
    "option_comparison": "Options and tradeoffs",
    "commitment_plan": "Commitment and plan",
    "opponent_assessment": "Opponent read",
    "negotiation": "Trade discussion",
    "reaction_update": "Plan update",
}

_STRONG_ROLES = frozenset(
    {"commitment_plan", "opponent_assessment", "negotiation", "reaction_update"}
)
_MIN_TRACE_CHARS = 150
_TARGET_TRACE_CHARS = 420
_MAX_TRACE_CHARS = 760
_RECENT_CONTEXT_SECONDS = 45.0
_MAX_VISIBLE_TRACES = 6


def _tokens(text: str) -> Set[str]:
    return {
        token
        for token in _WORD.findall(text.lower())
        if token not in _STOP_WORDS and len(token) > 1
    }


def _utterance_role(text: str) -> str:
    has_reference = bool(_NUMBER_REFERENCE.search(text))
    if _NEGOTIATION.search(text):
        return "negotiation"
    if _OPPONENT.search(text):
        return "opponent_assessment"
    if _COMMITMENT.search(text) or (_PLAN.search(text) and "we" in text.lower()):
        return "commitment_plan"
    if _OPTION.search(text) and (has_reference or _BOARD.search(text)):
        return "option_comparison"
    if has_reference or _BOARD.search(text):
        return "board_assessment"
    if _UPDATE.search(text):
        return "reaction_update"
    return "orientation"


def _dominant_role(utterances: Sequence[Dict[str, Any]]) -> str:
    roles = [_utterance_role(str(utterance["text"])) for utterance in utterances]
    weighted = Counter(roles)
    for role in roles:
        if role in _STRONG_ROLES:
            weighted[role] += 1
        elif role == "option_comparison":
            weighted[role] += 1
        elif role == "orientation":
            weighted[role] -= 1
    return weighted.most_common(1)[0][0]


def _boundary_score(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    previous_role: str,
    current_role: str,
) -> float:
    previous_tokens = _tokens(str(previous["text"]))
    current_tokens = _tokens(str(current["text"]))
    union = previous_tokens | current_tokens
    similarity = len(previous_tokens & current_tokens) / len(union) if union else 0.0
    score = (1.0 - similarity) * 2.0

    if current_role != previous_role:
        score += 1.25
    if current_role in _STRONG_ROLES and current_role != previous_role:
        score += 3.0
    if _UPDATE.search(str(current["text"])):
        score += 1.5
    if _COMMITMENT.search(str(current["text"])) and previous_role == "option_comparison":
        score += 2.5
    return score


def _should_share_boundary(
    previous_role: str,
    current_role: str,
    current_text: str,
) -> bool:
    if (
        previous_role in {"board_assessment", "option_comparison"}
        and current_role == "commitment_plan"
    ):
        return True
    if current_role == "opponent_assessment" and _NUMBER_REFERENCE.search(current_text):
        return True
    return bool(_UPDATE.search(current_text) and _BOARD.search(current_text))


def _partition_utterances(
    utterances: Sequence[Dict[str, Any]],
) -> List[Tuple[List[Dict[str, Any]], bool]]:
    """Partition on semantic shifts, sharing bridge evidence when appropriate."""
    if not utterances:
        return []

    groups: List[Tuple[List[Dict[str, Any]], bool]] = []
    current: List[Dict[str, Any]] = [utterances[0]]
    current_chars = len(str(utterances[0]["text"]))
    current_role = _utterance_role(str(utterances[0]["text"]))
    current_overlaps_previous = False
    candidates: List[Tuple[float, int, str]] = []

    def finish(split_at: int, next_role: str, share: bool) -> None:
        nonlocal current, current_chars, current_role
        nonlocal current_overlaps_previous, candidates
        left = current[:split_at]
        right = current[split_at - 1 :] if share else current[split_at:]
        if left:
            groups.append((left, current_overlaps_previous))
        current = right
        current_chars = sum(len(str(item["text"])) for item in current)
        current_role = next_role if right else "orientation"
        current_overlaps_previous = share
        candidates = []

    for utterance in utterances[1:]:
        text = str(utterance["text"])
        role = _utterance_role(text)
        score = _boundary_score(current[-1], utterance, current_role, role)
        chars_before = current_chars
        current.append(utterance)
        current_chars += len(text)

        semantic_shift = (
            role != current_role
            and role in _STRONG_ROLES
            and chars_before >= _MIN_TRACE_CHARS
        ) or (
            current_role == "orientation"
            and role in {"board_assessment", "option_comparison"}
            and chars_before >= _MIN_TRACE_CHARS
        ) or (
            current_role == "commitment_plan"
            and role == "option_comparison"
            and chars_before >= _MIN_TRACE_CHARS
        )
        if semantic_shift:
            share = _should_share_boundary(current_role, role, text)
            finish(len(current) - (0 if share else 1), role, share)
            if not current:
                current = [utterance]
                current_chars = len(text)
            continue

        if chars_before >= _MIN_TRACE_CHARS:
            candidates.append((score, len(current) - 1, role))

        if current_chars >= _TARGET_TRACE_CHARS and candidates:
            best_score, split_at, next_role = max(candidates, key=lambda item: item[0])
            if best_score >= 3.0 or current_chars >= _MAX_TRACE_CHARS:
                boundary_text = str(current[split_at]["text"])
                share = _should_share_boundary(
                    current_role, next_role, boundary_text
                )
                finish(split_at, next_role, share)

        if role in _STRONG_ROLES or (
            current_role == "orientation"
            and role in {"board_assessment", "option_comparison"}
        ):
            current_role = role

    if current:
        groups.append((current, current_overlaps_previous))
    return groups


def _trace_payload(
    utterances: Sequence[Dict[str, Any]],
    overlaps_previous: bool,
) -> Dict[str, Any]:
    role = _dominant_role(utterances)
    evidence = [
        {
            "start_s": utterance["start_s"],
            "end_s": utterance["end_s"],
            "text": utterance["text"],
            "source_start_index": utterance["source_start_index"],
            "source_end_index": utterance["source_end_index"],
            "source_segment_count": utterance["source_segment_count"],
        }
        for utterance in utterances
    ]
    source_start = min(item["source_start_index"] for item in evidence)
    source_end = max(item["source_end_index"] for item in evidence)
    identity = f"{role}:{source_start}:{source_end}:" + ":".join(
        str(item["source_start_index"]) for item in evidence
    )
    return {
        "trace_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        "role": role,
        "title": _ROLE_TITLES[role],
        "start_s": min(float(item["start_s"]) for item in evidence),
        "end_s": max(float(item["end_s"]) for item in evidence),
        "overlaps_previous": overlaps_previous,
        "text": " ".join(str(item["text"]) for item in evidence),
        "evidence": evidence,
    }


def build_semantic_transcript_traces(
    utterances: Sequence[Dict[str, Any]],
    context_start_s: Optional[float],
    context_end_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return coherent current evidence plus a bounded transition context."""
    selected_utterances = list(utterances)
    if context_start_s is not None:
        context_start = float(context_start_s)
        has_current_evidence = any(
            float(utterance["end_s"]) >= context_start
            for utterance in selected_utterances
        )
        long_empty_interval = (
            context_end_s is not None
            and float(context_end_s) - context_start > _RECENT_CONTEXT_SECONDS
            and not has_current_evidence
        )
        if long_empty_interval:
            return []
        recent_cutoff = max(0.0, context_start - _RECENT_CONTEXT_SECONDS)
        selected_utterances = [
            utterance
            for utterance in selected_utterances
            if float(utterance["end_s"]) >= recent_cutoff
        ]

    traces = [
        _trace_payload(group, overlaps_previous)
        for group, overlaps_previous in _partition_utterances(selected_utterances)
    ]
    return traces[-_MAX_VISIBLE_TRACES:]
