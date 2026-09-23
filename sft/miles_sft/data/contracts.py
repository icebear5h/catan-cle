"""Lightweight contracts for immutable, text-only Miles SFT preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from sft.json_types import JsonDict

STATIC_OPERATIONS = frozenset({
    "symbolic_direction", "symbolic_direction_choice", "symbolic_neighbors",
    "symbolic_incidence", "symbolic_oriented_step",
})
SCHEMA = "catan_miles_sft_data/v1"
ChatMessage = dict[str, str]


class ChatTokenizer(Protocol):
    """The saved native tokenizer, without importing Transformers or a trainer."""

    @property
    def chat_template(self) -> str | dict[str, str] | None: ...

    @property
    def name_or_path(self) -> str: ...

    @property
    def all_special_ids(self) -> list[int]: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def apply_chat_template(
        self, conversation: list[ChatMessage], *, tokenize: bool,
        add_generation_prompt: bool, return_dict: bool, enable_thinking: bool,
        preserve_thinking: bool, truncation: bool,
    ) -> list[int]: ...


class EncodedPair(TypedDict):
    """Unshifted complete tokens and a RESPONSE-SUFFIX mask (Miles convention).

    ``len(loss_mask) == response_length == len(tokens) - prefix_length``.
    The full-sequence equivalent is ``[0] * prefix_length + loss_mask``;
    causal shifting belongs to the trainer, not dataset preparation.
    """

    tokens: list[int]
    prefix_length: int
    response_length: int
    loss_mask: list[int]


class CorpusInspection(TypedDict):
    source: str
    source_sha256: str
    total_rows: int
    admitted_rows: int
    excluded_rows: int
    admitted_by_operation: dict[str, int]
    excluded_by_operation: dict[str, int]
    excluded_by_reason: dict[str, int]
    by_split: dict[str, int]


class PrepareReceipt(TypedDict):
    schema: str
    source: str
    output: str
    input_path: str
    metadata_path: str
    source_sha256: str
    input_sha256: str
    rows: int
    limited_rows: int
    selected_by_operation: dict[str, int]
    corpus: CorpusInspection
    tokenizer_identity: str
    chat_template_sha256: str
    max_tokens: int
    limit: int | None
    total_tokens: int
    supervised_tokens: int
    longest_sequence: int


@dataclass(frozen=True)
class SourceRow:
    row_id: str
    operation: str
    messages: list[ChatMessage]
    metadata: JsonDict
    line_number: int
    row_sha256: str
