"""Fresh, train-only static-atlas SFT data and native response-suffix masks."""

from .contracts import (
    STATIC_OPERATIONS,
    ChatMessage,
    ChatTokenizer,
    CorpusInspection,
    EncodedPair,
    PrepareReceipt,
)
from .encoding import encode_pair
from .prepare import inspect_corpus, prepare_dataset

__all__ = [
    "STATIC_OPERATIONS",
    "ChatMessage",
    "ChatTokenizer",
    "CorpusInspection",
    "EncodedPair",
    "PrepareReceipt",
    "encode_pair",
    "inspect_corpus",
    "prepare_dataset",
]
