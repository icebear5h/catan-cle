"""Stock-model Cartesian evaluation: no finetuning or added tokenizer tokens.

Only each row's first message is model input. Canonical IDs/receipts are internal.
The scoring API has no Miles dependency and validates gold before delegating.
"""

from .contracts import score_response, validate_metadata
from .dataset import validate_rows
from .geometry import SCHEMA

__all__ = ["SCHEMA", "score_response", "validate_metadata", "validate_rows"]
