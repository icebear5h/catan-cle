"""Shared structural types for the symbolic board dataset builder."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

from sft.board.symbolic_board_tasks._types import StatePayload
from sft.json_types import JsonDict, JsonLike

# A `{"kind": ..., "value": ...}` near-selector, as it appears inside a query.
SelectorDict: TypeAlias = dict[str, str]
# A task query bag; heterogeneous but always JSON-serializable into a row.
QueryDict: TypeAlias = dict[str, JsonLike]
# A component sampling slot: mode, roster position and polarity.
SlotDict: TypeAlias = dict[str, str | int | None]
# The grouping key `_balanced` sorts on; always a tuple within one call.
BucketKey: TypeAlias = tuple[str | int | bool, ...]


class SourceRecord(TypedDict):
    """One audited source state with its detached provenance."""

    state: StatePayload
    provenance: JsonDict
