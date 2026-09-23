"""Structural contracts shared by replay parsing and the replay runtime.

The mutable runtime object is duck-typed on purpose: the Flask viewer's
``ServerState`` and headless eval states both drive the same executor. These
protocols pin the attribute surface the replay package actually touches without
importing either concrete owner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypedDict, runtime_checkable

from cle.game_engine.game import GameEngine
from cle.replay.colonist.types import ActionHint, TradeLedgerRecord

if TYPE_CHECKING:
    from cle.replay.runtime.checkpoint import ReplayStepCheckpoint

#: A decoded Colonist replay archive (``parsed_actions``, ``events``, ...).
ReplayArchive: TypeAlias = Mapping[str, object]
#: A JSON-ish response body handed back to the viewer/eval callers.
ReplayPayload: TypeAlias = dict[str, object]
#: Either a success body or a ``(body, http_status)`` failure pair.
ReplayOutcome: TypeAlias = ReplayPayload | tuple[ReplayPayload, int]
#: Viewer broadcast hook; the replay runtime ignores whatever it returns.
BroadcastFn: TypeAlias = Callable[[], object]
#: Re-entrant stepping callback used by the navigation helpers.
ReplayStepFn: TypeAlias = Callable[[], ReplayOutcome]


class ReplayIssue(TypedDict, total=False):
    """One semantic divergence recorded while replaying a Colonist archive."""

    step: int
    raw_event_index: int | None
    action_type: str | None
    kind: str
    severity: str
    message: str
    details: Mapping[str, object]


__all__ = [
    "ActionHint",
    "BroadcastFn",
    "GameEngineHolder",
    "GameEngineReplacer",
    "ParsedActions",
    "ReplayArchive",
    "ReplayIssue",
    "ReplayOutcome",
    "ReplayPayload",
    "ReplayRuntimeState",
    "ReplayStepFn",
    "as_list",
    "as_mapping",
    "list_field",
    "mapping_field",
    "optional_list_field",
    "optional_mapping_field",
    "parsed_actions_field",
]


@runtime_checkable
class GameEngineHolder(Protocol):
    """A sandbox that owns the authoritative engine for the runtime state."""

    @property
    def game_engine(self) -> GameEngine: ...


@runtime_checkable
class GameEngineReplacer(Protocol):
    """A sandbox that can swap in a restored engine instance."""

    def replace_game_engine(self, engine: GameEngine) -> None: ...


class ReplayRuntimeState(Protocol):
    """Mutable replay bookkeeping shared by the viewer and headless runners."""

    current_game: GameEngine | None
    game_running: bool
    game_log: list[dict[str, object]]

    replay_mode: bool
    replay_data: ReplayArchive | None
    replay_index: int
    replay_actions_per_step: list[int]
    replay_step_checkpoints: list[ReplayStepCheckpoint]
    replay_semantic_issues: list[ReplayIssue]
    replay_final_state_synced: bool
    replay_pending_dev_card: dict[str, object] | None
    replay_trade_ledger: dict[object, TradeLedgerRecord]
    replay_revision: int
    first_divergence_step: dict[str, int]

    @property
    def current_sandbox(self) -> GameEngineHolder | None:
        """The live or replay sandbox, if one is loaded.

        Read-only, and therefore covariant, so a concrete state may narrow it
        to its own sandbox union. Only the engine-holding capability is named
        here: the replay layer sits below ``cle.sandbox`` and must not depend
        on it, and it probes for the replace capability separately.
        """

    @property
    def replay_mutation_lock(self) -> AbstractContextManager[bool, None] | None:
        """Guards every mutation; read-only so a plain ``RLock`` qualifies."""

    corner_to_node_map: dict[str, int]
    edge_to_edge_map: dict[str, list[int]]


def _shape_error(path: str, expected: str, value: object) -> ValueError:
    return ValueError(
        f"Colonist archive field {path!r} must be {expected}, "
        f"got {type(value).__name__}"
    )


def as_mapping(value: object, path: str) -> Mapping[str, object]:
    """Narrow a fragment that the source always writes as a JSON object.

    A present-but-wrong-shape fragment raises instead of reading as empty, so
    a corrupt scrape surfaces where the untyped code raised ``AttributeError``.
    """
    if isinstance(value, Mapping):
        return value
    raise _shape_error(path, "a JSON object", value)


def as_list(value: object, path: str) -> list[object]:
    """Narrow a fragment that the source always writes as a JSON array."""
    if isinstance(value, list):
        return value
    raise _shape_error(path, "a JSON array", value)


def mapping_field(
    container: Mapping[str, object],
    key: str,
    path: str | None = None,
) -> Mapping[str, object]:
    """Read one nested object. An absent key reads as empty, as it always did."""
    if key not in container:
        return {}
    return as_mapping(container[key], path or key)


def list_field(
    container: Mapping[str, object],
    key: str,
    path: str | None = None,
) -> list[object]:
    """Read one nested array. An absent key reads as empty, as it always did."""
    if key not in container:
        return []
    return as_list(container[key], path or key)


def optional_mapping_field(
    container: Mapping[str, object],
    key: str,
    path: str | None = None,
) -> Mapping[str, object]:
    """Read a nested object the source also writes as null or omits entirely."""
    value = container.get(key)
    if not value:
        return {}
    return as_mapping(value, path or key)


def optional_list_field(
    container: Mapping[str, object],
    key: str,
    path: str | None = None,
) -> list[object]:
    """Read a nested array the source also writes as null or omits entirely."""
    value = container.get(key)
    if not value:
        return []
    return as_list(value, path or key)


@runtime_checkable
class ParsedActions(Protocol):
    """The archive's parsed rows.

    Read only by length and index, so a lazily-materializing stand-in works
    exactly as a plain list does.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> ActionHint: ...


def parsed_actions_field(
    container: Mapping[str, object],
    key: str = "parsed_actions",
) -> ParsedActions:
    """Read the archive's parsed rows. An absent key reads as empty."""
    if key not in container:
        empty: list[ActionHint] = []
        return empty
    value = container[key]
    if isinstance(value, ParsedActions):
        return value
    raise _shape_error(key, "an indexable sequence of parsed rows", value)
