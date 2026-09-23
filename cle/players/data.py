"""Concrete transport, checkpoint, and acceptance data used by player contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    from cle.game_engine.communication import SocialCommitment
    from cle.game_engine.events import EngineTransition, GameEngineSnapshot, GameEvent, PlayerEvent
    from cle.game_engine.models.enums import Action
    from cle.game_engine.models.player import Color
    from cle.game_engine.observation import PlayerObservation
    from cle.game_engine.trading import TradeOffer
    from cle.harness.models import ModelRequest, ModelResponse
    from cle.players.agent import AgentPlayerSnapshot
    from cle.players.contracts import (
        CommitmentProposal,
        CommunicationChoice,
        PlayerAttempt,
        PlayerChoice,
        PlayerContext,
        TalkContext,
    )
    from cle.sandbox.action_batches import AutomaticBatchAction, PendingActionBatch
    from cle.sandbox.communication import CommunicationOpportunity
    from cle.sandbox.contracts import SandboxSnapshot, SandboxStepResult
    from cle.sandbox.trade_preauthorization import AutomaticTradeAction, TradePreauthorization

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
ActionCall: TypeAlias = dict[str, JsonValue]
FirstLegalSnapshot: TypeAlias = tuple[int, int]
ScriptedSnapshot: TypeAlias = "tuple[int, int, tuple[PlayerChoice | int, ...]]"
BaselineSnapshot: TypeAlias = "FirstLegalSnapshot | ScriptedSnapshot"
PlayerSnapshot: TypeAlias = "BaselineSnapshot | AgentPlayerSnapshot"
PlayerStatus: TypeAlias = dict[str, str | int]

# The slot-restoration hook is also used by sandbox checkpoints. Keep its field
# domain explicit, including nested tuples of player state and admitted actions.
RestoredContract: TypeAlias = (
    "PlayerContext | PlayerChoice | TalkContext | CommunicationChoice | SandboxStepResult | SandboxSnapshot"
)
ContractSlot: TypeAlias = (
    "JsonValue | tuple[ContractSlot, ...] | Action | Color | TradeOffer | PlayerObservation "
    "| PlayerEvent | SocialCommitment | ModelRequest | ModelResponse | CommitmentProposal "
    "| PlayerContext | PlayerAttempt | EngineTransition | GameEvent | GameEngineSnapshot "
    "| AgentPlayerSnapshot | PlayerChoice | CommunicationOpportunity | TradePreauthorization "
    "| PendingActionBatch | AutomaticTradeAction | AutomaticBatchAction"
)


class AcceptanceResult(Protocol):
    """Context shared by committed sandbox and historical acceptance callbacks.

    SandboxStepResult also supplies after_revision and transitions. Historical
    callbacks can omit either, so acceptance reads those optional metadata fields
    with their original transition/zero fallback.
    """

    @property
    def context(self) -> PlayerContext | None: ...
