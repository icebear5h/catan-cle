"""Live sandbox configuration records and their defaults."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from cle.game_engine.communication import CommunicationLimits
from cle.game_engine.trading import TradeLimits
from cle.harness.catan_board_surface import BoardSurfaceKind
from cle.harness.reasoning import native_reasoning_request
from cle.players.data import JsonValue
from cle.sandbox.palette import PaletteMode, validate_palette_mode

__all__ = [
    "CEREBRAS_MODEL_PREFIX",
    "DEFAULT_LIVE_MAX_DECISION_ATTEMPTS",
    "DEFAULT_LIVE_MODEL",
    "DEFAULT_LIVE_REASONING_EFFORT",
    "LivePromptSuiteSource",
    "LiveSandboxConfig",
    "ReasoningRequest",
]

DEFAULT_LIVE_MODEL = "qwen/qwen3.8-27b"
DEFAULT_LIVE_REASONING_EFFORT = "high"
DEFAULT_LIVE_MAX_DECISION_ATTEMPTS = 3
# A `cerebras/<id>` model routes to Cerebras per game; every other model keeps
# the env-selected vLLM/OpenRouter transport.
CEREBRAS_MODEL_PREFIX = "cerebras/"

# One normalized native-reasoning request as built by cle.harness.reasoning:
# {"enabled": False} | {"effort": str, "exclude": False} | {"max_tokens": int, ...}.
# The value domain matches what that module produces and accepts.
ReasoningRequest: TypeAlias = Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class LivePromptSuiteSource:
    """Exact validated suite text for request provenance or explicit selection."""

    id: str
    version: str
    sha256: str
    source: str


@dataclass(frozen=True, slots=True)
class LiveSandboxConfig:
    mode: Literal["random", "llm", "llm_vs_random"] = "random"
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int | None = None
    reasoning: ReasoningRequest | None = field(
        default_factory=lambda: native_reasoning_request(
            DEFAULT_LIVE_REASONING_EFFORT
        )
    )
    seed: int | None = None
    shuffle_players: bool = True
    palette: PaletteMode = "random_all"
    board_surface: BoardSurfaceKind = "indexed_tile_rows"
    context_suite_path: str | None = None
    communication_suite_path: str | None = None
    decision_suite: LivePromptSuiteSource | None = None
    communication_suite: LivePromptSuiteSource | None = None
    max_decision_attempts: int = DEFAULT_LIVE_MAX_DECISION_ATTEMPTS
    trade_limits: TradeLimits = field(default_factory=TradeLimits)
    communication_limits: CommunicationLimits = field(default_factory=CommunicationLimits)
    shared_suite: LivePromptSuiteSource | None = None
    shared_suite_path: str | None = None

    def __post_init__(self) -> None:
        validate_palette_mode(self.palette)
        if (
            self.max_tokens is not None
            and (
                isinstance(self.max_tokens, bool)
                or not isinstance(self.max_tokens, int)
                or self.max_tokens < 1
            )
        ):
            raise ValueError("max_tokens must be a positive integer or null")
        if self.board_surface not in {
            "legacy_semantic",
            "indexed_tile_rows",
            "image",
        }:
            raise ValueError(f"Unsupported board surface: {self.board_surface}")


LivePromptSuiteSource.__module__ = "cle.sandbox.factory"
LiveSandboxConfig.__module__ = "cle.sandbox.factory"
