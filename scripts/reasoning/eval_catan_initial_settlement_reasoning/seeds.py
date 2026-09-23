"""Deterministic fresh-game inputs for the initial-settlement reasoning probe."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cle.env.observation_formatter import CatanObservationFormatter
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness.board_surface import board_presentation_payload
from cle.harness.context import ContextAssembler
from cle.harness.models import ModelRequest, PlayerSession
from cle.harness.suite import ContextSuite, load_context_suite
from cle.players.contracts import PlayerContext
from cle.players.data import JsonValue
from cle.sandbox.decision import build_decision_context
from scripts.reasoning.eval_catan_initial_settlement_reasoning.artifacts import (
    JsonDict,
    canonical_sha256,
    message_payload,
    read_json,
)

COLORS = (Color.BLUE, Color.RED, Color.WHITE, Color.ORANGE)

__all__ = [
    "COLORS",
    "SeedInput",
    "build_seed_input",
    "legal_action_payload",
    "load_prompt_variant",
]


@dataclass(frozen=True)
class SeedInput:
    context: PlayerContext
    request: ModelRequest
    manifest: JsonDict
    suite: ContextSuite


def load_prompt_variant(
    path: Path,
    base_suite: ContextSuite,
) -> tuple[ContextSuite, JsonDict]:
    payload = read_json(path)
    expected_keys = {
        "schema",
        "id",
        "version",
        "base_context_suite",
        "phase",
        "guidance",
    }
    if set(payload) != expected_keys:
        raise ValueError("Prompt variant fields do not match the v1 contract")
    if payload["schema"] != "catan-eval-prompt-variant/v1":
        raise ValueError("Unsupported prompt variant schema")
    base_identity = f"{base_suite.id}@{base_suite.version}"
    if payload["base_context_suite"] != base_identity:
        raise ValueError("Prompt variant targets a different base context suite")
    if payload["phase"] != "initial_settlement_1":
        raise ValueError("Reasoning probe variant must target initial_settlement_1")
    identity = {key: payload[key] for key in ("id", "version", "guidance")}
    if not all(
        isinstance(value, str) and value.strip() for value in identity.values()
    ):
        raise ValueError("Prompt variant identity and guidance must be non-empty")
    guidance = str(payload["guidance"])

    suite_payload = base_suite.model_dump()
    suite_payload["version"] = (
        f"{base_suite.version}+{payload['id']}.{payload['version']}"
    )
    suite_payload["phase_guidance"][payload["phase"]] = payload["guidance"]
    active_suite = ContextSuite.model_validate(suite_payload)
    metadata: JsonDict = {
        "schema": payload["schema"],
        "id": payload["id"],
        "version": payload["version"],
        "base_context_suite": base_identity,
        "phase": payload["phase"],
        "guidance": payload["guidance"],
        "guidance_sha256": hashlib.sha256(guidance.encode("utf-8")).hexdigest(),
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return active_suite, metadata


def legal_action_payload(context: PlayerContext) -> list[JsonDict]:
    formatter = CatanObservationFormatter()
    return [
        {
            "index": index,
            "action": str(action),
            "description": formatter._format_single_action(
                action,
                context.observation,
            ),
        }
        for index, action in enumerate(context.legal_actions)
    ]


def build_seed_input(
    seed: int,
    *,
    suite: ContextSuite | None = None,
    prompt_variant: Mapping[str, JsonValue] | None = None,
) -> SeedInput:
    engine = GameEngine(
        COLORS,
        seed=seed,
        shuffle_players=False,
        capture_history=False,
    )
    engine.id = f"fresh-initial-settlement-seed-{seed}"
    context = build_decision_context(engine, Color.BLUE, context_revision=0)
    active_suite = suite or load_context_suite()
    session = PlayerSession(
        color=Color.BLUE,
        session_id=f"fresh-initial-settlement-seed-{seed}:template",
    )
    request = ContextAssembler(active_suite).assemble(context, session)
    board = board_presentation_payload(
        request.board_presentation,
        include_text_content=True,
    )
    actions = legal_action_payload(context)
    messages = message_payload(request.messages)

    if context.actor is not Color.BLUE:
        raise RuntimeError("Fresh reasoning probe must have BLUE acting first")
    if context.phase != "initial_placement":
        raise RuntimeError("Fresh reasoning probe is not in initial placement")
    if context.prompt_key != "initial_settlement_1":
        raise RuntimeError("Fresh reasoning probe is not the first settlement")
    if context.events:
        raise RuntimeError("Fresh reasoning probe unexpectedly contains game history")
    if len(actions) != 54:
        raise RuntimeError(f"Expected 54 first-settlement actions, got {len(actions)}")

    prompt_identity: JsonDict = {
        "messages": list(messages),
        "board": board,
        "legal_actions": list(actions),
    }
    manifest: JsonDict = {
        "schema": "catan-initial-settlement-input/v1",
        "seed": seed,
        "engine_id": engine.id,
        "colors": [color.value for color in engine.state.colors],
        "actor": context.actor.value,
        "turn_number": context.turn_number,
        "phase": context.phase,
        "prompt_key": context.prompt_key,
        "context_id": context.context_id,
        "event_count": len(context.events),
        "strategic_memory": "",
        "legal_actions": list(actions),
        "messages": list(messages),
        "board_presentation": board,
        "prompt_sha256": canonical_sha256(prompt_identity),
        "legal_actions_sha256": canonical_sha256(list(actions)),
    }
    if prompt_variant is not None:
        manifest["context_suite"] = f"{active_suite.id}@{active_suite.version}"
        manifest["prompt_variant"] = dict(prompt_variant)
    return SeedInput(
        context=context,
        request=request,
        manifest=manifest,
        suite=active_suite,
    )
