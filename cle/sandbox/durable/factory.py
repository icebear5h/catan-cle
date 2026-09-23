"""Live-policy construction with frozen settings for recoverable rollouts."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

from cle.harness.models import CompletionTransport
from cle.sandbox.contracts import SandboxSnapshot
from cle.sandbox.factory import (
    CEREBRAS_MODEL_PREFIX,
    LiveSandboxConfig,
    _migrate_player_snapshot,
    create_live_sandbox,
    create_text_transport,
    materialize_live_prompt_suites,
    resolve_live_model,
)
from cle.traces.journal import JournalConflict, SQLiteSandboxJournal
from cle.traces.sqlite.blobs import _json_text

from .runtime import DurableSandbox


def create_durable_sandbox(
    config: LiveSandboxConfig, journal: SQLiteSandboxJournal, *,
    sandbox_id: str | None = None, transport: CompletionTransport | None = None,
    recover_pending: bool = False,
) -> DurableSandbox:
    """Start, or resume by ID, with inference configuration pinned for recovery.

    Use the same config and checkpoint/model revision when recovering a pending
    command. An injected transport's weights/settings must likewise stay fixed.
    Idle checkpoint continuation may intentionally select a different policy.
    """
    config = deepcopy(config)
    frozen = materialize_live_prompt_suites(config) if config.mode != "random" else config
    route = "local"
    if config.mode != "random":
        frozen = replace(frozen, model=resolve_live_model(frozen.model))
        if transport is not None:
            route = f"injected:{type(transport).__module__}.{type(transport).__qualname__}"
        elif frozen.model is not None and frozen.model.startswith(CEREBRAS_MODEL_PREFIX):
            route = "cerebras:https://api.cerebras.ai/v1"
        elif endpoint := os.getenv("VLLM_BASE_URL"):
            # Only persist the digest: even an endpoint URL may contain secrets.
            route = "vllm:" + sha256(endpoint.encode()).hexdigest()
        else:
            route = "openrouter:https://openrouter.ai/api/v1/chat/completions"
    identity = sha256(_json_text((frozen, route)).encode()).hexdigest()
    snapshot = journal.head(sandbox_id).snapshot if sandbox_id is not None else None
    if snapshot is not None:
        head = journal.head(snapshot.engine.engine_id)
        if head.pending_command_id is not None:
            command = journal.command(head.sandbox_id, head.pending_command_id)
            if not recover_pending or command.ticket.policy_identity != identity:
                raise JournalConflict("Pending recovery requires explicit consent and the original policy")
    if config.mode != "random" and transport is None:
        transport = create_text_transport(frozen)
    sandbox = create_live_sandbox(frozen, transport=transport, snapshot=snapshot)
    if sandbox_id is None:
        return DurableSandbox.start(sandbox, journal, policy_identity=identity)
    def restore_checkpoint(checkpoint: SandboxSnapshot) -> None:
        migrated = replace(checkpoint, player_states=tuple(
            (color, _migrate_player_snapshot(state, sandbox.players.get(color)))
            for color, state in checkpoint.player_states
        ))
        sandbox.restore(migrated)

    return DurableSandbox.resume(
        sandbox, journal, policy_identity=identity, recover_pending=recover_pending,
        restore_checkpoint=restore_checkpoint,
    )
