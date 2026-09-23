"""The one place that turns an active prompt source into runtime suites."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.harness.communication import CommunicationSuite, parse_communication_suite
from cle.harness.shared_suite import SharedPromptSuite, parse_shared_prompt_suite
from cle.harness.suite import ContextSuite, parse_context_suite

if TYPE_CHECKING:
    # documents.py validates through compile_runtime_suites, so this is type-only.
    from cle.harness.prompt_store.documents import ActivePromptSuites

RuntimeSuites = tuple[ContextSuite, CommunicationSuite]


def compile_runtime_suites(
    shared: str | SharedPromptSuite | None,
    decision: str | None = None,
    communication: str | None = None,
) -> RuntimeSuites:
    """Compile a shared bundle, or a legacy decision/communication pair.

    A shared bundle may arrive already parsed so a request never parses it twice.
    """
    if shared is not None:
        if decision is not None or communication is not None:
            raise ValueError("Conflicting shared and legacy prompt suite sources")
        bundle = (
            shared if isinstance(shared, SharedPromptSuite)
            else parse_shared_prompt_suite(shared)
        )
        return bundle.decision_suite(), bundle.communication_suite()
    if decision is None or communication is None:
        raise ValueError("A legacy prompt suite pair needs decision and communication sources")
    return (
        parse_context_suite(decision),
        parse_communication_suite(communication),
    )


def compile_active_suites(active: ActivePromptSuites) -> RuntimeSuites:
    shared = active.shared
    return compile_runtime_suites(
        None if shared is None else shared.bundle or shared.source,
        None if active.decision is None else active.decision.source,
        None if active.communication is None else active.communication.source,
    )
