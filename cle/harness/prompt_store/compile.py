"""The one place that turns an active prompt source into runtime suites."""

from __future__ import annotations

from cle.harness import prompt_store
from cle.harness.communication import CommunicationSuite
from cle.harness.prompt_store.documents import ActivePromptSuites
from cle.harness.shared_suite import SharedPromptSuite
from cle.harness.suite import ContextSuite

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
            else prompt_store.parse_shared_prompt_suite(shared)
        )
        return bundle.decision_suite(), bundle.communication_suite()
    if decision is None or communication is None:
        raise ValueError("A legacy prompt suite pair needs decision and communication sources")
    return (
        prompt_store.parse_context_suite(decision),
        prompt_store.parse_communication_suite(communication),
    )


def compile_active_suites(active: ActivePromptSuites) -> RuntimeSuites:
    shared = active.shared
    return compile_runtime_suites(
        None if shared is None else shared.bundle or shared.source,
        None if active.decision is None else active.decision.source,
        None if active.communication is None else active.communication.source,
    )
