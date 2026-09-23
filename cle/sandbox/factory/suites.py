"""Resolve and validate the exact prompt-suite sources bound to one game."""

from __future__ import annotations

import warnings
from dataclasses import asdict, replace

from cle.harness.communication import parse_communication_suite
from cle.harness.models import PromptSource
from cle.harness.prompt_store import (
    RuntimeSuites,
    compile_runtime_suites,
    follow_latest_enabled,
    resolve_communication_suite_document,
    resolve_decision_suite_document,
    resolve_prompt_suites,
    source_sha256,
)
from cle.harness.suite import parse_context_suite
from cle.sandbox.factory.config import LivePromptSuiteSource, LiveSandboxConfig

__all__ = [
    "compile_live_suites",
    "materialize_live_prompt_suites",
]


def compile_live_suites(config: LiveSandboxConfig) -> RuntimeSuites:
    """Compile a materialized config's bound sources into runtime suites."""
    return compile_runtime_suites(*(
        None if record is None else record.source
        for record in (config.shared_suite, config.decision_suite, config.communication_suite)
    ))


def _prompt_sources(config: LiveSandboxConfig) -> tuple[PromptSource, ...]:
    return tuple(
        PromptSource(kind=kind, **asdict(source))
        for kind in ("shared", "decision", "communication")
        if (source := getattr(config, f"{kind}_suite")) is not None
    )


def materialize_live_prompt_suites(
    config: LiveSandboxConfig,
    *,
    restoring: bool = False,
    follow_latest: bool | None = None,
) -> LiveSandboxConfig:
    """Resolve one exact source snapshot for an inference boundary.

    `restoring` remains accepted for callers; saved sources are not required.
    Explicit source objects are deliberate runtime selections, not checkpoints.
    When follow-latest is enabled (explicitly or via
    CATAN_PROMPT_SUITE_FOLLOW_LATEST=1), config path pins and environment pins
    are ignored in favor of the built-in latest suite. Already-materialized
    source objects are still honored to preserve inference boundaries.
    """
    follow = follow_latest_enabled(follow_latest)
    shared_path = config.shared_suite_path
    context_path = config.context_suite_path
    communication_path = config.communication_suite_path
    if follow and any(
        value is not None for value in (
            shared_path, context_path, communication_path,
        )
    ):
        warnings.warn(
            "Ignoring configured prompt suite paths; following latest built-in suite",
            stacklevel=2,
        )
        shared_path = context_path = communication_path = None
    legacy_selected = any(value is not None for value in (
        config.decision_suite, config.communication_suite,
        context_path, communication_path,
    ))
    if (config.shared_suite is not None or shared_path is not None) and legacy_selected:
        raise ValueError("Conflicting shared and legacy prompt suite sources")
    if config.shared_suite is not None:
        decision_suite, _ = compile_runtime_suites(config.shared_suite.source)
        _validate_suite_identity(config.shared_suite, decision_suite.id, decision_suite.version)
        return config
    decision = config.decision_suite
    communication = config.communication_suite
    if decision is not None:
        _validate_decision_suite_source(decision)
    if communication is not None:
        _validate_communication_suite_source(communication)
    if decision is not None and communication is not None:
        return config
    if decision is not None:
        document = resolve_communication_suite_document(communication_path)
        communication = _suite_source(document.id, document.version, document.source)
    elif communication is not None:
        document = resolve_decision_suite_document(context_path)
        decision = _suite_source(document.id, document.version, document.source)
    else:
        active = resolve_prompt_suites(
            shared_path=shared_path,
            decision_path=context_path,
            communication_path=communication_path,
            legacy=legacy_selected,
            use_environment=not follow,
            follow_latest=follow,
        )
        if active.shared is not None:
            document = active.shared
            return replace(config, shared_suite=_suite_source(
                document.id, document.version, document.source,
            ))
        assert active.decision is not None and active.communication is not None
        decision = _suite_source(active.decision.id, active.decision.version, active.decision.source)
        communication = _suite_source(
            active.communication.id, active.communication.version, active.communication.source,
        )
    _validate_decision_suite_source(decision)
    _validate_communication_suite_source(communication)
    return replace(
        config,
        decision_suite=decision,
        communication_suite=communication,
    )


def _suite_source(
    suite_id: str,
    version: str | int,
    source: str,
) -> LivePromptSuiteSource:
    return LivePromptSuiteSource(
        id=suite_id,
        version=str(version),
        sha256=source_sha256(source),
        source=source,
    )


def _validate_decision_suite_source(record: LivePromptSuiteSource) -> None:
    suite = parse_context_suite(record.source, source_name="recorded decision suite")
    _validate_suite_identity(record, suite.id, suite.version)


def _validate_communication_suite_source(record: LivePromptSuiteSource) -> None:
    suite = parse_communication_suite(
        record.source,
        source_name="recorded communication suite",
    )
    _validate_suite_identity(record, suite.id, suite.version)


def _validate_suite_identity(
    record: LivePromptSuiteSource,
    suite_id: str,
    version: str | int,
) -> None:
    if record.sha256 != source_sha256(record.source):
        raise ValueError("Recorded prompt suite SHA-256 does not match its source")
    if record.id != suite_id or record.version != str(version):
        raise ValueError("Recorded prompt suite identity does not match its source")
