"""Validated prompt-suite documents, content digests, and conflict detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from cle.harness.communication import parse_communication_suite
from cle.harness.prompt_store.compile import compile_runtime_suites
from cle.harness.shared_suite import SharedPromptSuite, parse_shared_prompt_suite
from cle.harness.suite import parse_context_suite


class PromptSuiteConflictError(RuntimeError):
    """Raised when an editor submits hashes for superseded active sources."""


@dataclass(frozen=True, slots=True)
class PromptSuiteDocument:
    kind: str
    id: str
    version: str
    status: str
    sha256: str
    source: str
    overridden: bool
    # The parsed shared bundle, kept so one request never parses its source twice.
    bundle: SharedPromptSuite | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class ActivePromptSuites:
    decision: PromptSuiteDocument | None = None
    communication: PromptSuiteDocument | None = None
    shared: PromptSuiteDocument | None = None


def source_sha256(source: str) -> str:
    """The content digest every prompt-suite record and editor hash uses."""
    return sha256(source.encode("utf-8")).hexdigest()


def validate_shared_prompt_source(
    source: str, *, overridden: bool = True,
) -> ActivePromptSuites:
    suite = parse_shared_prompt_suite(source)
    # Validate the actual runtime contracts before admitting the authored bundle.
    compile_runtime_suites(suite)
    return ActivePromptSuites(shared=PromptSuiteDocument(
        kind="shared", id=suite.id, version=str(suite.version), status=suite.status,
        sha256=source_sha256(source), source=source, overridden=overridden, bundle=suite,
    ))


def check_shared_hash(current_sha256: str, expected: str) -> None:
    if current_sha256 != expected:
        raise PromptSuiteConflictError("Active prompt suites changed; refresh before saving")


def decision_document(source: str) -> PromptSuiteDocument:
    suite = parse_context_suite(source)
    return PromptSuiteDocument(
        kind="decision", id=suite.id, version=str(suite.version), status=suite.status,
        sha256=source_sha256(source), source=source, overridden=False,
    )


def communication_document(source: str) -> PromptSuiteDocument:
    suite = parse_communication_suite(source)
    return PromptSuiteDocument(
        kind="communication", id=suite.id, version=str(suite.version), status=suite.status,
        sha256=source_sha256(source), source=source, overridden=False,
    )
