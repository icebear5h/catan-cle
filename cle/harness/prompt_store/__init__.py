"""Validated static prompt-suite overrides for local Catan Lab use."""

from __future__ import annotations

# Re-exported so atomic replacement stays patchable at this module path.
import os as os

# Suite helpers stay resolvable here so tests can replace them at call time.
from cle.harness.communication import (
    default_communication_suite_path as default_communication_suite_path,
)
from cle.harness.communication import (
    parse_communication_suite as parse_communication_suite,
)
from cle.harness.prompt_store.compile import (
    RuntimeSuites,
    compile_active_suites,
    compile_runtime_suites,
)
from cle.harness.prompt_store.documents import (
    ActivePromptSuites,
    PromptSuiteConflictError,
    PromptSuiteDocument,
    source_sha256,
    validate_shared_prompt_source,
)
from cle.harness.prompt_store.overrides import (
    reset_shared_prompt_override,
    save_shared_prompt_override,
)
from cle.harness.prompt_store.resolution import (
    COMMUNICATION_SUITE_ENV,
    CONTEXT_SUITE_ENV,
    FOLLOW_LATEST_ENV,
    PROMPT_SUITE_PIN_ENVS,
    SHARED_SUITE_ENV,
    follow_latest_enabled,
    resolve_communication_suite_document,
    resolve_decision_suite_document,
    resolve_prompt_suites,
)
from cle.harness.prompt_store.storage import DEFAULT_PROMPT_SUITE_DIR, prompt_suite_directory

# The store lock is replaced by tests through this module path at call time.
from cle.harness.prompt_store.storage import _store_lock as _store_lock
from cle.harness.shared_suite import (
    default_shared_suite_path as default_shared_suite_path,
)
from cle.harness.shared_suite import (
    parse_shared_prompt_suite as parse_shared_prompt_suite,
)
from cle.harness.suite import default_suite_path as default_suite_path
from cle.harness.suite import parse_context_suite as parse_context_suite

__all__ = [
    "COMMUNICATION_SUITE_ENV",
    "CONTEXT_SUITE_ENV",
    "DEFAULT_PROMPT_SUITE_DIR",
    "FOLLOW_LATEST_ENV",
    "PROMPT_SUITE_PIN_ENVS",
    "SHARED_SUITE_ENV",
    "ActivePromptSuites",
    "PromptSuiteConflictError",
    "PromptSuiteDocument",
    "RuntimeSuites",
    "compile_active_suites",
    "compile_runtime_suites",
    "follow_latest_enabled",
    "prompt_suite_directory",
    "reset_shared_prompt_override",
    "resolve_communication_suite_document",
    "resolve_decision_suite_document",
    "resolve_prompt_suites",
    "save_shared_prompt_override",
    "source_sha256",
    "validate_shared_prompt_source",
]
