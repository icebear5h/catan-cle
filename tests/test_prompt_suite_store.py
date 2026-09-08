import os

import pytest

from cle.harness.communication import default_communication_suite_path
from cle.harness.prompt_store import (
    PromptSuiteConflictError,
    load_active_prompt_suites,
    reset_prompt_suite_overrides,
    save_prompt_suite_overrides,
)
from cle.harness.suite import default_suite_path, parse_context_suite
from cle.sandbox.factory import (
    LiveSandboxConfig,
    create_live_sandbox,
    materialize_live_prompt_suites,
)
from cle.game_engine.models.player import Color


class NeverTransport:
    async def complete(self, request):
        raise AssertionError("Prompt-store tests do not perform inference")


def _sources():
    return (
        default_suite_path().read_text(encoding="utf-8"),
        default_communication_suite_path().read_text(encoding="utf-8"),
    )


def _edited_sources():
    decision, communication = _sources()
    return (
        decision.replace(
            "Resolve the required discard.",
            "TEST OVERRIDE: resolve the required discard.",
        ),
        communication.replace(
            "Default to SILENCE.",
            "Default to SILENCE unless negotiation changes a decision.",
        ),
    )


def test_static_prompt_overrides_save_reset_and_detect_stale_edits(tmp_path):
    current = load_active_prompt_suites(tmp_path)
    assert current.decision.version == "10.0.0"
    assert current.communication.version == "5"
    decision_source, communication_source = _edited_sources()

    saved = save_prompt_suite_overrides(
        decision_source=decision_source,
        communication_source=communication_source,
        expected_decision_sha256=current.decision.sha256,
        expected_communication_sha256=current.communication.sha256,
        directory=tmp_path,
    )

    assert saved.decision.overridden is True
    assert saved.communication.overridden is True
    assert load_active_prompt_suites(tmp_path) == saved
    assert default_suite_path().read_text(encoding="utf-8") != decision_source

    with pytest.raises(PromptSuiteConflictError, match="refresh"):
        save_prompt_suite_overrides(
            decision_source=decision_source,
            communication_source=communication_source,
            expected_decision_sha256=current.decision.sha256,
            expected_communication_sha256=current.communication.sha256,
            directory=tmp_path,
        )

    reset = reset_prompt_suite_overrides(
        expected_decision_sha256=saved.decision.sha256,
        expected_communication_sha256=saved.communication.sha256,
        directory=tmp_path,
    )

    assert reset.decision.overridden is False
    assert reset.communication.overridden is False
    assert reset.decision.source == default_suite_path().read_text(encoding="utf-8")
    assert not (tmp_path / "decision.yaml").exists()
    assert not (tmp_path / "communication.yaml").exists()


def test_invalid_pair_never_changes_either_active_suite(tmp_path):
    before = load_active_prompt_suites(tmp_path)
    decision_source, communication_source = _edited_sources()

    with pytest.raises(ValueError):
        save_prompt_suite_overrides(
            decision_source=decision_source.replace(
                "{{ value }}",
                "{{ hidden_hand }}",
                1,
            ),
            communication_source=communication_source,
            expected_decision_sha256=before.decision.sha256,
            expected_communication_sha256=before.communication.sha256,
            directory=tmp_path,
        )

    assert load_active_prompt_suites(tmp_path) == before


def test_pair_write_rolls_back_if_second_atomic_replace_fails(
    tmp_path,
    monkeypatch,
):
    before = load_active_prompt_suites(tmp_path)
    decision_source, communication_source = _edited_sources()
    real_replace = os.replace
    failed = False

    def fail_second_replace(source, destination):
        nonlocal failed
        if str(destination).endswith("communication.yaml") and not failed:
            failed = True
            raise OSError("simulated second replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr("cle.harness.prompt_store.os.replace", fail_second_replace)

    with pytest.raises(OSError, match="simulated"):
        save_prompt_suite_overrides(
            decision_source=decision_source,
            communication_source=communication_source,
            expected_decision_sha256=before.decision.sha256,
            expected_communication_sha256=before.communication.sha256,
            directory=tmp_path,
        )

    assert load_active_prompt_suites(tmp_path) == before


def test_factory_explicit_paths_keep_legacy_suites_loadable():
    resolved = materialize_live_prompt_suites(
        LiveSandboxConfig(
            context_suite_path=str(
                default_suite_path().with_name("catan_v5.yaml")
            ),
            communication_suite_path=str(
                default_communication_suite_path().with_name(
                    "communication_v4.yaml"
                )
            ),
        )
    )

    assert resolved.decision_suite.version == "5.0.0"
    assert resolved.communication_suite.version == "4"


def test_factory_prefers_explicit_suite_path_over_local_override(
    tmp_path,
    monkeypatch,
):
    override_dir = tmp_path / "overrides"
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(override_dir))
    current = load_active_prompt_suites()
    decision_source, communication_source = _edited_sources()
    save_prompt_suite_overrides(
        decision_source=decision_source,
        communication_source=communication_source,
        expected_decision_sha256=current.decision.sha256,
        expected_communication_sha256=current.communication.sha256,
    )
    explicit_source = decision_source.replace(
        "TEST OVERRIDE: resolve the required discard.",
        "EXPLICIT PATH DISCARD GUIDANCE.",
    )
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text(explicit_source, encoding="utf-8")

    resolved = materialize_live_prompt_suites(
        LiveSandboxConfig(context_suite_path=str(explicit_path))
    )
    active = load_active_prompt_suites()

    assert resolved.decision_suite.source == explicit_source
    assert resolved.communication_suite.source == communication_source
    assert resolved.decision_suite.sha256 != active.decision.sha256

    sandbox = create_live_sandbox(
        LiveSandboxConfig(
            mode="llm_vs_random",
            seed=3,
            shuffle_players=False,
            palette="canonical_four",
        ),
        transport=NeverTransport(),
    )
    assert "TEST OVERRIDE: resolve the required discard." in (
        sandbox.players[Color.RED].suite.phase_guidance["discarding"]
    )


def test_persisted_v9_override_and_recorded_source_are_not_upgraded(tmp_path, monkeypatch):
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path))
    monkeypatch.delenv("CATAN_CONTEXT_SUITE", raising=False)
    monkeypatch.delenv("CATAN_COMMUNICATION_SUITE", raising=False)
    builtins = load_active_prompt_suites()
    old_source = default_suite_path().with_name("catan_v9.yaml").read_text(encoding="utf-8")
    saved = save_prompt_suite_overrides(
        decision_source=old_source,
        communication_source=builtins.communication.source,
        expected_decision_sha256=builtins.decision.sha256,
        expected_communication_sha256=builtins.communication.sha256,
    )
    assert load_active_prompt_suites().decision.source == old_source
    frozen = materialize_live_prompt_suites(LiveSandboxConfig())
    assert frozen.decision_suite.version == "9.0.0"
    assert frozen.decision_suite.sha256 == saved.decision.sha256
    assert parse_context_suite(frozen.decision_suite.source).context.social_context is False
    assert "discard" not in parse_context_suite(frozen.decision_suite.source).response.tags

    reset_prompt_suite_overrides(
        expected_decision_sha256=saved.decision.sha256,
        expected_communication_sha256=saved.communication.sha256,
    )
    current = materialize_live_prompt_suites(LiveSandboxConfig())
    assert current.decision_suite.version == "10.0.0"
    assert parse_context_suite(current.decision_suite.source).context.social_context is True
    assert "discard" in parse_context_suite(current.decision_suite.source).response.tags
    restored = materialize_live_prompt_suites(frozen)
    assert restored.decision_suite == frozen.decision_suite
    assert restored.decision_suite.source == old_source
