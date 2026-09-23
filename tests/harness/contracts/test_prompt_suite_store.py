import os
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.models.player import Color
from cle.harness.communication import default_communication_suite_path
from cle.harness.models import ModelRequest
from cle.harness.prompt_store import (
    PromptSuiteConflictError,
    follow_latest_enabled,
    load_active_prompt_suites,
    reset_prompt_suite_overrides,
    resolve_prompt_suites,
    save_prompt_suite_overrides,
    save_shared_prompt_override,
)
from cle.harness.shared_suite import default_shared_suite_path, parse_shared_prompt_suite
from cle.harness.suite import default_suite_path, parse_context_suite
from cle.sandbox.factory import (
    LiveSandboxConfig,
    create_live_sandbox,
    materialize_live_prompt_suites,
)


class NeverTransport:
    async def complete(self, request: ModelRequest) -> None:
        raise AssertionError("Prompt-store tests do not perform inference")


def _sources() -> tuple[str, str]:
    return (
        default_suite_path().read_text(encoding="utf-8"),
        default_communication_suite_path().read_text(encoding="utf-8"),
    )


def _edited_sources() -> tuple[str, str]:
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


def test_static_prompt_overrides_save_reset_and_detect_stale_edits(tmp_path: Path) -> None:
    current: Any = load_active_prompt_suites(tmp_path)
    assert current.decision.version == "11.0.0"
    assert current.communication.version == "5"
    decision_source, communication_source = _edited_sources()

    saved: Any = save_prompt_suite_overrides(
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

    reset: Any = reset_prompt_suite_overrides(
        expected_decision_sha256=saved.decision.sha256,
        expected_communication_sha256=saved.communication.sha256,
        directory=tmp_path,
    )

    assert reset.decision.overridden is False
    assert reset.communication.overridden is False
    assert reset.decision.source == default_suite_path().read_text(encoding="utf-8")
    assert not (tmp_path / "decision.yaml").exists()
    assert not (tmp_path / "communication.yaml").exists()


def test_invalid_pair_never_changes_either_active_suite(tmp_path: Path) -> None:
    before: Any = load_active_prompt_suites(tmp_path)
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before: Any = load_active_prompt_suites(tmp_path)
    decision_source, communication_source = _edited_sources()
    real_replace = os.replace
    failed = False

    def fail_second_replace(source: str | Path, destination: str | Path) -> None:
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


def test_factory_explicit_paths_keep_legacy_suites_loadable() -> None:
    resolved: Any = materialize_live_prompt_suites(
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_dir = tmp_path / "overrides"
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(override_dir))
    current: Any = load_active_prompt_suites()
    decision_source, communication_source = _edited_sources()
    save_prompt_suite_overrides(
        decision_source=decision_source,
        communication_source=communication_source,
        expected_decision_sha256=current.decision.sha256,
        expected_communication_sha256=current.communication.sha256,
    )
    explicit_source: Any = decision_source.replace(
        "TEST OVERRIDE: resolve the required discard.",
        "EXPLICIT PATH DISCARD GUIDANCE.",
    )
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text(explicit_source, encoding="utf-8")

    resolved: Any = materialize_live_prompt_suites(
        LiveSandboxConfig(context_suite_path=str(explicit_path))
    )
    active: Any = load_active_prompt_suites()

    assert resolved.decision_suite.source == explicit_source
    assert resolved.communication_suite.source == communication_source
    assert resolved.decision_suite.sha256 != active.decision.sha256

    sandbox: Any = create_live_sandbox(
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


@pytest.mark.parametrize("version", ["9.0.0", "10.0.0"])
def test_persisted_override_and_recorded_source_are_not_upgraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path))
    monkeypatch.delenv("CATAN_SHARED_SUITE", raising=False)
    monkeypatch.delenv("CATAN_CONTEXT_SUITE", raising=False)
    monkeypatch.delenv("CATAN_COMMUNICATION_SUITE", raising=False)
    builtins: Any = load_active_prompt_suites()
    old_source: Any = (
        default_suite_path()
        .with_name(f"catan_v{version.split('.')[0]}.yaml")
        .read_text(encoding="utf-8")
    )
    saved: Any = save_prompt_suite_overrides(
        decision_source=old_source,
        communication_source=builtins.communication.source,
        expected_decision_sha256=builtins.decision.sha256,
        expected_communication_sha256=builtins.communication.sha256,
    )
    assert load_active_prompt_suites().decision.source == old_source
    frozen: Any = materialize_live_prompt_suites(LiveSandboxConfig())
    assert frozen.decision_suite.version == version
    assert frozen.decision_suite.sha256 == saved.decision.sha256
    historical = parse_context_suite(frozen.decision_suite.source)
    assert historical.response.format == "xml"
    assert historical.context.social_context == (version == "10.0.0")
    assert ("discard" in historical.response.tags) == (version == "10.0.0")

    reset_prompt_suite_overrides(
        expected_decision_sha256=saved.decision.sha256,
        expected_communication_sha256=saved.communication.sha256,
    )
    current: Any = materialize_live_prompt_suites(LiveSandboxConfig())
    active: Any = resolve_prompt_suites()
    assert active.decision is active.communication is None
    assert active.shared is not None
    assert active.shared.overridden is False
    assert current.decision_suite is current.communication_suite is None
    assert current.shared_suite.source == active.shared.source
    assert current.shared_suite.sha256 == active.shared.sha256
    default_suite = parse_shared_prompt_suite(current.shared_suite.source).decision_suite()
    assert default_suite.context.mode == "shared"
    assert default_suite.context.memory_mode == "fresh_notes"
    assert default_suite.response.format == "json"
    assert default_suite.response.tags == ("tool", "arguments", "notes")
    assert default_suite.context.social_context is True
    restored: Any = materialize_live_prompt_suites(frozen)
    assert restored.shared_suite is None
    assert restored.decision_suite == frozen.decision_suite
    assert restored.decision_suite.source == old_source
    assert restored.communication_suite == frozen.communication_suite


def test_follow_latest_ignores_stale_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path))
    for var in ("CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE",
                "CATAN_PROMPT_SUITE_FOLLOW_LATEST"):
        monkeypatch.delenv(var, raising=False)
    assert follow_latest_enabled(None) is False
    assert follow_latest_enabled(True) is True
    builtin: Any = resolve_prompt_suites()
    assert builtin.shared is not None and builtin.shared.overridden is False
    edited = builtin.shared.source.replace("{{ resources }}", "EDITED: {{ resources }}")
    saved: Any = save_shared_prompt_override(
        source=edited, expected_sha256=builtin.shared.sha256, directory=tmp_path,
    )
    assert saved.shared.overridden is True
    assert resolve_prompt_suites().shared.overridden is True
    with pytest.warns(UserWarning, match="stale local"):
        followed: Any = resolve_prompt_suites(follow_latest=True)
    assert followed.shared.overridden is False
    assert followed.shared.source == builtin.shared.source
    monkeypatch.setenv("CATAN_SHARED_SUITE", str(default_shared_suite_path()))
    with pytest.warns(UserWarning, match="environment pins"):
        env_followed: Any = resolve_prompt_suites(follow_latest=True)
    assert env_followed.shared.overridden is False
    assert env_followed.shared.source == builtin.shared.source
    monkeypatch.delenv("CATAN_SHARED_SUITE")
    legacy_config: Any = LiveSandboxConfig(
        context_suite_path=str(default_suite_path().with_name("catan_v5.yaml")),
        communication_suite_path=str(
            default_communication_suite_path().with_name("communication_v4.yaml")),
    )
    assert materialize_live_prompt_suites(legacy_config).decision_suite.version == "5.0.0"
    with pytest.warns(UserWarning, match="configured prompt suite paths"):
        latest = materialize_live_prompt_suites(legacy_config, follow_latest=True)
    assert latest.shared_suite is not None
    assert latest.shared_suite.source == builtin.shared.source
    monkeypatch.setenv("CATAN_PROMPT_SUITE_FOLLOW_LATEST", "1")
    with pytest.warns(UserWarning, match="stale local"):
        assert resolve_prompt_suites().shared.source == builtin.shared.source
