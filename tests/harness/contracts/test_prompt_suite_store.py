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
    reset_shared_prompt_override,
    resolve_prompt_suites,
    save_shared_prompt_override,
)
from cle.harness.shared_suite import default_shared_suite_path, parse_shared_prompt_suite
from cle.harness.suite import default_suite_path, parse_context_suite
from cle.sandbox.factory import (
    LiveSandboxConfig,
    create_live_sandbox,
    materialize_live_prompt_suites,
)

SHARED_DISCARD = "Choose exactly the required number of cards from your holdings"
EDITED_DISCARD = "TEST OVERRIDE: choose exactly the required number of cards from your holdings"


class NeverTransport:
    async def complete(self, request: ModelRequest) -> None:
        raise AssertionError("Prompt-store tests do not perform inference")


def _edited_shared_source() -> str:
    source = default_shared_suite_path().read_text(encoding="utf-8")
    assert SHARED_DISCARD in source
    return source.replace(SHARED_DISCARD, EDITED_DISCARD)


def _clear_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE",
                "CATAN_PROMPT_SUITE_FOLLOW_LATEST"):
        monkeypatch.delenv(var, raising=False)


def test_static_prompt_overrides_save_reset_and_detect_stale_edits(tmp_path: Path) -> None:
    current: Any = resolve_prompt_suites(directory=tmp_path, use_environment=False)
    assert current.decision is current.communication is None
    assert current.shared.overridden is False
    assert current.shared.source == default_shared_suite_path().read_text(encoding="utf-8")
    edited = _edited_shared_source()

    saved: Any = save_shared_prompt_override(
        source=edited, expected_sha256=current.shared.sha256, directory=tmp_path,
    )

    assert saved.shared.overridden is True
    assert resolve_prompt_suites(directory=tmp_path, use_environment=False) == saved
    assert (tmp_path / "shared.yaml").read_text(encoding="utf-8") == edited
    assert default_shared_suite_path().read_text(encoding="utf-8") != edited

    with pytest.raises(PromptSuiteConflictError, match="refresh"):
        save_shared_prompt_override(
            source=edited, expected_sha256=current.shared.sha256, directory=tmp_path,
        )
    with pytest.raises(PromptSuiteConflictError, match="refresh"):
        reset_shared_prompt_override(expected_sha256=current.shared.sha256, directory=tmp_path)

    reset: Any = reset_shared_prompt_override(
        expected_sha256=saved.shared.sha256, directory=tmp_path,
    )

    assert reset.shared.overridden is False
    assert reset.shared.source == default_shared_suite_path().read_text(encoding="utf-8")
    assert reset == current
    assert not (tmp_path / "shared.yaml").exists()


def test_invalid_shared_source_never_changes_active_suite(tmp_path: Path) -> None:
    before: Any = resolve_prompt_suites(directory=tmp_path, use_environment=False)
    invalid = before.shared.source.replace("{{ resources }}", "{{ hidden_hand }}", 1)

    with pytest.raises(ValueError):
        save_shared_prompt_override(
            source=invalid, expected_sha256=before.shared.sha256, directory=tmp_path,
        )

    assert resolve_prompt_suites(directory=tmp_path, use_environment=False) == before
    assert not (tmp_path / "shared.yaml").exists()


def test_failed_atomic_replace_keeps_active_suite_and_cleans_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before: Any = resolve_prompt_suites(directory=tmp_path, use_environment=False)

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("simulated replace failure")

    # overrides.py calls os.replace through the os module, so patch it there.
    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        save_shared_prompt_override(
            source=_edited_shared_source(), expected_sha256=before.shared.sha256,
            directory=tmp_path,
        )

    assert resolve_prompt_suites(directory=tmp_path, use_environment=False) == before
    assert not list(tmp_path.glob(".prompt-suite-*.tmp"))


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
    _clear_pins(monkeypatch)
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path / "overrides"))
    current: Any = resolve_prompt_suites()
    saved: Any = save_shared_prompt_override(
        source=_edited_shared_source(), expected_sha256=current.shared.sha256,
    )
    explicit_source = default_suite_path().read_text(encoding="utf-8").replace(
        "Resolve the required discard.",
        "EXPLICIT PATH DISCARD GUIDANCE.",
    )
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text(explicit_source, encoding="utf-8")

    resolved: Any = materialize_live_prompt_suites(
        LiveSandboxConfig(context_suite_path=str(explicit_path))
    )

    assert resolved.shared_suite is None
    assert resolved.decision_suite.source == explicit_source
    assert resolved.communication_suite.source == (
        default_communication_suite_path().read_text(encoding="utf-8")
    )
    assert resolved.decision_suite.sha256 != saved.shared.sha256

    sandbox: Any = create_live_sandbox(
        LiveSandboxConfig(
            mode="llm_vs_random",
            seed=3,
            shuffle_players=False,
            palette="canonical_four",
        ),
        transport=NeverTransport(),
    )
    assert EDITED_DISCARD in sandbox.players[Color.RED].suite.phase_guidance["discarding"]


@pytest.mark.parametrize("version", ["9.0.0", "10.0.0"])
def test_recorded_legacy_source_is_not_upgraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    _clear_pins(monkeypatch)
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path))
    old_path = default_suite_path().with_name(f"catan_v{version.split('.')[0]}.yaml")
    old_source = old_path.read_text(encoding="utf-8")
    frozen: Any = materialize_live_prompt_suites(
        LiveSandboxConfig(context_suite_path=str(old_path))
    )
    assert frozen.shared_suite is None
    assert frozen.decision_suite.version == version
    assert frozen.decision_suite.source == old_source
    historical = parse_context_suite(frozen.decision_suite.source)
    assert historical.response.format == "xml"
    assert historical.context.social_context == (version == "10.0.0")
    assert ("discard" in historical.response.tags) == (version == "10.0.0")

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


def test_legacy_pair_override_files_are_ignored(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    builtin: Any = resolve_prompt_suites(directory=tmp_path, use_environment=False)
    (tmp_path / "decision.yaml").write_text(
        default_suite_path().read_text(encoding="utf-8"), encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="cle.harness.prompt_store.resolution"):
        active: Any = resolve_prompt_suites(directory=tmp_path, use_environment=False)

    assert active == builtin
    assert active.decision is None
    assert "Ignoring legacy prompt pair override files decision.yaml" in caplog.text


def test_follow_latest_ignores_stale_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATAN_PROMPT_SUITE_DIR", str(tmp_path))
    _clear_pins(monkeypatch)
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
