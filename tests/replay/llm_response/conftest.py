"""Legacy prompt-suite fixture for the replay LLM response route tests."""

from pathlib import Path

import pytest

from cle.harness.prompt_store import resolve_prompt_suites


@pytest.fixture
def legacy_prompt_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "playground.game_viewer.replay.decision_preview.resolve_prompt_suites",
        lambda: resolve_prompt_suites(
            directory=tmp_path, legacy=True, use_environment=False,
        ),
    )
