from pathlib import Path

import pytest

from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox
from cle.traces.journal import SQLiteSandboxJournal


@pytest.fixture
def sandbox() -> CatanSandbox:
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
    return CatanSandbox.create(
        colors, {color: FirstLegalPlayer(color) for color in colors},
        seed=7, shuffle_players=False,
    )


@pytest.fixture
def journal(tmp_path: Path, sandbox: CatanSandbox) -> SQLiteSandboxJournal:
    store = SQLiteSandboxJournal(tmp_path / "journal.sqlite3")
    store.initialize(sandbox.game_engine.id, sandbox.snapshot(), "first-legal-v1")
    return store
