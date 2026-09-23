"""Shared fixtures for openrouter http failure reasons, redaction, and retry behaviour."""

from unittest.mock import AsyncMock

import pytest

from cle.harness.models import ModelMessage, ModelRequest
from cle.harness.providers import openrouter


@pytest.fixture
def model_request() -> ModelRequest:
    return ModelRequest(
        decision_id="decision-403",
        session_id="game-1:RED",
        messages=(
            ModelMessage("system", "private system rules"),
            ModelMessage("user", 'private state: "wood/ore"\nprivate second line'),
        ),
    )


@pytest.fixture(autouse=True)
def retry_sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    sleep = AsyncMock()
    monkeypatch.setattr(openrouter.asyncio, "sleep", sleep)
    return sleep
