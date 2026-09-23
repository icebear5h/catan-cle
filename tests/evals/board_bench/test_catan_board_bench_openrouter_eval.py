from types import TracebackType
from typing import Any

import pytest

from scripts.board_bench.run.eval_catan_board_bench_openrouter import call_openrouter


def test_openrouter_call_pins_endpoint_and_disables_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Any = {}
    response_payload = {
        "model": "qwen/qwen3.8-27b",
        "provider": "AkashML",
        "choices": [{"message": {"content": "EMPTY"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 1},
    }

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return response_payload

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> bool | None:
            return False

        def post(self, url: str, *, headers: dict[str, str], json: object) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(
        "scripts.board_bench.run.eval_catan_board_bench_openrouter.httpx.Client",
        FakeClient,
    )

    result = call_openrouter(
        "secret",
        "qwen/qwen3.8-27b",
        image_bytes=b"image",
        prompt="question",
        system_prompt="system",
        temperature=0.0,
        max_tokens=96,
        timeout=120.0,
        provider_order=("akashml/bf16",),
        allow_provider_fallbacks=False,
        disable_reasoning=True,
    )

    assert captured["payload"]["provider"] == {
        "order": ["akashml/bf16"],
        "allow_fallbacks": False,
    }
    assert captured["payload"]["reasoning"] == {"enabled": False}
    assert captured["payload"]["max_tokens"] == 256
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert result["response"] == "EMPTY"
    assert result["provider"] == "AkashML"
