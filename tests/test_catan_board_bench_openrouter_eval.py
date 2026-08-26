from scripts.eval_catan_board_bench_openrouter import call_openrouter


def test_openrouter_call_pins_endpoint_and_disables_reasoning(monkeypatch) -> None:
    captured = {}
    response_payload = {
        "model": "qwen/qwen3.8-27b",
        "provider": "AkashML",
        "choices": [{"message": {"content": "EMPTY"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 1},
    }

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return response_payload

    class FakeClient:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(
        "scripts.eval_catan_board_bench_openrouter.httpx.Client",
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
