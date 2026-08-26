import json

from playground import openrouter_client


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    responses = []
    payloads = []

    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, headers, json):
        self.payloads.append(json)
        return _FakeResponse(self.responses.pop(0))


def test_query_text_with_tools_runs_forced_tool_then_returns_json(monkeypatch):
    _FakeClient.payloads = []
    _FakeClient.responses = [
        {
            "model": "openai/gpt-5.6-sol",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-board",
                                "type": "function",
                                "function": {
                                    "name": "inspect_board",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.01,
            },
        },
        {
            "model": "openai/gpt-5.6-sol",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"paragraphs": []}',
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": 24,
                "cost": 0.02,
            },
        },
    ]
    monkeypatch.setattr(
        openrouter_client,
        "_get_provider_config",
        lambda provider: ("https://example.invalid", "secret", {}),
    )
    monkeypatch.setattr(openrouter_client.httpx, "Client", _FakeClient)
    handled = []

    result = openrouter_client.query_text_with_tools(
        "openai/gpt-5.6-sol",
        [{"role": "user", "content": "rewrite"}],
        [
            {
                "type": "function",
                "function": {
                    "name": "inspect_board",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        lambda name, arguments: handled.append((name, arguments)) or {"tiles": 19},
        forced_first_tool="inspect_board",
        response_format={"type": "json_object"},
    )

    assert result["content"] == '{"paragraphs": []}'
    assert result["called_tools"] == ["inspect_board"]
    assert result["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 9,
        "total_tokens": 39,
        "cost": 0.03,
    }
    assert handled == [("inspect_board", {})]
    assert _FakeClient.payloads[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "inspect_board"},
    }
    assert _FakeClient.payloads[1]["tool_choice"] == "auto"
    tool_message = _FakeClient.payloads[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"]) == {"tiles": 19}
