import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from cle.harness.suite import load_context_suite
from scripts.reasoning.eval_catan_initial_settlement_reasoning import (
    build_seed_input,
    capture_trace,
    catalog_cost_bounds,
    load_prompt_variant,
    render_trace_markdown,
)

ROOT = Path(__file__).resolve().parents[3]


def test_fresh_initial_settlement_input_is_deterministic_and_history_free() -> None:
    first: Any = build_seed_input(117)
    second = build_seed_input(117)

    assert first.manifest == second.manifest
    assert first.manifest["engine_id"] == "fresh-initial-settlement-seed-117"
    assert first.manifest["colors"] == ["BLUE", "RED", "WHITE", "ORANGE"]
    assert first.manifest["actor"] == "BLUE"
    assert first.manifest["phase"] == "initial_placement"
    assert first.manifest["prompt_key"] == "initial_settlement_1"
    assert first.manifest["event_count"] == 0
    assert first.manifest["strategic_memory"] == ""
    model_prompt: Any = first.manifest["messages"][-1]["content"]
    assert "Round 1 (first settlement + road): BLUE -> RED -> WHITE -> ORANGE" in model_prompt
    assert "Round 2 (second settlement + road): ORANGE -> WHITE -> RED -> BLUE" in model_prompt
    assert "Your positions: round 1 = 1/4; round 2 = 4/4." in model_prompt
    assert len(first.manifest["legal_actions"]) == 54
    assert len(first.manifest["prompt_sha256"]) == 64
    assert first.manifest["board_presentation"]["kind"] == "text"
    assert first.manifest["board_presentation"]["identity_space"] == (
        "canonical_engine_ids"
    )


def test_strategy_variant_changes_only_the_prompt_contract() -> None:
    variant_path = ROOT / "evals/suites/catan_initial_settlement_strategy_v2.json"
    base_suite = load_context_suite(
        ROOT / "cle/harness/suites/catan_v8.yaml"
    )
    guided_suite, variant = load_prompt_variant(variant_path, base_suite)
    baseline: Any = build_seed_input(117, suite=base_suite)
    guided: Any = build_seed_input(
        117,
        suite=guided_suite,
        prompt_variant=variant,
    )

    assert guided.manifest["board_presentation"] == baseline.manifest[
        "board_presentation"
    ]
    assert guided.manifest["legal_actions_sha256"] == baseline.manifest[
        "legal_actions_sha256"
    ]
    assert guided.manifest["prompt_sha256"] != baseline.manifest["prompt_sha256"]
    assert guided.manifest["prompt_variant"]["id"] == (
        "initial-settlement-strategy"
    )
    assert guided.manifest["context_suite"].startswith("catan-agent@8.0.0+")
    guided_prompt: Any = guided.manifest["messages"][-1]["content"]
    assert "Maximizing raw pip count is not the objective" in guided_prompt
    assert "opening archetype" in (
        guided_prompt
    )
    assert "Maximizing raw pip count is not the objective" not in (
        baseline.manifest["messages"][-1]["content"]
    )

    historical_v9 = build_seed_input(
        117, suite=load_context_suite(ROOT / "cle/harness/suites/catan_v9.yaml")
    )
    assert historical_v9.manifest["messages"] != guided.manifest["messages"]

    current_default: Any = build_seed_input(117)
    assert current_default.suite.version == "11.0.0"
    assert current_default.suite.response.format == "json"
    assert current_default.suite.response.tags == ("game_plan", "tool", "arguments")
    assert current_default.suite.phase_guidance["initial_settlement_1"] == (
        guided_suite.phase_guidance["initial_settlement_1"]
    )
    current_prompt: Any = current_default.manifest["messages"][-1]["content"]
    assert '"tool"' in current_prompt
    assert '"arguments"' in current_prompt
    assert "<N00>" in current_prompt
    assert "action_index" not in current_prompt
    assert "<action>" not in current_prompt
    assert current_default.manifest["messages"] != guided.manifest["messages"]
    assert current_default.manifest["board_presentation"] == (
        guided.manifest["board_presentation"]
    )
    assert current_default.manifest["legal_actions_sha256"] == (
        guided.manifest["legal_actions_sha256"]
    )


def test_uncapped_cost_bounds_use_catalog_max_completion_and_prompt_allowance() -> None:
    catalog: Any = {
        "models": [
            {
                "model_id": "model/a",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "top_provider": {"max_completion_tokens": 10_000},
            }
        ]
    }

    bounds = catalog_cost_bounds(catalog, ["model/a"], prompt_token_allowance=20_000)

    assert bounds["model/a"]["request_upper_bound_usd"] == pytest.approx(0.04)
    assert bounds["model/a"]["max_completion_tokens"] == 10_000


@pytest.mark.asyncio
async def test_capture_trace_omits_max_tokens_and_separates_native_reasoning() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "request-1"},
            json={
                "id": "response-1",
                "model": "model/a",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "native_finish_reason": "stop",
                        "message": {
                            "reasoning": "private native reasoning",
                            "reasoning_details": [
                                {
                                    "type": "reasoning.text",
                                    "text": "private native reasoning",
                                }
                            ],
                            "content": (
                                "<game_plan>Prefer production diversity.</game_plan>\n"
                                "<action>7</action>"
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost": 0.001,
                    "completion_tokens_details": {"reasoning_tokens": 12},
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    base_suite = load_context_suite(
        ROOT / "cle/harness/suites/catan_v8.yaml"
    )
    guided_suite, variant = load_prompt_variant(
        ROOT / "evals/suites/catan_initial_settlement_strategy_v2.json",
        base_suite,
    )
    trace: Any = await capture_trace(
        build_seed_input(117, suite=guided_suite, prompt_variant=variant),
        "model/a",
        reasoning_effort="high",
        temperature=0.2,
        api_key="test-key",
        client=client,
    )
    await client.aclose()

    assert "max_tokens" not in captured["payload"]
    assert "Maximizing raw pip count is not the objective" in (
        captured["payload"]["messages"][-1]["content"]
    )
    assert trace["request"]["max_tokens_omitted"] is True
    assert trace["request"]["reasoning"] == {
        "effort": "high",
        "exclude": False,
    }
    assert trace["response"]["native_reasoning"] == "private native reasoning"
    assert trace["response"]["final_response"].endswith("<action>7</action>")
    assert trace["response"]["reasoning_tokens"] == 12
    assert trace["response"]["finish_reason"] == "stop"
    assert trace["parse"]["choice"]["action_index"] == 7
    assert "human" not in json.dumps(trace).lower()
    assert "agreement" not in json.dumps(trace).lower()

    markdown = render_trace_markdown(trace)
    assert "## Native provider reasoning" in markdown
    assert "## Final model response" in markdown
    assert markdown.index("private native reasoning") < markdown.index("<action>7")
