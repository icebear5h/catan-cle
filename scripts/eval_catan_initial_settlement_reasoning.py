#!/usr/bin/env python3
"""Capture native reasoning traces on fresh seeded first-settlement choices."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from cle.env.observation_formatter import CatanObservationFormatter
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness.board_surface import board_presentation_payload
from cle.harness.context import ContextAssembler
from cle.harness.decision import request_player_attempt
from cle.harness.models import ModelMessage, ModelRequest, PlayerSession
from cle.harness.providers import OpenRouterConfig, OpenRouterTransport
from cle.harness.reasoning import (
    native_reasoning_request,
    native_reasoning_returned,
    reasoning_token_count,
)
from cle.harness.suite import ContextSuite, load_context_suite
from cle.players.contracts import PlayerContext
from cle.sandbox.decision import build_decision_context


SCHEMA = "catan-initial-settlement-reasoning-trace/v1"
PLAN_SCHEMA = "catan-initial-settlement-reasoning-plan/v1"
COLORS = (Color.BLUE, Color.RED, Color.WHITE, Color.ORANGE)


@dataclass(frozen=True)
class SeedInput:
    context: PlayerContext
    request: ModelRequest
    manifest: dict[str, Any]
    suite: ContextSuite


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_key(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Immutable JSON artifact changed: {path}")
        return
    write_json(path, value)


def message_payload(messages: Sequence[ModelMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def load_prompt_variant(
    path: Path,
    base_suite: ContextSuite,
) -> tuple[ContextSuite, dict[str, Any]]:
    payload = read_json(path)
    expected_keys = {
        "schema",
        "id",
        "version",
        "base_context_suite",
        "phase",
        "guidance",
    }
    if set(payload) != expected_keys:
        raise ValueError("Prompt variant fields do not match the v1 contract")
    if payload["schema"] != "catan-eval-prompt-variant/v1":
        raise ValueError("Unsupported prompt variant schema")
    base_identity = f"{base_suite.id}@{base_suite.version}"
    if payload["base_context_suite"] != base_identity:
        raise ValueError("Prompt variant targets a different base context suite")
    if payload["phase"] != "initial_settlement_1":
        raise ValueError("Reasoning probe variant must target initial_settlement_1")
    if not all(
        isinstance(payload[key], str) and payload[key].strip()
        for key in ("id", "version", "guidance")
    ):
        raise ValueError("Prompt variant identity and guidance must be non-empty")

    suite_payload = base_suite.model_dump()
    suite_payload["version"] = (
        f"{base_suite.version}+{payload['id']}.{payload['version']}"
    )
    suite_payload["phase_guidance"][payload["phase"]] = payload["guidance"]
    active_suite = ContextSuite.model_validate(suite_payload)
    metadata = {
        "schema": payload["schema"],
        "id": payload["id"],
        "version": payload["version"],
        "base_context_suite": base_identity,
        "phase": payload["phase"],
        "guidance": payload["guidance"],
        "guidance_sha256": hashlib.sha256(
            payload["guidance"].encode("utf-8")
        ).hexdigest(),
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return active_suite, metadata


def legal_action_payload(context: PlayerContext) -> list[dict[str, Any]]:
    formatter = CatanObservationFormatter()
    return [
        {
            "index": index,
            "action": str(action),
            "description": formatter._format_single_action(
                action,
                context.observation,
            ),
        }
        for index, action in enumerate(context.legal_actions)
    ]


def build_seed_input(
    seed: int,
    *,
    suite: ContextSuite | None = None,
    prompt_variant: Mapping[str, Any] | None = None,
) -> SeedInput:
    engine = GameEngine(
        COLORS,
        seed=seed,
        shuffle_players=False,
        capture_history=False,
    )
    engine.id = f"fresh-initial-settlement-seed-{seed}"
    context = build_decision_context(engine, Color.BLUE, context_revision=0)
    active_suite = suite or load_context_suite()
    session = PlayerSession(
        color=Color.BLUE,
        session_id=f"fresh-initial-settlement-seed-{seed}:template",
    )
    request = ContextAssembler(active_suite).assemble(context, session)
    board = board_presentation_payload(
        request.board_presentation,
        include_text_content=True,
    )
    actions = legal_action_payload(context)
    messages = message_payload(request.messages)

    if context.actor is not Color.BLUE:
        raise RuntimeError("Fresh reasoning probe must have BLUE acting first")
    if context.phase != "initial_placement":
        raise RuntimeError("Fresh reasoning probe is not in initial placement")
    if context.prompt_key != "initial_settlement_1":
        raise RuntimeError("Fresh reasoning probe is not the first settlement")
    if context.events:
        raise RuntimeError("Fresh reasoning probe unexpectedly contains game history")
    if len(actions) != 54:
        raise RuntimeError(f"Expected 54 first-settlement actions, got {len(actions)}")

    prompt_identity = {
        "messages": messages,
        "board": board,
        "legal_actions": actions,
    }
    manifest = {
        "schema": "catan-initial-settlement-input/v1",
        "seed": seed,
        "engine_id": engine.id,
        "colors": [color.value for color in engine.state.colors],
        "actor": context.actor.value,
        "turn_number": context.turn_number,
        "phase": context.phase,
        "prompt_key": context.prompt_key,
        "context_id": context.context_id,
        "event_count": len(context.events),
        "strategic_memory": "",
        "legal_actions": actions,
        "messages": messages,
        "board_presentation": board,
        "prompt_sha256": canonical_sha256(prompt_identity),
        "legal_actions_sha256": canonical_sha256(actions),
    }
    if prompt_variant is not None:
        manifest["context_suite"] = f"{active_suite.id}@{active_suite.version}"
        manifest["prompt_variant"] = dict(prompt_variant)
    return SeedInput(
        context=context,
        request=request,
        manifest=manifest,
        suite=active_suite,
    )


def catalog_cost_bounds(
    catalog_snapshot: Mapping[str, Any],
    models: Sequence[str],
    *,
    prompt_token_allowance: int,
) -> dict[str, dict[str, Any]]:
    if prompt_token_allowance <= 0:
        raise ValueError("prompt_token_allowance must be positive")
    rows = catalog_snapshot.get("models")
    if not isinstance(rows, list):
        raise ValueError("Catalog snapshot has no models list")
    by_id = {row.get("model_id"): row for row in rows if isinstance(row, dict)}
    bounds: dict[str, dict[str, Any]] = {}
    for model_id in models:
        row = by_id.get(model_id)
        if row is None:
            raise ValueError(f"Catalog snapshot is missing {model_id}")
        pricing = row.get("pricing") or {}
        provider = row.get("top_provider") or {}
        max_completion_tokens = provider.get("max_completion_tokens")
        if not isinstance(max_completion_tokens, int) or max_completion_tokens <= 0:
            raise ValueError(
                f"Catalog has no positive max completion for {model_id}"
            )
        try:
            prompt_rate = float(pricing["prompt"])
            completion_rate = float(pricing["completion"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Catalog has invalid pricing for {model_id}") from exc
        if not all(
            math.isfinite(value) and value >= 0
            for value in (prompt_rate, completion_rate)
        ):
            raise ValueError(f"Catalog has invalid pricing for {model_id}")
        upper_bound = (
            prompt_token_allowance * prompt_rate
            + max_completion_tokens * completion_rate
        )
        bounds[model_id] = {
            "prompt_token_allowance": prompt_token_allowance,
            "max_completion_tokens": max_completion_tokens,
            "prompt_price_per_token_usd": prompt_rate,
            "completion_price_per_token_usd": completion_rate,
            "request_upper_bound_usd": upper_bound,
        }
    return bounds


def usage_cost_usd(trace: Mapping[str, Any]) -> float:
    response = trace.get("response")
    usage = response.get("usage") if isinstance(response, dict) else None
    value = usage.get("cost") if isinstance(usage, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return 0.0
    return float(value)


def trace_path(output_dir: Path, seed: int, model_id: str) -> Path:
    return output_dir / "traces" / f"seed_{seed}" / f"{model_key(model_id)}.json"


def existing_traces(output_dir: Path) -> list[dict[str, Any]]:
    traces = []
    for path in sorted((output_dir / "traces").glob("seed_*/*.json")):
        traces.append(read_json(path))
    return traces


def _choice_payload(attempt: Any, context: PlayerContext) -> dict[str, Any] | None:
    choice = attempt.choice
    if choice is None:
        return None
    actions = legal_action_payload(context)
    selected = actions[choice.action_index]
    return {
        "action_index": choice.action_index,
        "action": selected["action"],
        "action_description": selected["description"],
        "game_plan": choice.game_plan,
        "parse_warning": choice.parse_warning,
    }


async def capture_trace(
    seed_input: SeedInput,
    model_id: str,
    *,
    reasoning_effort: str,
    temperature: float,
    api_key: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    reasoning = native_reasoning_request(reasoning_effort)
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model=model_id,
            temperature=temperature,
            max_tokens=None,
            timeout_seconds=900.0,
            max_retries=0,
            reasoning=reasoning,
        ),
        api_key=api_key,
        client=client,
    )
    try:
        attempt = await request_player_attempt(
            seed_input.context,
            transport,
            game_plan="",
            session_id=(
                f"fresh-initial-settlement-seed-{seed_input.manifest['seed']}:"
                f"{model_id}"
            ),
            suite=seed_input.suite,
        )
    finally:
        await transport.aclose()

    response = attempt.model_response
    request = attempt.model_request
    if response is None or request is None:
        raise RuntimeError("Reasoning probe did not retain its provider exchange")
    provider_request = response.provider_request_payload
    if not isinstance(provider_request, dict):
        raise RuntimeError("Reasoning probe did not retain provider request metadata")
    if "max_tokens" in provider_request:
        raise RuntimeError("Uncapped reasoning request unexpectedly sent max_tokens")
    if message_payload(request.messages) != seed_input.manifest["messages"]:
        raise RuntimeError("Model-specific reasoning prompt changed")
    board = board_presentation_payload(
        request.board_presentation,
        include_text_content=True,
    )
    if board != seed_input.manifest["board_presentation"]:
        raise RuntimeError("Model-specific board presentation changed")

    usage = dict(response.usage)
    trace = {
        "schema": SCHEMA,
        "trace_id": (
            f"fresh-initial-settlement-seed-{seed_input.manifest['seed']}:"
            f"{model_id}"
        ),
        "recorded_at": utc_now(),
        "input": seed_input.manifest,
        "request": {
            "requested_model": model_id,
            "temperature": temperature,
            "reasoning": reasoning,
            "max_tokens_omitted": True,
            "provider_payload": provider_request,
        },
        "response": {
            "served_model": response.model,
            "provider_response_id": response.provider_response_id,
            "provider_request_id": response.provider_request_id,
            "finish_reason": response.finish_reason,
            "provider_native_finish_reason": response.provider_native_finish_reason,
            "latency_ms": response.latency_ms,
            "usage": usage,
            "reasoning_tokens": reasoning_token_count(usage),
            "native_reasoning_returned": native_reasoning_returned(
                response.native_reasoning,
                response.native_reasoning_details,
                usage,
            ),
            "native_reasoning": response.native_reasoning,
            "native_reasoning_details": list(response.native_reasoning_details),
            "final_response": response.content,
            "provider_payload": response.provider_response_payload,
        },
        "parse": {
            "error": attempt.validation_error,
            "choice": _choice_payload(attempt, seed_input.context),
        },
    }
    return trace


def markdown_fence(value: str, language: str = "text") -> str:
    return f"~~~~{language}\n{value}\n~~~~"


def render_trace_markdown(trace: Mapping[str, Any]) -> str:
    request = trace["request"]
    response = trace["response"]
    parsed = trace["parse"]
    input_record = trace["input"]
    lines = [
        f"# {request['requested_model']} — seed {input_record['seed']}",
        "",
        f"- Trace: `{trace['trace_id']}`",
        f"- Prompt SHA-256: `{input_record['prompt_sha256']}`",
        f"- Board SHA-256: `{input_record['board_presentation']['board_sha256']}`",
        f"- Reasoning request: `{json.dumps(request['reasoning'], sort_keys=True)}`",
        "- `max_tokens`: omitted",
        f"- Served model: `{response['served_model']}`",
        f"- Finish reason: `{response['finish_reason']}`",
        f"- Native finish reason: `{response['provider_native_finish_reason']}`",
        f"- Latency: `{response['latency_ms']} ms`",
        f"- Usage: `{json.dumps(response['usage'], sort_keys=True)}`",
        "",
        "## Exact provider request",
        "",
        markdown_fence(
            json.dumps(request["provider_payload"], indent=2, ensure_ascii=False),
            "json",
        ),
        "",
        "## Native provider reasoning",
        "",
        markdown_fence(response["native_reasoning"]),
        "",
        "## Final model response",
        "",
        markdown_fence(response["final_response"]),
        "",
        "## Parse convenience view",
        "",
        markdown_fence(json.dumps(parsed, indent=2, ensure_ascii=False), "json"),
        "",
    ]
    return "\n".join(lines)


def write_trace(output_dir: Path, trace: Mapping[str, Any]) -> None:
    seed = trace["input"]["seed"]
    model_id = trace["request"]["requested_model"]
    path = trace_path(output_dir, seed, model_id)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing trace: {path}")
    write_json(path, trace)
    path.with_suffix(".md").write_text(
        render_trace_markdown(trace),
        encoding="utf-8",
    )


def write_indexes(output_dir: Path, plan: Mapping[str, Any]) -> None:
    traces = existing_traces(output_dir)
    rows = []
    for trace in traces:
        response = trace["response"]
        parsed = trace["parse"]
        rows.append(
            {
                "trace_id": trace["trace_id"],
                "seed": trace["input"]["seed"],
                "model_id": trace["request"]["requested_model"],
                "prompt_sha256": trace["input"]["prompt_sha256"],
                "finish_reason": response["finish_reason"],
                "native_reasoning_returned": response[
                    "native_reasoning_returned"
                ],
                "reasoning_tokens": response["reasoning_tokens"],
                "final_response_characters": len(response["final_response"]),
                "parsed_action_index": (
                    parsed["choice"]["action_index"]
                    if parsed["choice"] is not None
                    else None
                ),
                "parse_error": parsed["error"],
                "latency_ms": response["latency_ms"],
                "cost_usd": usage_cost_usd(trace),
                "json_path": str(
                    trace_path(
                        output_dir,
                        trace["input"]["seed"],
                        trace["request"]["requested_model"],
                    ).relative_to(output_dir)
                ),
            }
        )
    index_path = output_dir / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    finish_reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row["finish_reason"])
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    summary = {
        "schema": "catan-initial-settlement-reasoning-summary/v1",
        "updated_at": utc_now(),
        "expected_trace_count": len(plan["seeds"]) * len(plan["models"]),
        "trace_count": len(rows),
        "complete": len(rows) == len(plan["seeds"]) * len(plan["models"]),
        "recorded_cost_usd": sum(row["cost_usd"] for row in rows),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "native_reasoning_trace_count": sum(
            row["native_reasoning_returned"] for row in rows
        ),
        "final_response_count": sum(
            row["final_response_characters"] > 0 for row in rows
        ),
        "parsed_action_count": sum(
            row["parsed_action_index"] is not None for row in rows
        ),
        "traces": rows,
    }
    write_json(output_dir / "summary.json", summary)
    links = [
        "# Initial-settlement native reasoning traces",
        "",
        "Fresh seeded games only. There is no replay history, human label, or agreement score.",
        "`max_tokens` was omitted from every provider request.",
        "",
    ]
    for row in rows:
        markdown_path = Path(row["json_path"]).with_suffix(".md")
        links.append(
            f"- Seed {row['seed']} — [{row['model_id']}]({markdown_path}) "
            f"— finish `{row['finish_reason']}`, reasoning tokens "
            f"`{row['reasoning_tokens']}`, cost `${row['cost_usd']:.6f}`"
        )
    (output_dir / "README.md").write_text("\n".join(links) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--max-seeds",
        type=int,
        help="Run only the first N planned seeds while keeping one immutable plan.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog-snapshot", type=Path, required=True)
    parser.add_argument(
        "--prompt-variant",
        type=Path,
        help="Eval-only phase-guidance variant; the default suite is not mutated.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--prior-cost-usd", type=float, default=0.0)
    parser.add_argument("--prompt-token-allowance", type=int, default=20_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be within 0..2")
    if args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be positive")
    if args.prior_cost_usd < 0:
        parser.error("--prior-cost-usd cannot be negative")
    if args.prior_cost_usd >= args.max_cost_usd:
        parser.error("--prior-cost-usd must be below --max-cost-usd")
    if args.prompt_token_allowance <= 0:
        parser.error("--prompt-token-allowance must be positive")
    if len(args.models) != len(set(args.models)):
        parser.error("--models cannot contain duplicates")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds cannot contain duplicates")
    if args.max_seeds is not None and args.max_seeds <= 0:
        parser.error("--max-seeds must be positive")
    return args


def main() -> int:
    args = parse_args()
    base_suite = load_context_suite()
    suite = base_suite
    prompt_variant = None
    if args.prompt_variant is not None:
        suite, prompt_variant = load_prompt_variant(args.prompt_variant, base_suite)
    catalog = read_json(args.catalog_snapshot)
    bounds = catalog_cost_bounds(
        catalog,
        args.models,
        prompt_token_allowance=args.prompt_token_allowance,
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "generated_at": utc_now(),
        "models": args.models,
        "seeds": args.seeds,
        "colors": [color.value for color in COLORS],
        "actor": Color.BLUE.value,
        "decision": "fresh first settlement",
        "context_suite": f"{suite.id}@{suite.version}",
        "board_surface": "indexed_tile_rows/v3",
        "reasoning_request": native_reasoning_request(args.reasoning_effort),
        "temperature": args.temperature,
        "max_tokens_omitted": True,
        "timeout_seconds": 900.0,
        "max_retries": 0,
        "max_cost_usd": args.max_cost_usd,
        "prior_cost_usd": args.prior_cost_usd,
        "prompt_token_allowance": args.prompt_token_allowance,
        "catalog_snapshot": str(args.catalog_snapshot),
        "request_cost_upper_bounds": bounds,
        "scheduling": "sequential",
        "human_action_labels": False,
        "replay_input": False,
    }
    if prompt_variant is not None:
        plan["prompt_variant"] = prompt_variant
    plan_path = args.output_dir / "plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        comparable = {key: value for key, value in plan.items() if key != "generated_at"}
        existing_comparable = {
            key: value for key, value in existing.items() if key != "generated_at"
        }
        if existing_comparable != comparable:
            raise RuntimeError("Cannot resume after direct reasoning plan changed")
    else:
        write_json(plan_path, plan)

    inputs = [
        build_seed_input(
            seed,
            suite=suite,
            prompt_variant=prompt_variant,
        )
        for seed in args.seeds
    ]
    for seed_input in inputs:
        write_json_once(
            args.output_dir / "inputs" / f"seed_{seed_input.manifest['seed']}.json",
            seed_input.manifest,
        )
    write_indexes(args.output_dir, plan)
    if args.dry_run:
        return 0

    load_dotenv()
    spent = args.prior_cost_usd + sum(
        usage_cost_usd(trace) for trace in existing_traces(args.output_dir)
    )
    selected_inputs = inputs[: args.max_seeds] if args.max_seeds else inputs
    for seed_input in selected_inputs:
        for model_id in args.models:
            path = trace_path(args.output_dir, seed_input.manifest["seed"], model_id)
            if path.exists():
                continue
            request_bound = bounds[model_id]["request_upper_bound_usd"]
            if spent + request_bound > args.max_cost_usd:
                raise RuntimeError(
                    f"Worst-case request for {model_id} would exceed the cost cap: "
                    f"${spent:.6f} spent/reserved + ${request_bound:.6f} bound > "
                    f"${args.max_cost_usd:.6f}"
                )
            print(
                f"seed={seed_input.manifest['seed']} model={model_id} "
                f"worst_case=${request_bound:.6f}",
                flush=True,
            )
            trace = asyncio.run(
                capture_trace(
                    seed_input,
                    model_id,
                    reasoning_effort=args.reasoning_effort,
                    temperature=args.temperature,
                )
            )
            write_trace(args.output_dir, trace)
            spent += usage_cost_usd(trace)
            if spent > args.max_cost_usd:
                raise RuntimeError("Recorded cost exceeded the direct-trace budget")
            write_indexes(args.output_dir, plan)

    write_indexes(args.output_dir, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
