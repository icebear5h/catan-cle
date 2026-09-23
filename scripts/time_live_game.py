#!/usr/bin/env python3
"""Play one headless live LLM game, save it for the viewer, and report its speed.

Drives the viewer's own /api/start-game and /api/step routes in-process, so the
game lands in the live trace database exactly as a browser game would and can be
opened from the viewer's saved-game list while it runs or afterwards.

    uv run python scripts/time_live_game.py --seed 1
    uv run python scripts/time_live_game.py --model qwen/qwen3.8-27b --reasoning off
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from dotenv import load_dotenv
from flask import Flask

from cle.harness.reasoning import native_reasoning_request, reasoning_token_count
from cle.replay.contracts import GameEngineHolder
from cle.traces import SQLiteLiveTraceStore
from playground.game_viewer.routes.live_game import live_game_bp
from playground.game_viewer.state import ServerState

load_dotenv()


@dataclass
class _NoBrowserSocket:
    """The routes broadcast state to browsers; a headless run has none."""

    emissions: int = field(default=0)

    def emit(self, event: str, payload: object) -> None:
        self.emissions += 1


# playground.game_viewer is unannotated; these typed handles keep the calls and
# attribute reads here checkable without changing what runs.
_new_server_state: Callable[[], ServerState] = ServerState


def _turns_played(state: ServerState) -> int:
    sandbox = state.current_sandbox
    if sandbox is None:
        raise RuntimeError("no active sandbox on the viewer server state")
    # The viewer stores the sandbox as ``object``; the repo's typed contract for
    # reading its engine back out is GameEngineHolder.
    holder = cast(GameEngineHolder, sandbox)
    return int(holder.game_engine.state.num_turns)


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _model_call_rows(store: SQLiteLiveTraceStore, game_id: str) -> list[sqlite3.Row]:
    connection = sqlite3.connect(store.path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            """
            SELECT accepted,
                   json_extract(response_json, '$.latency_ms') AS latency_ms,
                   json_extract(response_json, '$.usage') AS usage
            FROM model_calls
            WHERE game_id = ? AND response_json IS NOT NULL
            """,
            (game_id,),
        ).fetchall()
    finally:
        connection.close()


def _report(
    args: argparse.Namespace,
    store: SQLiteLiveTraceStore,
    game_id: str,
    *,
    wall_seconds: float,
    steps: int,
    failed_steps: int,
    turns: int,
    outcome: str,
) -> None:
    rows = _model_call_rows(store, game_id)
    latencies = [int(row["latency_ms"]) for row in rows if row["latency_ms"] is not None]
    usages = [json.loads(row["usage"]) if row["usage"] else {} for row in rows]
    prompt_tokens = sum(u.get("prompt_tokens") or 0 for u in usages)
    completion_tokens = sum(u.get("completion_tokens") or 0 for u in usages)
    reasoning_tokens = sum(reasoning_token_count(u) or 0 for u in usages)
    rejected = sum(1 for row in rows if not row["accepted"])
    inference_seconds = sum(latencies) / 1000

    print(f"\nmodel            {args.model}  (reasoning={args.reasoning}, seed={args.seed})")
    print(f"trace            {game_id}  in {store.path}")
    print(f"outcome          {outcome}")
    print(f"wall time        {wall_seconds:.1f}s  ({wall_seconds / 60:.2f} min)")
    print(f"turns / steps    {turns} / {steps}  ({failed_steps} failed steps retried)")
    # Calls from failed steps live in live_failures and are not counted here.
    print(f"recorded calls   {len(rows)}  ({rejected} rejected attempts)")
    if latencies:
        print(
            f"latency ms       mean {statistics.fmean(latencies):.0f}  "
            f"p50 {_percentile(latencies, 0.5)}  p95 {_percentile(latencies, 0.95)}  "
            f"max {max(latencies)}"
        )
    print(
        f"tokens           prompt {prompt_tokens:,}  completion {completion_tokens:,}  "
        f"(reasoning {reasoning_tokens:,})"
    )
    if inference_seconds > 0:
        # Summed latency exceeds wall time when barrier steps run calls in parallel.
        print(
            f"throughput       {completion_tokens / inference_seconds:,.0f} completion tok/s "
            f"per call, {completion_tokens / wall_seconds:,.0f} tok/s of wall time"
        )


def run(args: argparse.Namespace) -> int:
    store = SQLiteLiveTraceStore(
        os.getenv("CATAN_LIVE_TRACE_DB", ".cle/live_traces.sqlite3")
    )
    state = _new_server_state()
    state.live_trace_store = store
    app = Flask(__name__)
    app.config["SERVER_STATE"] = state
    app.config["SOCKETIO"] = _NoBrowserSocket()
    app.register_blueprint(live_game_bp)
    client = app.test_client()

    name = args.name or f"headless {args.model} r={args.reasoning} seed={args.seed}"
    started = client.post(
        "/api/start-game",
        json={
            "mode": "llm",
            "model": args.model,
            "seed": args.seed,
            "reasoning": native_reasoning_request(args.reasoning),
            "name": name,
        },
    )
    if started.status_code != 200:
        print(f"could not start game: {started.get_json()}")
        return 1
    game_id = started.get_json()["trace_game_id"]
    print(f"recording as {name!r} ({game_id}); open it from the viewer's saved games", flush=True)

    steps = 0
    failed_steps = 0
    consecutive_failures = 0
    outcome = f"stopped at --max-steps {args.max_steps}"
    exit_code = 1
    started_at = time.monotonic()
    while steps < args.max_steps:
        response = client.post("/api/step")
        body = response.get_json()
        if response.status_code != 200:
            failed_steps += 1
            consecutive_failures += 1
            # Same policy as viewer auto-play: a retryable failure is a fresh step.
            if body.get("retryable") and consecutive_failures <= args.max_step_retries:
                continue
            outcome = f"aborted ({response.status_code}): {body.get('error')}: {body.get('details')}"
            break
        consecutive_failures = 0
        steps += 1
        if body["game_over"]:
            outcome = f"{body['winner']} won"
            exit_code = 0
            break
        if steps % 25 == 0:
            print(
                f"  step {steps}  turn {_turns_played(state)}  "
                f"{time.monotonic() - started_at:.0f}s",
                flush=True,
            )
    wall_seconds = time.monotonic() - started_at
    if exit_code != 0:
        store.mark_game_status(game_id, status="stopped")

    _report(
        args,
        store,
        game_id,
        wall_seconds=wall_seconds,
        steps=steps,
        failed_steps=failed_steps,
        turns=_turns_played(state),
        outcome=outcome,
    )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="cerebras/qwen-3.8-27b")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--reasoning",
        default="high",
        choices=["off", "low", "medium", "high"],
    )
    parser.add_argument("--name", help="Saved-game name shown in the viewer")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--max-step-retries",
        type=int,
        default=5,
        help="Consecutive retryable step failures tolerated before aborting",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
