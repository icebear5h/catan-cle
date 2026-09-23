"""Recorded spend across the sweep's response and trace artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from scripts.board_bench.run.run_catan_openrouter_model_sweep.catalog import read_json
from scripts.board_bench.shapes import JsonDict

__all__ = ["read_jsonl", "response_cost_usd", "track_cost"]


def response_cost_usd(row: JsonDict) -> float:
    usage = row.get("usage")
    if not isinstance(usage, dict):
        result = row.get("result")
        usage = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(usage, dict):
        response = row.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
    value = usage.get("cost") if isinstance(usage, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if value >= 0 else 0.0


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append(row)
    return rows


def track_cost(output_dir: Path, track: str) -> float:
    root = output_dir / track
    if not root.exists():
        return 0.0
    response_rows: Iterator[JsonDict] = (
        row
        for path in root.rglob("responses.jsonl")
        for row in read_jsonl(path)
    )
    trace_rows: Iterator[JsonDict] = (
        (read_json(path) for path in (root / "traces").glob("seed_*/*.json"))
        if (root / "traces").exists()
        else iter(())
    )
    return sum(response_cost_usd(row) for row in response_rows) + sum(
        response_cost_usd(row) for row in trace_rows
    )
