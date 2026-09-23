"""Catalog-derived cost bounds and recorded spend for the reasoning probe."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from cle.players.data import JsonValue
from scripts.reasoning.eval_catan_initial_settlement_reasoning.artifacts import JsonDict

__all__ = ["catalog_cost_bounds", "usage_cost_usd"]


def _rate(pricing: Mapping[str, JsonValue], key: str) -> float:
    """Read one price, raising the same errors ``float()`` would raise."""
    value = pricing[key]
    if isinstance(value, (str, int, float)):
        return float(value)
    raise TypeError(f"Catalog pricing {key} is not numeric")


def catalog_cost_bounds(
    catalog_snapshot: Mapping[str, JsonValue],
    models: Sequence[str],
    *,
    prompt_token_allowance: int,
) -> dict[str, JsonDict]:
    if prompt_token_allowance <= 0:
        raise ValueError("prompt_token_allowance must be positive")
    rows = catalog_snapshot.get("models")
    if not isinstance(rows, list):
        raise ValueError("Catalog snapshot has no models list")
    by_id = {row.get("model_id"): row for row in rows if isinstance(row, dict)}
    bounds: dict[str, JsonDict] = {}
    for model_id in models:
        row = by_id.get(model_id)
        if row is None:
            raise ValueError(f"Catalog snapshot is missing {model_id}")
        pricing = row.get("pricing") or {}
        provider = row.get("top_provider") or {}
        if not isinstance(pricing, dict) or not isinstance(provider, dict):
            raise ValueError(f"Catalog has invalid pricing for {model_id}")
        max_completion_tokens = provider.get("max_completion_tokens")
        if not isinstance(max_completion_tokens, int) or max_completion_tokens <= 0:
            raise ValueError(f"Catalog has no positive max completion for {model_id}")
        try:
            prompt_rate = _rate(pricing, "prompt")
            completion_rate = _rate(pricing, "completion")
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


def usage_cost_usd(trace: Mapping[str, JsonValue]) -> float:
    response = trace.get("response")
    usage = response.get("usage") if isinstance(response, dict) else None
    value = usage.get("cost") if isinstance(usage, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return 0.0
    return float(value)
