"""Sweep configuration defaults, catalog fetching, and config validation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from scripts.board_bench.shapes import JsonDict, number, obj, objs, values

DEFAULT_CONFIG = Path(
    "configs/model_sweeps/openrouter_vlm_27b_72b_v1.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/runs/model_selection/openrouter_vlm_27b_72b_20260831"
)
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "OPENROUTER_MODELS_URL",
    "fetch_catalog",
    "model_key",
    "read_json",
    "validate_config",
    "write_json",
]


def read_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def model_key(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def fetch_catalog() -> JsonDict:
    response = httpx.get(OPENROUTER_MODELS_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("OpenRouter model catalog has an unexpected shape")
    return payload


def validate_config(
    config: JsonDict,
    catalog: JsonDict,
) -> list[JsonDict]:
    if config.get("schema") != "catan-openrouter-model-sweep/v1":
        raise ValueError("Unsupported model-sweep config schema")
    budget = config.get("authorized_budget_usd")
    allocations = obj(config.get("budget_allocation_usd") or {}, "budget_allocation_usd")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("authorized_budget_usd must be positive")
    allocated = sum(
        number(value, "budget allocation") for value in allocations.values()
    )
    if allocated > budget:
        raise ValueError("Track budget allocations exceed the authorized budget")

    by_id = {str(row.get("id")): row for row in objs(catalog["data"], "catalog data")}
    raw_candidates = config.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("Model sweep has no candidates")
    candidates = objs(raw_candidates, "candidates")
    seen: set[str] = set()
    selected_catalog: list[JsonDict] = []
    for candidate in candidates:
        model_id = candidate.get("model_id")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            raise ValueError(f"Invalid or duplicate model ID: {model_id!r}")
        seen.add(model_id)
        model = by_id.get(model_id)
        if model is None:
            raise ValueError(f"OpenRouter catalog does not contain {model_id}")
        architecture = obj(model.get("architecture") or {}, "architecture")
        modalities = values(architecture.get("input_modalities") or [], "input_modalities")
        if "image" not in modalities:
            raise ValueError(f"Candidate does not advertise image input: {model_id}")
        selected_catalog.append(
            {
                "model_id": model_id,
                "name": model.get("name"),
                "context_length": model.get("context_length"),
                "architecture": architecture,
                "supported_parameters": model.get("supported_parameters") or [],
                "pricing": model.get("pricing") or {},
                "top_provider": model.get("top_provider") or {},
            }
        )
    return selected_catalog
