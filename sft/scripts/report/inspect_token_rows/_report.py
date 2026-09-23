"""Assemble the per-adapter row report and its console table."""

from __future__ import annotations

from pathlib import Path

from sft.json_types import as_dict, as_float, as_list, opt_float

from ._base import (
    DEFAULT_FAMILY_FLOOR,
    DEFAULT_TWIN_THRESHOLD,
    FAMILIES,
    FOCUS_TOKEN,
    SIDE_KEYS,
    TABLE_FLOOR_PREVIEW,
    JsonDict,
)
from ._rows import inspect_rows, load_token_inventory, load_token_rows


def inspect_adapter(
    path: str | Path,
    tokens: list[str],
    *,
    twin_threshold: float = DEFAULT_TWIN_THRESHOLD,
    family_floor: float = DEFAULT_FAMILY_FLOOR,
) -> JsonDict:
    """Inspect both trainable-token sides of one adapter_model.safetensors."""
    return {
        side: inspect_rows(tensor, tokens, twin_threshold=twin_threshold, family_floor=family_floor)
        for side, tensor in load_token_rows(path).items()
    }


def build_report(
    adapters: list[tuple[str, Path]],
    token_inventory: Path,
    *,
    twin_threshold: float = DEFAULT_TWIN_THRESHOLD,
    family_floor: float = DEFAULT_FAMILY_FLOOR,
) -> JsonDict:
    tokens = load_token_inventory(token_inventory)
    entries: JsonDict = {}
    for label, path in adapters:
        if label in entries:
            raise ValueError(f"duplicate adapter label {label!r}")
        entries[label] = {
            "path": str(path),
            **inspect_adapter(
                path, tokens, twin_threshold=twin_threshold, family_floor=family_floor
            ),
        }
    return {
        "token_inventory": str(token_inventory),
        "token_count": len(tokens),
        "twin_threshold": twin_threshold,
        "family_floor": family_floor,
        "adapters": entries,
    }


def format_table(report: JsonDict) -> str:
    """One line per adapter and side, compact enough to scan on stderr."""
    adapters = as_dict(report["adapters"])
    width = max(len("adapter"), *(len(label) for label in adapters))
    header = (
        f"{'adapter':<{width}}  {'side':<6}  {'pair mean/sd':<13}  twins  "
        f"{'own-family median N/E/T/P':<27}  {FOCUS_TOKEN:>6}  "
        f"below family floor {report['family_floor']:g}"
    )
    lines = [header]
    for label, entry in adapters.items():
        for side in SIDE_KEYS:
            result = as_dict(as_dict(entry)[side])
            medians = as_dict(result["own_family_median"])
            median_text = "/".join(
                f"{as_float(medians[family]):+.2f}" if family in medians else "-"
                for family in FAMILIES
            )
            focus = opt_float(as_dict(result["own_family_loading"]).get(FOCUS_TOKEN))
            focus_text = f"{focus:+.3f}" if focus is not None else "n/a"
            below = [as_list(item) for item in as_list(result["rows_below_family_floor"])]
            below_text = ", ".join(
                f"{token} {as_float(loading):+.2f}" for token, loading in below[:TABLE_FLOOR_PREVIEW]
            )
            if len(below) > TABLE_FLOOR_PREVIEW:
                below_text += ", ..."
            pair_text = (
                f"{as_float(result['pairwise_cosine_mean']):.3f}/"
                f"{as_float(result['pairwise_cosine_sd']):.3f}"
            )
            lines.append(
                f"{label:<{width}}  {side:<6}  {pair_text:<13}  {result['tokens_with_twin']:>5}  "
                f"{median_text:<27}  {focus_text:>6}  {len(below)}: {below_text}"
            )
    return "\n".join(lines)
