"""Local, Modal-free helpers behind the gradient-conflict probe launcher."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict

from sft.analysis.gradient_diagnostics import (
    SelectedProbeRow,
    iter_jsonl,
    select_probe_rows,
    selection_manifest,
)
from sft.json_types import JsonDict, JsonLike, as_int, as_str, loads_json

__all__ = [
    "JsonDict",
    "ProbeArgs",
    "build_local_selection",
    "canonical_hash",
    "probe_args_from_json",
    "read_json_object",
    "write_json_atomic",
]

class ProbeArgs(TypedDict):
    """Keyword arguments of `run_gradient_probe`, as recorded in the launch payload."""

    adapter_dir: str
    probe_jsonl: str
    image_root: str
    token_inventory: str
    selection_manifest_path: str
    output_dir: str
    microbatch_size: NotRequired[int]


_PROBE_ARG_TEXT = ("adapter_dir", "probe_jsonl", "image_root", "token_inventory",
                   "selection_manifest_path", "output_dir")


def probe_args_from_json(payload: Mapping[str, JsonLike]) -> ProbeArgs:
    """Narrow decoded probe arguments; unknown keys fail like `**` into the probe would."""
    unknown = sorted(set(payload) - {*_PROBE_ARG_TEXT, "microbatch_size"})
    if unknown:
        raise TypeError(f"run_gradient_probe got unexpected arguments {unknown}")
    args = ProbeArgs(
        adapter_dir=as_str(payload["adapter_dir"]), probe_jsonl=as_str(payload["probe_jsonl"]),
        image_root=as_str(payload["image_root"]), token_inventory=as_str(payload["token_inventory"]),
        selection_manifest_path=as_str(payload["selection_manifest_path"]),
        output_dir=as_str(payload["output_dir"]),
    )
    if "microbatch_size" in payload:
        args["microbatch_size"] = as_int(payload["microbatch_size"])
    return args


def canonical_hash(payload: Mapping[str, JsonLike]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json_object(path: Path) -> JsonDict:
    loaded = loads_json(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return loaded


def write_json_atomic(path: Path, payload: Mapping[str, JsonLike]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_local_selection(
    pair_jsonl: Path,
    single_jsonl: Path,
    *,
    seed: int,
    rows_per_behavior: int,
) -> tuple[list[SelectedProbeRow], JsonDict]:
    selected = select_probe_rows(
        list(iter_jsonl(pair_jsonl)),
        list(iter_jsonl(single_jsonl)),
        rows_per_behavior=rows_per_behavior,
        seed=seed,
    )
    manifest = selection_manifest(
        selected,
        seed=seed,
        rows_per_behavior=rows_per_behavior,
    )
    return selected, manifest
