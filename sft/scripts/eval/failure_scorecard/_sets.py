"""sets."""

from __future__ import annotations

from pathlib import Path

from sft.json_types import as_dict, as_list, as_str, load_json_dict, loads_json
from sft.scripts.eval import failure_scorecard

from ._base import FIRST_PANEL_SET, JsonDict


def set_id_for(records_path: Path, info: JsonDict, metadata: JsonDict, overrides: dict[str, Path]) -> str | None:
    """Identify the eval set of one records file.

    Newer summaries carry ``eval_set_id``. Older ones only carry the remote
    ``eval_jsonl`` path, whose parent directory is the set name inside a
    multi-set panel and the panel root for the panel's first set. Failing
    both, a ``regression-panel-<set>-eval`` directory name is used, and a
    flat single-set directory is resolved by a lone ``--set`` override.
    """

    set_id = info.get("eval_set_id") or metadata.get("eval_set_id")
    if set_id:
        return str(set_id)
    remote = info.get("eval_jsonl")
    if remote:
        parent = Path(str(remote)).parent.name
        if parent == "regression-panel":
            return FIRST_PANEL_SET
        if parent.count("-") >= 2 or parent in overrides:
            return parent
    for ancestor in records_path.parents:
        name = ancestor.name
        if name == "catan-qwen-series-eval-regression-panel-eval":
            return FIRST_PANEL_SET
        if name.startswith("regression-panel-") and name.endswith("-eval"):
            return name[len("regression-panel-") : -len("-eval")]
    if len(overrides) == 1:
        return next(iter(overrides))
    return None


def discover_sets(panel_dir: Path, overrides: dict[str, Path] | None = None) -> list[tuple[str, Path]]:
    """Every original-variant records.jsonl under ``panel_dir`` with its eval set id."""

    overrides = overrides or {}
    found: list[tuple[str, Path]] = []
    for records_path in sorted(panel_dir.rglob("records.jsonl")):
        summary_path = records_path.with_name("summary.json")
        info = load_json_dict(summary_path) if summary_path.is_file() else {}
        with records_path.open() as handle:
            metadata = as_dict(as_dict(loads_json(next(handle, "{}"))).get("metadata", {}))
        if (info.get("image_variant") or metadata.get("eval_variant") or "original") != "original":
            continue
        set_id = set_id_for(records_path, info, metadata, overrides)
        if not set_id:
            raise ValueError(f"cannot tell which eval set {records_path} belongs to; pass --set NAME=PATH")
        found.append((set_id, records_path))
    return found


def eval_jsonl_for(set_id: str, overrides: dict[str, Path]) -> Path:
    """Invert an eval set id (its last three path parts joined by dashes) under the replay root."""

    if set_id in overrides:
        return overrides[set_id]
    parts = set_id.rsplit("-", 2)
    replay_root = failure_scorecard.REPLAY_ROOT
    for root in (replay_root, replay_root.parent):
        candidate = root.joinpath(*parts).with_suffix(".jsonl")
        if len(parts) == 3 and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no eval jsonl found for set {set_id!r}; pass --set {set_id}=PATH")


def count_of(value: object) -> int | None:
    return len(value) if isinstance(value, (list, tuple, dict, set)) else value if isinstance(value, int) else None


def row_entanglement(adapter: Path | None, inventory: Path) -> JsonDict | None:
    if adapter is None:
        return None
    inspect_adapter = failure_scorecard.inspect_adapter
    if inspect_adapter is None:
        return {"available": False, "reason": "sft.scripts.report.inspect_token_rows is not importable"}
    tokens = [as_str(token) for token in as_list(load_json_dict(inventory)["atlas_tokens"])]
    rows = as_dict(inspect_adapter(adapter, tokens)["input"])
    twins, floor = rows.get("tokens_with_twin"), rows.get("rows_below_family_floor")
    return {
        "available": True, "adapter": str(adapter), "twins": rows.get("twins"),
        "tokens_with_twin": twins, "tokens_with_twin_count": count_of(twins),
        "rows_below_family_floor": floor, "rows_below_family_floor_count": count_of(floor),
    }
