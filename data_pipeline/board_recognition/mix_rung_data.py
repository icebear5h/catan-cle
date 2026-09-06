"""Mix existing board-recognition exports into one rung by quota (``mix_rung_data``).

The ladder's rungs forget each other when trained one family at a time: the
terrain rung dropped the markers, the piece rung dropped terrain. A mixed
rung draws every head from the exports already on disk, with quotas that
say how many rows of each group come from each board density, so no head
starves and no density dominates. The recipe is a JSON file, not code:

    {
      "sources": {"pieces": "<export dir>", "terrain": "<export dir>"},
      "groups": [
        {"name": "piece_occupied", "source": "pieces", "categories": ["node.occupancy", "edge.owner"],
         "polarity": "positive", "rows": 6000, "density": {"setup": 1, "sparse": 1, "dense": 1},
         "balance": ["piece", "color"], "piece_shares": {"ROAD": 0.5, "SETTLEMENT": 0.25, "CITY": 0.25}},
        {"name": "piece_empty", ..., "polarity": "hard_negative", "kind_shares": {"adjacent": 0.5, ...}},
        {"name": "terrain_short", "source": "terrain", "categories": ["tile.resource", "tile.number", "port.port_type"], ...},
        {"name": "node_readout", "source": "pieces", "categories": ["node.readout"], "rows": 100, ...}
      ],
      "eval_sample": [{"source": "pieces", "split": "validation", "every": 8}, {"source": "terrain", "split": "validation", "every": 6}]
    }

Within a group, rows are apportioned to density bins by the ``density``
weights, then inside each bin spread evenly over the ``balance`` keys (each
colour, each piece type) as far as the pool allows, each cell in a stable
hashed order. A cell that runs short is topped up from the rest of its bin,
then from the group's other bins; a group with ``max_repeats`` above one
oversamples a short cell's own rows first (ids gain ``#r<n>``). Every
shortfall is reported in ``metadata.json``. Rows keep their source ids and metadata, so the panel and
the scorecard read the mixed run like any other. Images are hard-linked
into one root. Completion-token shares are estimated per group (atlas
tokens count one, other words and separators one each) so a recipe can be
sanity-checked without a tokenizer.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.board_recognition.replay_dataset import file_sha256, read_jsonl
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _deterministic_shuffle,
    _stable_rank,
    _write_json,
    _write_jsonl,
)
from data_pipeline.board_recognition.terrain_readout import link_or_copy


JsonDict = dict[str, Any]
EXPORT_SCHEMA = "catan_mixed_rung/v1"
DENSITY_BINS = ("empty", "setup", "sparse", "dense")
ATLAS_TOKEN_RE = re.compile(r"<[NETP][0-9_]+>")
WORD_RE = re.compile(r"[A-Za-z0-9:]+|;")


def estimate_tokens(answer: str) -> int:
    """Rough completion tokens: one per atlas token, one per word or separator, one end token."""

    without_tokens = ATLAS_TOKEN_RE.sub(" ", answer)
    return len(ATLAS_TOKEN_RE.findall(answer)) + len(WORD_RE.findall(without_tokens)) + 1


def answer_of(row: JsonDict) -> str:
    return row["messages"][1]["content"]


def cell_key(row: JsonDict, balance: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(key)) for key in balance)


def apportion(weights: dict[str, float], total: int) -> dict[str, int]:
    """Largest-remainder split of ``total`` over ``weights``."""

    mass = sum(weights.values())
    if mass <= 0:
        raise SpatialLocalizationError("apportion needs positive weights")
    raw = {key: total * value / mass for key, value in weights.items()}
    counts = {key: int(value) for key, value in raw.items()}
    for key, _ in sorted(((key, raw[key] - counts[key]) for key in raw), key=lambda item: (-item[1], item[0]))[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def matches(row: JsonDict, group: JsonDict) -> bool:
    if row["category"] not in group["categories"]:
        return False
    if group.get("polarity") and row.get("polarity") != group["polarity"]:
        return False
    return True


def ranked(rows: Sequence[JsonDict], salt: str) -> list[JsonDict]:
    return sorted(rows, key=lambda row: (_stable_rank(salt, row["row_id"]), row["row_id"]))


def repeated(row: JsonDict, repeat: int) -> JsonDict:
    """A copy of ``row`` marked as its ``repeat``-th oversample; the first copy is the row itself."""

    if repeat == 0:
        return row
    return {**row, "row_id": f"{row['row_id']}#r{repeat}", "mix_repeat": repeat}


def draw_cells(pool: Sequence[JsonDict], quotas: dict[tuple[str, ...], int], balance: Sequence[str], salt: str, max_repeats: int = 1) -> tuple[list[JsonDict], list[JsonDict]]:
    """Take up to ``quotas[cell]`` rows per balance cell, oversampling a short cell up to ``max_repeats`` times.

    Returns the picks and the untouched remainder in rank order.
    """

    by_cell: dict[tuple[str, ...], list[JsonDict]] = defaultdict(list)
    for row in ranked(pool, salt):
        by_cell[cell_key(row, balance)].append(row)
    picked: list[JsonDict] = []
    remainder: list[JsonDict] = []
    for cell, rows in by_cell.items():
        want = quotas.get(cell, 0)
        picked.extend(rows[:want])
        remainder.extend(rows[want:])
        for repeat in range(1, max_repeats):
            missing = want - len(rows) * repeat
            if missing <= 0 or not rows:
                break
            picked.extend(repeated(row, repeat) for row in rows[:missing])
    return picked, ranked(remainder, salt + ":fill")


def sample_group(group: JsonDict, rows: Sequence[JsonDict]) -> tuple[list[JsonDict], JsonDict]:
    """Draw one group by density weights, balance keys and optional per-key shares."""

    name = group["name"]
    pool = [row for row in rows if matches(row, group)]
    if not pool:
        raise SpatialLocalizationError(f"group {name} matches no rows")
    balance = list(group.get("balance", []))
    shares: dict[str, dict[str, float]] = {}
    for key in balance:
        given = group.get(f"{key}_shares")
        if given:
            shares[key] = {str(k): float(v) for k, v in given.items()}
    by_density: dict[str, list[JsonDict]] = defaultdict(list)
    for row in pool:
        by_density[str(row.get("density_bin"))].append(row)
    requested = group.get("density") or {bin_name: 1 for bin_name in by_density}  # no "density": every bin in the pool, equally
    density_weights = {bin_name: float(weight) for bin_name, weight in requested.items() if bin_name in by_density}
    if not density_weights:
        raise SpatialLocalizationError(f"group {name}: no requested density bin has rows")
    per_bin = apportion(density_weights, int(group["rows"]))
    picked_all: list[JsonDict] = []
    leftovers: list[JsonDict] = []
    shortfall = 0
    for bin_name, want in per_bin.items():
        bin_pool = by_density[bin_name]
        if balance:
            cells = sorted({cell_key(row, balance) for row in bin_pool})
            weights = {}
            for cell in cells:
                weight = 1.0
                for index, key in enumerate(balance):
                    if key in shares:
                        weight *= shares[key].get(cell[index], 0.0)
                weights[cell] = weight
            if sum(weights.values()) <= 0:
                weights = {cell: 1.0 for cell in cells}
            quotas = apportion(weights, want)
            picked, remainder = draw_cells(bin_pool, quotas, balance, f"{name}:{bin_name}", int(group.get("max_repeats", 1)))
        else:
            ordered = ranked(bin_pool, f"{name}:{bin_name}")
            picked, remainder = ordered[:want], ordered[want:]
        if len(picked) < want:
            top_up = remainder[: want - len(picked)]
            picked.extend(top_up)
            remainder = remainder[len(top_up):]
        shortfall += max(0, want - len(picked))
        picked_all.extend(picked)
        leftovers.extend(remainder)
    if shortfall and leftovers:
        extra = ranked(leftovers, f"{name}:cross-bin")[:shortfall]
        picked_all.extend(extra)
        shortfall -= len(extra)
    report = {
        "requested": int(group["rows"]),
        "taken": len(picked_all),
        "shortfall": shortfall,
        "pool": len(pool),
        "by_density": dict(sorted(Counter(str(row.get("density_bin")) for row in picked_all).items())),
        "by_category": dict(sorted(Counter(row["category"] for row in picked_all).items())),
        "estimated_tokens": sum(estimate_tokens(answer_of(row)) for row in picked_all),
        "repeated_rows": sum(1 for row in picked_all if row.get("mix_repeat")),
    }
    for key in balance:
        report[f"by_{key}"] = dict(sorted(Counter(str(row.get(key)) for row in picked_all).items()))
    return picked_all, report


def load_recipe(path: Path) -> JsonDict:
    recipe = json.loads(path.read_text())
    for key in ("sources", "groups"):
        if key not in recipe:
            raise SpatialLocalizationError(f"recipe lacks {key!r}")
    names = [group["name"] for group in recipe["groups"]]
    if len(names) != len(set(names)):
        raise SpatialLocalizationError("group names must be unique")
    return recipe


def source_rows(source_dir: Path, split: str) -> list[JsonDict]:
    path = source_dir / "stage1" / f"{split}.jsonl"
    if not path.is_file():
        raise SpatialLocalizationError(f"missing split file: {path}")
    return read_jsonl(path)


def source_image_root(source_dir: Path) -> Path:
    """The export's own image root from its metadata, else ``<source>/images``."""

    metadata = source_dir / "metadata.json"
    if metadata.is_file():
        root = json.loads(metadata.read_text()).get("image_root")
        if root and Path(root).is_dir():
            return Path(root)
    return source_dir / "images"


def eval_sample(recipe: JsonDict, sources: dict[str, Path]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for spec in recipe.get("eval_sample", []):
        every = int(spec.get("every", 1))
        rows.extend(row for index, row in enumerate(source_rows(sources[spec["source"]], spec.get("split", "validation"))) if index % every == 0)
    return rows


def export_mixed_rung(recipe_path: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> JsonDict:
    recipe = load_recipe(Path(recipe_path))
    output = Path(output_dir).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True)
    sources = {name: Path(path).resolve() for name, path in recipe["sources"].items()}
    image_roots = {name: source_image_root(path) for name, path in sources.items()}
    train_by_source = {name: source_rows(path, recipe.get("train_split", "train")) for name, path in sources.items()}

    picked: list[JsonDict] = []
    reports: dict[str, JsonDict] = {}
    for group in recipe["groups"]:
        rows, report = sample_group(group, train_by_source[group["source"]])
        for row in rows:
            row = dict(row)
            row["mix_group"] = group["name"]
            row["mix_source"] = group["source"]
            picked.append(row)
        reports[group["name"]] = report
    seen = Counter(row["row_id"] for row in picked)
    duplicates = [row_id for row_id, count in seen.items() if count > 1]
    if duplicates:
        raise SpatialLocalizationError(f"row ids picked by more than one group: {duplicates[:5]}")
    train = _deterministic_shuffle(picked, "mixed_rung")
    for row in train:
        link_or_copy(image_roots[row["mix_source"]] / row["images"][0], images_dir / row["images"][0])
    train_path = output / "stage1" / "train.jsonl"
    _write_jsonl(train_path, train)

    evaluation = eval_sample(recipe, sources)
    for spec in recipe.get("eval_sample", []):
        for row in evaluation:
            candidate = image_roots[spec["source"]] / row["images"][0]
            if candidate.is_file():
                link_or_copy(candidate, images_dir / row["images"][0])
    eval_path = output / "stage1" / "eval_sample.jsonl"
    _write_jsonl(eval_path, evaluation)

    total_tokens = sum(report["estimated_tokens"] for report in reports.values()) or 1
    metadata = {
        "schema": EXPORT_SCHEMA,
        "recipe": str(Path(recipe_path).resolve()),
        "recipe_sha256": file_sha256(Path(recipe_path)),
        "sources": {name: str(path) for name, path in sources.items()},
        "source_image_roots": {name: str(path) for name, path in image_roots.items()},
        "image_root": str(images_dir),
        "rows": len(train),
        "unique_images": len({row["images"][0] for row in train}),
        "groups": {name: {**report, "estimated_token_share": round(report["estimated_tokens"] / total_tokens, 4)} for name, report in reports.items()},
        "by_density": dict(sorted(Counter(str(row.get("density_bin")) for row in train).items())),
        "by_category": dict(sorted(Counter(row["category"] for row in train).items())),
        "eval_sample_rows": len(evaluation),
        "eval_sample_by_category": dict(sorted(Counter(row["category"] for row in evaluation).items())),
        "files": {"stage1/train.jsonl": file_sha256(train_path), "stage1/eval_sample.jsonl": file_sha256(eval_path)},
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    metadata = export_mixed_rung(args.recipe, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({key: value for key, value in metadata.items() if key != "files"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
