"""Token-budgeted resampling of node_edge_readout_v1; no loss or label edits.

The default completion-token budget is 50% short occupied, 25% short empty,
25% complete readouts. Occupied mass is split road/settlement/city 50/25/25,
then uniformly across colours within each piece. Empty-kind and readout-density
proportions are retained within node/edge families. Evaluation files are copied
byte-for-byte. A checkpoint tokenizer (including atlas tokens) is required.

These are corpus token-exposure targets, NOT exact gradient/loss shares: batch
normalization, sequence difficulty and the training prefix also affect updates.
Reducing readout exposure means fewer readout images, not deleting empty items.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

from data_pipeline.board_recognition.node_edge_readout import (
    CATEGORY, EXPORT_SCHEMA, FAMILIES, FORWARD_QUERY, TASK_TYPE,
)
from data_pipeline.board_recognition.replay_dataset import file_sha256, read_jsonl
from data_pipeline.board_recognition.single_piece_localization import COLORS as ALL_COLORS
from data_pipeline.board_recognition.spatial_localization import (
    _deterministic_shuffle, _stable_rank, _write_json, _write_jsonl,
)
from data_pipeline.board_recognition.terrain_readout import link_or_copy


JsonDict = dict[str, Any]
SCHEMA = "catan_node_edge_token_resampling/v1"
PIECE_SHARES = {"ROAD": 0.5, "SETTLEMENT": 0.25, "CITY": 0.25}
EVAL_SPLITS = ("validation", "test", "color_diagnostic")
COMPLETION_SUFFIX = "<|im_end|>\n"


@dataclass(frozen=True)
class MixConfig:
    occupied_share: float = 0.5
    empty_share: float = 0.25
    readout_share: float = 0.25
    seed: int = 42
    max_repeats: int = 4

    def validate(self) -> None:
        shares = (self.occupied_share, self.empty_share, self.readout_share)
        if not all(math.isfinite(x) and 0 < x < 1 for x in shares):
            raise ValueError("all three token shares must be finite and strictly between 0 and 1")
        if not math.isclose(sum(shares), 1.0, abs_tol=1e-9):
            raise ValueError("token shares must sum to 1")
        if self.max_repeats < 1:
            raise ValueError("max_repeats must be positive")


def answer(row: JsonDict) -> str:
    return row["messages"][1]["content"]


def group(row: JsonDict) -> str:
    if row["task_type"] in {"node_readout", "edge_readout"}:
        return "readout"
    return "empty" if answer(row) == "empty" else "occupied"


def parse_readout(row: JsonDict) -> dict[str, str]:
    family = row["task_type"].removesuffix("_readout")
    pattern = r"<N\d{2}>" if family == "node" else r"<E\d{2}_\d{2}>"
    items: dict[str, str] = {}
    for item in answer(row).split("; "):
        token, value = item.split(" ", 1)
        if re.fullmatch(pattern, token) is None or token in items:
            raise ValueError(f"invalid or repeated token in {row['row_id']}: {token}")
        if value != "empty":
            colour, piece = value.rsplit(" ", 1)
            valid_pieces = {"road"} if family == "edge" else {"settlement", "city"}
            if colour.upper().replace(" ", "_") not in ALL_COLORS or piece not in valid_pieces:
                raise ValueError(f"invalid piece value in {row['row_id']}: {value}")
        items[token] = value
    if len(items) != {"node": 54, "edge": 72}[family] or list(items) != sorted(items):
        raise ValueError(f"incomplete or unordered readout: {row['row_id']}")
    return items


def training_pool(rows: Sequence[JsonDict]) -> list[JsonDict]:
    """Expose every already-labelled occupied spot, not just the old four/image.

    Full readouts are the label source. Existing short rows must agree with them;
    contradictory labels, missing boards, duplicate IDs and eval input fail closed.
    """
    ids: set[str] = set()
    boards: dict[tuple[str, str], tuple[JsonDict, dict[str, str]]] = {}
    for row in rows:
        if row.get("split") != "train" or row["row_id"] in ids:
            raise ValueError("resampling requires unique training rows, never evaluation rows")
        ids.add(row["row_id"])
        if row["task_type"] not in {*TASK_TYPE.values(), "node_readout", "edge_readout"}:
            raise ValueError(f"unsupported task: {row['task_type']}")
        if group(row) == "readout":
            key = (row["state_id"], row["task_type"].removesuffix("_readout"))
            if key in boards:
                raise ValueError(f"expected one readout per state/family: {key}")
            boards[key] = row, parse_readout(row)
    states = {row["state_id"] for row in rows}
    if set(boards) != {(state, family) for state in states for family in FAMILIES}:
        raise ValueError("every training state requires both complete readouts")
    short: dict[tuple[str, str], JsonDict] = {}
    for row in rows:
        if group(row) == "readout":
            continue
        parent, items = boards[(row["state_id"], row["entity_type"])]
        if items.get(row["target_token"]) != answer(row) or row["images"] != parent["images"]:
            raise ValueError(f"short/readout label or image disagreement: {row['row_id']}")
        key = row["state_id"], row["target_token"]
        if key in short:
            raise ValueError(f"duplicate short query: {key}")
        short[key] = row
    pool = [dict(row, source_row_id=row["row_id"]) for row in rows if group(row) != "occupied"]
    for (state_id, family), (parent, items) in sorted(boards.items()):
        for token, value in items.items():
            if value == "empty":
                continue
            colour, piece = value.rsplit(" ", 1)
            source = short.get((state_id, token))
            if source is not None:
                if source["color"] != colour.upper().replace(" ", "_") or source["piece"] != piece.upper():
                    raise ValueError(f"piece metadata disagrees with answer: {source['row_id']}")
                pool.append(dict(source, source_row_id=source["row_id"]))
                continue
            derived = {k: v for k, v in parent.items() if k not in {"item_count", "occupied_count"}}
            derived.update(
                row_id=f"{state_id}_{token[1:-1]}_{TASK_TYPE[family]}",
                source_row_id=parent["row_id"], derived_from_readout=True,
                task_type=TASK_TYPE[family], category=CATEGORY[family], entity_type=family,
                target_token=token, queried_token=token, polarity="positive",
                color=colour.upper().replace(" ", "_"), piece=piece.upper(),
                messages=[{"role": "user", "content": f"<image>\n{token} {FORWARD_QUERY[family]}"},
                          {"role": "assistant", "content": value}],
            )
            pool.append(derived)
    return pool


def bucket(row: JsonDict) -> str:
    kind = group(row)
    if kind == "occupied":
        return f"occupied/{row['piece']}/{row['color']}"
    if kind == "empty":
        return f"empty/{row['entity_type']}/{row['negative_kind']}"
    return f"readout/{row['task_type'].removesuffix('_readout')}/{row['density_bin']}"


def apportion(masses: dict[str, float], total: int) -> dict[str, int]:
    """Largest remainder allocation; stable ties and an exact row budget."""
    scale = total / sum(masses.values())
    exact = {key: value * scale for key, value in masses.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda k: (-(exact[k] - counts[k]), k))[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def resample(
    pool: Sequence[JsonDict], count_tokens: Callable[[str], int], *, row_count: int,
    config: MixConfig = MixConfig(),
) -> tuple[list[JsonDict], JsonDict]:
    config.validate()
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    buckets: dict[str, list[JsonDict]] = defaultdict(list)
    for row in pool:
        if row.get("split") != "train":
            raise ValueError("evaluation rows cannot enter the training pool")
        buckets[bucket(row)].append(row)
    required = {f"occupied/{piece}/{colour}" for piece in PIECE_SHARES for colour in ALL_COLORS}
    if missing := required - set(buckets):
        raise ValueError(f"missing colour x piece training coverage: {sorted(missing)}")
    masses: dict[str, float] = {}
    token_targets: dict[str, float] = {}
    for key, candidates in sorted(buckets.items()):
        kind, family, _ = key.split("/")
        lengths = [count_tokens(answer(row)) for row in candidates]
        if any(length <= 0 for length in lengths):
            raise ValueError("token counts must be positive")
        mean_length = sum(lengths) / len(lengths)
        if kind == "occupied":
            target = config.occupied_share * PIECE_SHARES[family] / len(ALL_COLORS)
        else:
            # Preserve source row proportions across kinds/densities within family.
            family_tokens = sum(count_tokens(answer(r)) for k, v in buckets.items()
                                if k.startswith(f"{kind}/{family}/") for r in v)
            target = getattr(config, f"{kind}_share") * 0.5 * sum(lengths) / family_tokens
        token_targets[key] = target
        masses[key] = target / mean_length
    for kind in ("empty", "readout"):
        for family in FAMILIES:
            if not any(k.startswith(f"{kind}/{family}/") for k in buckets):
                raise ValueError(f"missing {kind}/{family} coverage")
    quotas = apportion(masses, row_count)
    selected: list[JsonDict] = []
    repeats: Counter[str] = Counter()
    for key, candidates in sorted(buckets.items()):
        if quotas[key] == 0:
            raise ValueError(f"row budget drops stratum {key}; increase row_count")
        if quotas[key] > config.max_repeats * len(candidates):
            raise ValueError(f"{key} exceeds max_repeats={config.max_repeats}; add examples or change the mix")
        remaining = quotas[key]
        cycle = 0
        while remaining:
            ordered = sorted(candidates, key=lambda row: (_stable_rank(config.seed, key, cycle, row['row_id']), row['row_id']))
            for row in ordered[:remaining]:
                repeat = repeats[row["row_id"]]
                selected.append(dict(row, source_query_id=row["row_id"],
                                     row_id=f"{row['row_id']}__mix{repeat}", sampling_repeat=repeat))
                repeats[row["row_id"]] += 1
            remaining -= min(remaining, len(ordered))
            cycle += 1
    selected = _deterministic_shuffle(selected, f"node_edge_token_mix:{config.seed}")
    return selected, {
        "config": asdict(config), "piece_token_shares_within_occupied": PIECE_SHARES,
        "bucket_target_token_shares": token_targets, "bucket_row_quotas": quotas,
        "pool_rows": len(pool), "unique_selected_queries": len(repeats),
        "max_query_repeats": max(repeats.values()),
    }


def audit(rows: Sequence[JsonDict], count_tokens: Callable[[str], int]) -> JsonDict:
    by_group: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    colours: Counter[str] = Counter()
    colour_piece: Counter[str] = Counter()
    colour_piece_tokens: Counter[str] = Counter()
    semantic_items: Counter[str] = Counter()
    readout_items: Counter[str] = Counter()
    for row in rows:
        kind = group(row)
        by_group[kind] += 1
        tokens[kind] += count_tokens(answer(row))
        if kind == "occupied":
            colours[row["color"]] += 1
            colour_piece[f"{row['color']}/{row['piece']}"] += 1
            colour_piece_tokens[f"{row['color']}/{row['piece']}"] += count_tokens(answer(row))
        values = list(parse_readout(row).values()) if kind == "readout" else [answer(row)]
        for value in values:
            polarity = "empty" if value == "empty" else "occupied"
            semantic_items[polarity] += 1
            if kind == "readout":
                readout_items[polarity] += 1
    return {
        "rows": len(rows), "rows_by_group": dict(by_group), "completion_tokens_by_group": dict(tokens),
        "completion_token_shares": {key: value / sum(tokens.values()) for key, value in tokens.items()},
        "short_positive_colours": dict(sorted(colours.items())),
        "short_positive_colour_piece_rows": dict(sorted(colour_piece.items())),
        "short_positive_colour_piece_tokens": dict(sorted(colour_piece_tokens.items())),
        "semantic_items": dict(semantic_items), "readout_items": dict(readout_items),
        "empty_item_fraction": semantic_items["empty"] / sum(semantic_items.values()),
        "unique_images": len({row["images"][0] for row in rows}),
        "unique_readout_images": len({row["images"][0] for row in rows if group(row) == "readout"}),
        "density_rows": dict(Counter(row["density_bin"] for row in rows)),
        "empty_kinds": dict(Counter(row["negative_kind"] for row in rows if group(row) == "empty")),
    }


def export_reweighted(
    source: Path, output: Path, tokenizer_path: Path, *, config: MixConfig = MixConfig(),
    row_count: int | None = None,
) -> JsonDict:
    from tokenizers import Tokenizer

    config.validate()
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from the immutable source export")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["schema"] != EXPORT_SCHEMA:
        raise ValueError("expected a node_edge_readout_v1 export")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.token_to_id("<|im_end|>") is None:
        raise ValueError("tokenizer is missing the Qwen completion delimiter")
    rows = read_jsonl(source / "stage1/train.jsonl")
    pool = training_pool(rows)
    atlas = {token for row in rows if group(row) == "readout" for token in parse_readout(row)}
    if any(tokenizer.token_to_id(token) is None or len(tokenizer.encode(token, add_special_tokens=False).ids) != 1 for token in atlas):
        raise ValueError("use the checkpoint tokenizer with atomic atlas tokens, not the base tokenizer")

    @lru_cache(maxsize=None)
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text + COMPLETION_SUFFIX, add_special_tokens=False).ids)

    selected, recipe = resample(pool, count_tokens, row_count=len(rows) if row_count is None else row_count, config=config)
    files = {}
    eval_rows = {}
    train_layouts = {row["layout_id"] for row in rows}
    for split in ("train", *EVAL_SPLITS):
        path = source / "stage1" / f"{split}.jsonl"
        digest = file_sha256(path)
        if digest != metadata["files"][f"stage1/{split}.jsonl"]["sha256"]:
            raise ValueError(f"source fingerprint changed: {path}")
        files[split] = {"source_sha256": digest}
        if split != "train":
            eval_rows[split] = read_jsonl(path)
            if {row["layout_id"] for row in eval_rows[split]} & train_layouts:
                raise ValueError(f"train/eval layout overlap: {split}")
    image_names = {row["images"][0] for row in [*selected, *(r for rows_ in eval_rows.values() for r in rows_)]}
    for name in image_names:
        if Path(name).name != name or not (source / "images" / name).is_file():
            raise ValueError(f"invalid or missing image: {name}")
    result = {
        "schema": SCHEMA, "source_export": str(source), "source_metadata_sha256": file_sha256(source / "metadata.json"),
        "image_root": str(output / "images"), "tokenizer_sha256": file_sha256(tokenizer_path),
        "token_count_basis": "tokenizer.encode(answer + completion_suffix, add_special_tokens=False)",
        "completion_suffix": COMPLETION_SUFFIX,
        "caveat": "Corpus token exposure, not exact gradient shares; no within-readout weighting. Common EOT/newline tokens included; no prompt tokens.",
        "recipe": recipe, "before": audit(rows, count_tokens), "after": audit(selected, count_tokens),
        "prefix_16384": audit(selected[:16384], count_tokens), "files": files,
    }
    output.mkdir(parents=True)
    (output / "images").mkdir()
    # Match the existing exporter's immutable-image hard-link/copy convention.
    # Symlinks outside image_root would violate the trainer's path guard.
    for name in sorted(image_names):
        link_or_copy(source / "images" / name, output / "images" / name)
    _write_jsonl(output / "stage1/train.jsonl", selected)
    for split in EVAL_SPLITS:
        shutil.copyfile(source / "stage1" / f"{split}.jsonl", output / "stage1" / f"{split}.jsonl")
    for split in files:
        files[split]["sha256"] = file_sha256(output / "stage1" / f"{split}.jsonl")
        if split != "train" and files[split]["sha256"] != files[split]["source_sha256"]:
            raise RuntimeError(f"evaluation changed while copying: {split}")
    _write_json(output / "metadata.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--occupied-share", type=float, default=0.5)
    parser.add_argument("--empty-share", type=float, default=0.25)
    parser.add_argument("--readout-share", type=float, default=0.25)
    parser.add_argument("--row-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-repeats", type=int, default=4)
    args = parser.parse_args(argv)
    result = export_reweighted(args.source, args.output_dir, args.tokenizer_json,
                               config=MixConfig(args.occupied_share, args.empty_share, args.readout_share, args.seed, args.max_repeats),
                               row_count=args.row_count)
    print(json.dumps({"before": result["before"], "after": result["after"], "output": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
