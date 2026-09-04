"""Report the geometry of the trainable atlas-token rows in a PEFT adapter.

Training gates use this to catch entangled rows (near-duplicate directions, "twins") and
rows that point away from their own family (node, edge, tile, port) or away from the shared
mean direction. Both the input embedding rows and the lm_head rows are inspected. Every
cosine is computed over L2-normalized float32 rows; a family centroid is the normalized mean
of that family's raw rows, and the mean direction is the normalized mean of all rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


JsonDict = dict[str, Any]

INPUT_ROWS_KEY = (
    "base_model.model.model.language_model.embed_tokens.token_adapter.trainable_tokens_delta"
)
OUTPUT_ROWS_KEY = "base_model.model.lm_head.token_adapter.trainable_tokens_delta"
SIDE_KEYS = {"input": INPUT_ROWS_KEY, "output": OUTPUT_ROWS_KEY}
DEFAULT_TOKEN_INVENTORY = Path(
    "artifacts/generated/board_recognition/replay_v1",
    "ms_swift_bidirectional_v1/trainable_tokens.json",
)
FAMILIES = ("N", "E", "T", "P")
FAMILY_PAIRS = ("NE", "NT", "NP", "ET", "EP", "TP")
FOCUS_TOKEN = "<T10>"
DEFAULT_TWIN_THRESHOLD = 0.25
DEFAULT_FAMILY_FLOOR = 0.05
TABLE_FLOOR_PREVIEW = 8


def token_family(token: str) -> str:
    """Return the family letter of an atlas token, its second character."""
    family = token[1:2]
    if not token.startswith("<") or family not in FAMILIES:
        raise ValueError(
            f"cannot derive a family from token {token!r}; expected <N..>, <E..>, <T..> or <P..>"
        )
    return family


def load_token_inventory(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).expanduser().read_text())
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"{path} must contain a 'tokens' list of strings")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{path} lists duplicate tokens")
    for token in tokens:
        token_family(token)
    return list(tokens)


def load_token_rows(path: str | Path) -> dict[str, torch.Tensor]:
    """Read only the input and output trainable-token delta tensors from an adapter file."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        present = list(handle.keys())
        for side, key in SIDE_KEYS.items():
            if key not in present:
                token_keys = sorted(name for name in present if "token" in name)
                raise ValueError(
                    f"{path} lacks {key}; keys containing 'token': {token_keys or 'none'}"
                )
            rows[side] = handle.get_tensor(key)
    return rows


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vectors, dim=-1)


def _value(number: torch.Tensor | float) -> float:
    return round(float(number), 6)


def inspect_rows(
    rows: torch.Tensor,
    tokens: list[str],
    *,
    twin_threshold: float = DEFAULT_TWIN_THRESHOLD,
    family_floor: float = DEFAULT_FAMILY_FLOOR,
) -> JsonDict:
    """Describe one [tokens, hidden] tensor whose row i belongs to tokens[i]."""
    if rows.ndim != 2:
        raise ValueError(f"expected a 2-D [tokens, hidden] tensor, got shape {tuple(rows.shape)}")
    if rows.shape[0] != len(tokens):
        raise ValueError(f"tensor has {rows.shape[0]} rows but the inventory has {len(tokens)}")
    if len(tokens) < 2:
        raise ValueError("need at least two rows to compare")
    families = [token_family(token) for token in tokens]
    raw = rows.detach().to(device="cpu", dtype=torch.float32)
    unit = _unit(raw)
    cosine = unit @ unit.T
    count = len(tokens)

    upper = torch.triu_indices(count, count, offset=1)
    pair_values = cosine[upper[0], upper[1]]
    best = int(pair_values.argmax())
    twin_mask = pair_values > twin_threshold
    twins = [
        [tokens[int(left)], tokens[int(right)], _value(value)]
        for left, right, value in zip(
            upper[0][twin_mask], upper[1][twin_mask], pair_values[twin_mask]
        )
    ]
    twins.sort(key=lambda pair: -pair[2])
    twinned = {token for pair in twins for token in pair[:2]}

    centroids: dict[str, torch.Tensor] = {}
    for family in FAMILIES:
        member = torch.tensor([row_family == family for row_family in families])
        if bool(member.any()):
            centroids[family] = _unit(raw[member].mean(dim=0))
    family_centroid_cosines = {
        pair: _value(centroids[pair[0]] @ centroids[pair[1]])
        for pair in FAMILY_PAIRS
        if pair[0] in centroids and pair[1] in centroids
    }

    own_family_loading = {
        token: _value(unit[index] @ centroids[family])
        for index, (token, family) in enumerate(zip(tokens, families))
    }
    own_family_median = {
        family: _value(
            statistics.median(
                own_family_loading[token]
                for token, row_family in zip(tokens, families)
                if row_family == family
            )
        )
        for family in centroids
    }
    rows_below_family_floor = sorted(
        (
            [token, loading]
            for token, loading in own_family_loading.items()
            if loading < family_floor
        ),
        key=lambda item: item[1],
    )

    mean_direction = _unit(raw.mean(dim=0))
    mean_direction_loading = {
        token: _value(unit[index] @ mean_direction) for index, token in enumerate(tokens)
    }
    rows_negative_on_mean = [
        token for token, loading in mean_direction_loading.items() if loading < 0
    ]

    zero_rows = torch.nonzero(raw.norm(dim=-1) == 0).flatten().tolist()
    return {
        "row_count": count,
        "hidden_size": int(raw.shape[1]),
        "zero_norm_tokens": [tokens[index] for index in zero_rows],
        "pairwise_cosine_mean": _value(pair_values.mean()),
        "pairwise_cosine_sd": _value(pair_values.std(correction=0)),
        "max_pair": _value(pair_values[best]),
        "max_pair_tokens": [tokens[int(upper[0][best])], tokens[int(upper[1][best])]],
        "tokens_with_twin": len(twinned),
        "twins": twins,
        "family_centroid_cosines": family_centroid_cosines,
        "own_family_loading": own_family_loading,
        "own_family_median": own_family_median,
        "rows_below_family_floor": rows_below_family_floor,
        "mean_direction_loading": mean_direction_loading,
        "rows_negative_on_mean": rows_negative_on_mean,
    }


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
    entries: dict[str, JsonDict] = {}
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
    width = max(len("adapter"), *(len(label) for label in report["adapters"]))
    header = (
        f"{'adapter':<{width}}  {'side':<6}  {'pair mean/sd':<13}  twins  "
        f"{'own-family median N/E/T/P':<27}  {FOCUS_TOKEN:>6}  "
        f"below family floor {report['family_floor']:g}"
    )
    lines = [header]
    for label, entry in report["adapters"].items():
        for side in SIDE_KEYS:
            result = entry[side]
            medians = result["own_family_median"]
            median_text = "/".join(
                f"{medians[family]:+.2f}" if family in medians else "-" for family in FAMILIES
            )
            focus = result["own_family_loading"].get(FOCUS_TOKEN)
            focus_text = f"{focus:+.3f}" if focus is not None else "n/a"
            below = result["rows_below_family_floor"]
            below_text = ", ".join(
                f"{token} {loading:+.2f}" for token, loading in below[:TABLE_FLOOR_PREVIEW]
            )
            if len(below) > TABLE_FLOOR_PREVIEW:
                below_text += ", ..."
            pair_text = f"{result['pairwise_cosine_mean']:.3f}/{result['pairwise_cosine_sd']:.3f}"
            lines.append(
                f"{label:<{width}}  {side:<6}  {pair_text:<13}  {result['tokens_with_twin']:>5}  "
                f"{median_text:<27}  {focus_text:>6}  {len(below)}: {below_text}"
            )
    return "\n".join(lines)


def parse_adapter_spec(spec: str) -> tuple[str, Path]:
    label, separator, raw_path = spec.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {spec!r}")
    return label.strip(), Path(raw_path.strip()).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--adapter",
        action="append",
        type=parse_adapter_spec,
        required=True,
        metavar="LABEL=PATH",
        help="adapter_model.safetensors to inspect; repeat for several adapters",
    )
    parser.add_argument(
        "--token-inventory",
        type=Path,
        default=DEFAULT_TOKEN_INVENTORY,
        help="trainable_tokens.json whose 'tokens' list orders the adapter rows",
    )
    parser.add_argument(
        "--twin-threshold",
        type=float,
        default=DEFAULT_TWIN_THRESHOLD,
        help="pairwise cosine above which two rows count as twins",
    )
    parser.add_argument(
        "--family-floor",
        type=float,
        default=DEFAULT_FAMILY_FLOOR,
        help="own-family loading below which a row is reported",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the JSON report here instead of printing it to stdout",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="skip the human-readable table on stderr"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        args.adapter,
        args.token_inventory,
        twin_threshold=args.twin_threshold,
        family_floor=args.family_floor,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    if not args.quiet:
        print(format_table(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
