"""Load atlas-token rows and measure their cosine geometry."""

from __future__ import annotations

import statistics
from pathlib import Path

import torch

from sft.json_types import JsonList, json_dict, json_list, load_json
from sft.safetensor_types import open_tensors

from ._base import (
    DEFAULT_FAMILY_FLOOR,
    DEFAULT_TWIN_THRESHOLD,
    FAMILIES,
    FAMILY_PAIRS,
    SIDE_KEYS,
    JsonDict,
)


def token_family(token: str) -> str:
    """Return the family letter of an atlas token, its second character."""
    family = token[1:2]
    if not token.startswith("<") or family not in FAMILIES:
        raise ValueError(
            f"cannot derive a family from token {token!r}; expected <N..>, <E..>, <T..> or <P..>"
        )
    return family


def load_token_inventory(path: str | Path) -> list[str]:
    payload = load_json(Path(path).expanduser())
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"{path} must contain a 'tokens' list of strings")
    names = [token for token in tokens if isinstance(token, str)]
    if len(set(names)) != len(names):
        raise ValueError(f"{path} lists duplicate tokens")
    for token in names:
        token_family(token)
    return names


def load_token_rows(path: str | Path) -> dict[str, torch.Tensor]:
    """Read only the input and output trainable-token delta tensors from an adapter file."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, torch.Tensor] = {}
    with open_tensors(str(path), framework="pt", device="cpu") as handle:
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
    twin_pairs = [
        (tokens[int(left)], tokens[int(right)], _value(value))
        for left, right, value in zip(
            upper[0][twin_mask], upper[1][twin_mask], pair_values[twin_mask]
        )
    ]
    twin_pairs.sort(key=lambda pair: -pair[2])
    twins: JsonList = [[left, right, value] for left, right, value in twin_pairs]
    twinned = {token for pair in twin_pairs for token in pair[:2]}

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
    below_floor = sorted(
        (
            (token, loading)
            for token, loading in own_family_loading.items()
            if loading < family_floor
        ),
        key=lambda item: item[1],
    )
    rows_below_family_floor: JsonList = [[token, loading] for token, loading in below_floor]

    mean_direction = _unit(raw.mean(dim=0))
    mean_direction_loading = {
        token: _value(unit[index] @ mean_direction) for index, token in enumerate(tokens)
    }
    rows_negative_on_mean = [
        token for token, loading in mean_direction_loading.items() if loading < 0
    ]

    zero_rows: list[int] = torch.nonzero(raw.norm(dim=-1) == 0).flatten().tolist()
    return {
        "row_count": count,
        "hidden_size": int(raw.shape[1]),
        "zero_norm_tokens": json_list(tokens[index] for index in zero_rows),
        "pairwise_cosine_mean": _value(pair_values.mean()),
        "pairwise_cosine_sd": _value(pair_values.std(correction=0)),
        "max_pair": _value(pair_values[best]),
        "max_pair_tokens": json_list([tokens[int(upper[0][best])], tokens[int(upper[1][best])]]),
        "tokens_with_twin": len(twinned),
        "twins": twins,
        "family_centroid_cosines": json_dict(family_centroid_cosines),
        "own_family_loading": json_dict(own_family_loading),
        "own_family_median": json_dict(own_family_median),
        "rows_below_family_floor": rows_below_family_floor,
        "mean_direction_loading": json_dict(mean_direction_loading),
        "rows_negative_on_mean": json_list(rows_negative_on_mean),
    }
