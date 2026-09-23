"""Post-resampling token and piece-share audit."""


from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

from data_pipeline.board_recognition.reweight_node_edge._config import (
    JsonDict,
)
from data_pipeline.board_recognition.reweight_node_edge._pool import (
    answer,
    group,
    parse_readout,
)
from data_pipeline.json_coerce import as_list, as_str


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
            colours[as_str(row["color"])] += 1
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
        "unique_images": len({as_str(as_list(row["images"])[0]) for row in rows}),
        "unique_readout_images": len(
            {as_str(as_list(row["images"])[0]) for row in rows if group(row) == "readout"}
        ),
        "density_rows": dict(Counter(as_str(row["density_bin"]) for row in rows)),
        "empty_kinds": dict(
            Counter(as_str(row["negative_kind"]) for row in rows if group(row) == "empty")
        ),
    }


__all__ = ["audit"]
