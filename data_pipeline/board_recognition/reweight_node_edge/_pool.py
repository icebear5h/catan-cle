"""Row readout parsing, grouping and the eligible training pool."""


from __future__ import annotations

import re
from typing import Sequence

from data_pipeline.board_recognition.node_edge_readout import (
    CATEGORY,
    FAMILIES,
    TASK_TYPE,
)
from data_pipeline.board_recognition.reweight_node_edge._config import (
    JsonDict,
)
from data_pipeline.board_recognition.single_piece_localization import (
    COLORS as ALL_COLORS,
)
from data_pipeline.board_recognition.single_piece_localization import FORWARD_QUERY
from data_pipeline.json_coerce import as_dict, as_list, as_str


def answer(row: JsonDict) -> str:
    return as_str(as_dict(as_list(row["messages"])[1])["content"])


def group(row: JsonDict) -> str:
    if row["task_type"] in {"node_readout", "edge_readout"}:
        return "readout"
    return "empty" if answer(row) == "empty" else "occupied"


def parse_readout(row: JsonDict) -> dict[str, str]:
    family = as_str(row["task_type"]).removesuffix("_readout")
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
        ids.add(as_str(row["row_id"]))
        if row["task_type"] not in {*TASK_TYPE.values(), "node_readout", "edge_readout"}:
            raise ValueError(f"unsupported task: {row['task_type']}")
        if group(row) == "readout":
            key = (as_str(row["state_id"]), as_str(row["task_type"]).removesuffix("_readout"))
            if key in boards:
                raise ValueError(f"expected one readout per state/family: {key}")
            boards[key] = row, parse_readout(row)
    states = {as_str(row["state_id"]) for row in rows}
    if set(boards) != {(state, family) for state in states for family in FAMILIES}:
        raise ValueError("every training state requires both complete readouts")
    short: dict[tuple[str, str], JsonDict] = {}
    for row in rows:
        if group(row) == "readout":
            continue
        parent, items = boards[(as_str(row["state_id"]), as_str(row["entity_type"]))]
        if items.get(as_str(row["target_token"])) != answer(row) or row["images"] != parent["images"]:
            raise ValueError(f"short/readout label or image disagreement: {row['row_id']}")
        key = as_str(row["state_id"]), as_str(row["target_token"])
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
    family_name = as_str(row["task_type"]).removesuffix("_readout")
    return f"readout/{family_name}/{row['density_bin']}"


__all__ = ["answer", "bucket", "group", "parse_readout", "training_pool"]
