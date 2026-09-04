"""Build an enriched held-out eval suite from existing Catan recognition splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def read_jsonl(path: Path) -> list[JsonDict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def message_text(row: JsonDict, index: int) -> str:
    return str(row["messages"][index]["content"]).strip()


def assert_aligned(row: JsonDict, audit: JsonDict) -> None:
    prompt = message_text(row, 0)
    expected_prompt = str(audit.get("semantic_prompt") or audit.get("prompt") or "")
    if prompt.replace("<image>\n", "", 1) != expected_prompt.replace("<image>\n", "", 1):
        raise ValueError(f"prompt mismatch for {audit['query_id']}")
    answer = message_text(row, 1)
    expected_answer = str(audit.get("semantic_answer") or audit.get("answer") or "")
    if answer != expected_answer:
        raise ValueError(f"answer mismatch for {audit['query_id']}")


def compact_metadata(audit: JsonDict, *, suite: str, category: str) -> JsonDict:
    keys = (
        "query_id",
        "state_id",
        "density_bin",
        "entity_type",
        "head",
        "attribute",
        "curriculum_stage",
        "task_family",
        "task_type",
        "relationship",
        "polarity",
    )
    metadata = {
        "suite": suite,
        "category": category,
        **{key: audit[key] for key in keys if audit.get(key) is not None},
    }
    return metadata


def build_rows(root: Path, split: str) -> list[JsonDict]:
    mixed = read_jsonl(root / "ms_swift_bidirectional_v1" / "mixed" / f"{split}.jsonl")
    mixed_index = read_jsonl(
        root / "ms_swift_bidirectional_v1" / "mixed_index" / f"{split}.jsonl"
    )
    forward = {
        row["query_id"]: row
        for row in read_jsonl(root / "ms_swift_semantic_v1" / "audit" / f"{split}.jsonl")
    }
    inverse = {
        row["query_id"]: row
        for row in read_jsonl(
            root / "ms_swift_bidirectional_v1" / "audit" / f"{split}.jsonl"
        )
    }
    if len(mixed) != len(mixed_index):
        raise ValueError("mixed rows and index have different lengths")

    output = []
    seen_ids = set()
    for offset, (row, index) in enumerate(zip(mixed, mixed_index, strict=True)):
        if index["mixed_index"] != offset:
            raise ValueError(f"mixed_index is not contiguous at row {offset}")
        audit = (forward if index["row_kind"] == "forward" else inverse)[index["query_id"]]
        assert_aligned(row, audit)
        category = (
            str(audit["head"])
            if index["row_kind"] == "forward"
            else f"inverse.{audit['entity_type']}"
        )
        metadata = compact_metadata(audit, suite="bidirectional", category=category)
        metadata["row_kind"] = index["row_kind"]
        enriched = {
            "id": audit["query_id"],
            "images": list(row["images"]),
            "messages": row["messages"],
            "metadata": metadata,
        }
        output.append(enriched)
        seen_ids.add(enriched["id"])

    spatial = read_jsonl(root / "spatial_robber_v1" / f"{split}.jsonl")
    spatial_audit = read_jsonl(root / "spatial_robber_v1" / "audit" / f"{split}.jsonl")
    if len(spatial) != len(spatial_audit):
        raise ValueError("spatial rows and audit have different lengths")
    for row, audit in zip(spatial, spatial_audit, strict=True):
        assert_aligned(row, audit)
        if audit["query_id"] in seen_ids:
            raise ValueError(f"duplicate query_id: {audit['query_id']}")
        enriched = {
            "id": audit["query_id"],
            "images": list(row["images"]),
            "messages": row["messages"],
            "metadata": compact_metadata(
                audit,
                suite="spatial_robber",
                category=str(audit["task_family"]),
            ),
        }
        output.append(enriched)
        seen_ids.add(enriched["id"])
    return output


def smoke_rows(rows: Iterable[JsonDict], limit: int) -> list[JsonDict]:
    """Round-robin metadata strata so a small smoke touches each task family."""

    groups: dict[tuple[str, str], deque[JsonDict]] = defaultdict(deque)
    for row in rows:
        metadata = row["metadata"]
        groups[(metadata["suite"], metadata["category"])].append(row)
    selected = []
    queues = deque(groups[key] for key in sorted(groups))
    while queues and len(selected) < limit:
        queue = queues.popleft()
        selected.append(queue.popleft())
        if queue:
            queues.append(queue)
    return selected


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in materialized:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(materialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/generated/board_recognition/replay_v1"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--smoke-rows", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args.root.resolve(), args.split)
    count = write_jsonl(args.output, rows)
    print(f"output={args.output} rows={count}")
    if args.smoke_output:
        smoke_count = write_jsonl(args.smoke_output, smoke_rows(rows, args.smoke_rows))
        print(f"smoke_output={args.smoke_output} rows={smoke_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
