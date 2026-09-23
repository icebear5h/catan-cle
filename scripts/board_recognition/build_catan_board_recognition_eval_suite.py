"""Build an enriched held-out eval suite from existing Catan recognition splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path

from cle.players.data import JsonValue

JsonDict = dict[str, JsonValue]


def read_jsonl(path: Path) -> list[JsonDict]:
    with path.open() as handle:
        rows: list[JsonDict] = []
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} contains a non-object row")
            rows.append(row)
        return rows


def _values(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    return value


def _objects(value: JsonValue, label: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for item in _values(value, label):
        if not isinstance(item, dict):
            raise ValueError(f"{label} entries must be JSON objects")
        rows.append(item)
    return rows


def _object(value: JsonValue, label: str) -> JsonDict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def message_text(row: JsonDict, index: int) -> str:
    return str(_objects(row["messages"], "messages")[index]["content"]).strip()


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


def build_rows(
    root: Path,
    split: str,
    *,
    supplement_dir: Path | None = None,
    spatial_only: bool = False,
    answer_only: bool = False,
) -> list[JsonDict]:
    if answer_only and not spatial_only:
        raise ValueError("--answer-only requires --spatial-only")
    output: list[JsonDict] = []
    seen_ids: set[JsonValue] = set()
    if not spatial_only:
        mixed = read_jsonl(root / "ms_swift_bidirectional_v1" / "mixed" / f"{split}.jsonl")
        mixed_index = read_jsonl(
            root / "ms_swift_bidirectional_v1" / "mixed_index" / f"{split}.jsonl"
        )
        forward = {
            str(row["query_id"]): row
            for row in read_jsonl(root / "ms_swift_semantic_v1" / "audit" / f"{split}.jsonl")
        }
        inverse = {
            str(row["query_id"]): row
            for row in read_jsonl(root / "ms_swift_bidirectional_v1" / "audit" / f"{split}.jsonl")
        }
        if len(mixed) != len(mixed_index):
            raise ValueError("mixed rows and index have different lengths")

        for offset, (row, index) in enumerate(zip(mixed, mixed_index, strict=True)):
            if index["mixed_index"] != offset:
                raise ValueError(f"mixed_index is not contiguous at row {offset}")
            audit = (forward if index["row_kind"] == "forward" else inverse)[
                str(index["query_id"])
            ]
            assert_aligned(row, audit)
            category = (
                str(audit["head"])
                if index["row_kind"] == "forward"
                else f"inverse.{audit['entity_type']}"
            )
            metadata = compact_metadata(audit, suite="bidirectional", category=category)
            metadata["row_kind"] = index["row_kind"]
            enriched: JsonDict = {
                "id": audit["query_id"],
                "images": list(_values(row["images"], "images")),
                "messages": list(_objects(row["messages"], "messages")),
                "metadata": metadata,
            }
            output.append(enriched)
            seen_ids.add(enriched["id"])

    supplement_root = supplement_dir if supplement_dir is not None else root / "spatial_robber_v1"
    spatial = read_jsonl(supplement_root / f"{split}.jsonl")
    spatial_audit = read_jsonl(supplement_root / "audit" / f"{split}.jsonl")
    if len(spatial) != len(spatial_audit):
        raise ValueError("spatial rows and audit have different lengths")
    for row, audit in zip(spatial, spatial_audit, strict=True):
        assert_aligned(row, audit)
        if audit.get("split", split) != split:
            raise ValueError(f"split mismatch for {audit['query_id']}")
        if "image_name" in audit and row["images"] != [audit["image_name"]]:
            raise ValueError(f"image mismatch for {audit['query_id']}")
        if spatial_only and audit["task_family"] != "spatial_grounding":
            continue
        if audit["query_id"] in seen_ids:
            raise ValueError(f"duplicate query_id: {audit['query_id']}")
        metadata = compact_metadata(
            audit,
            suite="spatial_robber",
            category=str(audit["task_family"]),
        )
        if audit["task_family"] == "spatial_grounding":
            entity_type = str(audit["task_type"]).split("_", 1)[0]
            if entity_type not in {"node", "tile"}:
                raise ValueError(f"unknown spatial entity for {audit['query_id']}")
            metadata["entity_type"] = entity_type
        metadata["source_supplement"] = str(supplement_root.resolve())
        metadata["source_split"] = split
        eval_id = str(audit["query_id"])
        if supplement_dir is not None or spatial_only:
            # Versioned sources must not alias historical prompt-blind fingerprints.
            eval_id = f"{supplement_root.name}:{split}:{eval_id}"
        spatial_enriched: JsonDict = {
            "id": eval_id,
            "images": list(_values(row["images"], "images")),
            "messages": list(_objects(row["messages"], "messages")),
            "metadata": metadata,
        }
        if answer_only:
            task_type = str(audit["task_type"])
            if task_type.endswith("_direction_token"):
                instruction = "Answer with only one of the two tokens shown in the question."
            elif task_type.endswith(("_yes", "_no")):
                instruction = "Answer with exactly one word: yes or no."
            else:
                raise ValueError(f"unsupported answer-only spatial task: {task_type}")
            messages = _objects(row["messages"], "messages")
            spatial_enriched["messages"] = [
                {**messages[0], "content": f"{messages[0]['content']}\n{instruction}"},
                *messages[1:],
            ]
            spatial_enriched["id"] = f"{eval_id}:answer_only"
            metadata["prompt_variant"] = "answer_only"
        output.append(spatial_enriched)
        seen_ids.add(audit["query_id"])
    return output


def smoke_rows(rows: Iterable[JsonDict], limit: int) -> list[JsonDict]:
    """Round-robin metadata strata so a small smoke touches each task family."""

    groups: dict[tuple[str, str], deque[JsonDict]] = defaultdict(deque)
    for row in rows:
        metadata = _object(row["metadata"], "metadata")
        groups[(str(metadata["suite"]), str(metadata["category"]))].append(row)
    selected: list[JsonDict] = []
    queues = deque(groups[key] for key in sorted(groups))
    while queues and len(selected) < limit:
        queue = queues.popleft()
        selected.append(queue.popleft())
        if queue:
            queues.append(queue)
    return selected


def write_jsonl(path: Path, rows: Iterable[JsonDict], *, overwrite: bool = False) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x") as handle:
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
    parser.add_argument(
        "--supplement-dir",
        type=Path,
        help="Supplement source directory (default: ROOT/spatial_robber_v1).",
    )
    parser.add_argument(
        "--spatial-only",
        action="store_true",
        help="Skip base board rows and retain only task_family=spatial_grounding.",
    )
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Append a neutral answer-format instruction (requires --spatial-only).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--smoke-rows", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing eval outputs.")
    args = parser.parse_args()
    if args.answer_only and not args.spatial_only:
        parser.error("--answer-only requires --spatial-only")
    return args


def main() -> int:
    args = parse_args()
    outputs = [args.output] + ([args.smoke_output] if args.smoke_output else [])
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("output and smoke-output must be different paths")
    for path in outputs:
        if not args.overwrite and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"output already exists: {path}; pass --overwrite")
    rows = build_rows(
        args.root.resolve(),
        args.split,
        supplement_dir=args.supplement_dir,
        spatial_only=args.spatial_only,
        answer_only=args.answer_only,
    )
    count = write_jsonl(args.output, rows, overwrite=args.overwrite)
    print(f"output={args.output} rows={count}")
    if args.smoke_output:
        smoke_count = write_jsonl(
            args.smoke_output, smoke_rows(rows, args.smoke_rows), overwrite=args.overwrite
        )
        print(f"smoke_output={args.smoke_output} rows={smoke_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
