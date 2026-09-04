"""Build leakage-checked Catan VLM SFT JSONL files.

The output format is intentionally simple JSONL:

{
  "id": "...",
  "image": "path/to/image.png",
  "messages": [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ],
  "metadata": {...}
}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from sft.paths import repository_relative_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXCLUDE = (
    ROOT / "evals/catan_board_bench/datasets/catan_board_bench_100/leakage/benchmark_game_ids.json"
)
DEFAULT_PROMPT_PREFIX = "Answer exactly using the Catan tokens requested. Do not explain."


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def norm_game_id(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    return match.group(1) if match else str(value)


def load_excluded_game_ids(path: Path | None) -> set[str]:
    if path is None:
        raise ValueError("A held-out game-ID ledger is required")
    if not path.is_file():
        raise FileNotFoundError(f"Held-out game-ID ledger not found: {path}")
    payload = read_json(path)
    if isinstance(payload, list):
        ids = payload
    elif isinstance(payload, dict) and isinstance(payload.get("benchmark_game_ids"), list):
        ids = payload["benchmark_game_ids"]
    else:
        raise ValueError(f"Held-out game-ID ledger has an invalid schema: {path}")
    return {game_id for game_id in (norm_game_id(value) for value in ids) if game_id}


def load_manifest_game_ids(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for _, row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if not sample_id:
            continue
        source = row.get("source") or {}
        game_id = norm_game_id(source.get("game_id") or source.get("replay_file"))
        mapping[sample_id] = {
            "game_id": game_id,
            "source": source,
        }
    return mapping


def resolve_image_path(image_root: Path, image_path: str) -> str:
    path = Path(image_path)
    resolved = path.resolve() if path.is_absolute() else (image_root / path).resolve()
    return repository_relative_path(resolved)


def build_user_text(question: str, prompt_prefix: str) -> str:
    question = question.strip()
    if not prompt_prefix:
        return question
    return f"{prompt_prefix.strip()}\n\nQuestion: {question}"


def convert_row(
    row: dict[str, Any],
    *,
    image_root: Path,
    manifest_by_sample: dict[str, dict[str, Any]],
    prompt_prefix: str,
) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    manifest_info = manifest_by_sample.get(sample_id, {})
    game_id = norm_game_id(
        row.get("game_id")
        or (row.get("source") or {}).get("game_id")
        or manifest_info.get("game_id")
    )
    image = resolve_image_path(image_root, row["image_path"])
    question = str(row["question"])
    answer = str(row["answer"])

    return {
        "id": row.get("id"),
        "image": image,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_user_text(question, prompt_prefix)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "metadata": {
            "sample_id": sample_id,
            "game_id": game_id,
            "category": row.get("category"),
            "scoring": row.get("scoring"),
            "source": manifest_info.get("source", row.get("source")),
            "target": row.get("target"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-jsonl", type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-game-ids", default=DEFAULT_EXCLUDE, type=Path)
    parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    excluded_ids = load_excluded_game_ids(args.exclude_game_ids)
    manifest_by_sample = load_manifest_game_ids(args.manifest_jsonl)

    rows: list[dict[str, Any]] = []
    blocked: list[tuple[str | None, str | None]] = []
    for _, row in iter_jsonl(args.qa_jsonl):
        converted = convert_row(
            row,
            image_root=args.image_root,
            manifest_by_sample=manifest_by_sample,
            prompt_prefix=args.prompt_prefix,
        )
        game_id = converted["metadata"].get("game_id")
        if game_id in excluded_ids:
            blocked.append((converted.get("id"), game_id))
            continue
        rows.append(converted)
        if args.limit and len(rows) >= args.limit:
            break

    if blocked:
        preview = ", ".join(f"{qid}:{gid}" for qid, gid in blocked[:10])
        print(
            f"Refusing to write SFT data: {len(blocked)} rows use excluded held-out game IDs. "
            f"Examples: {preview}",
            file=sys.stderr,
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    categories = sorted({row["metadata"].get("category") for row in rows})
    game_ids = sorted(
        {row["metadata"].get("game_id") for row in rows if row["metadata"].get("game_id")}
    )
    print(f"wrote_rows={len(rows)}")
    print(f"unique_games={len(game_ids)}")
    print(f"categories={','.join(str(category) for category in categories)}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
