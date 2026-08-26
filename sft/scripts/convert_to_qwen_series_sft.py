"""Convert Catan SFT JSONL into Qwen-VL-Series-Finetune JSON.

The upstream Qwen trainer expects a JSON list with LLaVA-style conversation
objects. This script keeps our local dataset builders free to emit the simpler
messages JSONL format while making the training backend upstream-compatible.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from sft.paths import repository_relative_path, resolve_dataset_asset


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


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "image":
                parts.append("<image>")
            elif item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    raise TypeError(f"unsupported message content type: {type(content)!r}")


def _assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) for item in content if item.get("type") == "text"
        ).strip()
    return str(content).strip()


def convert_row(
    row: dict[str, Any],
    *,
    image_name: str | None = None,
) -> dict[str, Any]:
    """Convert one local SFT row into the upstream Qwen conversation schema."""

    if "conversations" in row:
        converted = dict(row)
        if image_name is not None:
            converted["image"] = image_name
        return converted

    messages = row["messages"]
    if len(messages) < 2:
        raise ValueError(f"row {row.get('id')} must contain user and assistant messages")

    user_text = _content_text(messages[0].get("content", ""))
    answer_text = _assistant_text(messages[1])

    converted = {
        "id": row.get("id"),
        "conversations": [
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": answer_text},
        ],
    }
    if image_name is not None:
        converted["image"] = image_name
    elif row.get("image"):
        converted["image"] = str(row["image"])
    return converted


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    copy_images_to: Path | None = None,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_images_to is not None:
        copy_images_to.mkdir(parents=True, exist_ok=True)

    rows = []
    image_count = 0
    image_map: dict[str, str] = {}
    for _, row in iter_jsonl(input_path):
        image_name = None
        if row.get("image"):
            image_path = resolve_dataset_asset(input_path, row["image"])
            if copy_images_to is not None:
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                image_name = image_map.get(str(image_path))
                if image_name is None:
                    suffix = image_path.suffix or ".png"
                    image_name = f"{len(image_map):06d}{suffix}"
                    shutil.copy2(image_path, copy_images_to / image_name)
                    image_map[str(image_path)] = image_name
                    image_count += 1
            else:
                image_name = repository_relative_path(image_path)

        rows.append(convert_row(row, image_name=image_name))

    with output_path.open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return len(rows), image_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copy-images-to", type=Path)
    args = parser.parse_args()

    rows, images = convert_file(
        args.input,
        args.output,
        copy_images_to=args.copy_images_to,
    )
    print(f"wrote_rows={rows}")
    print(f"copied_images={images}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
