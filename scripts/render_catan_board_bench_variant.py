#!/usr/bin/env python
"""Render a resolution/framing variant from frozen CatanBoardBench contracts."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from catan_board_bench.render import DEFAULT_RENDER_STYLE, render_contract_image


COPIED_DIRECTORIES = ("contracts", "questions", "leakage")
COPIED_FILES = ("manifest.jsonl",)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_inputs(source_dir: Path, output_dir: Path, image_size: int) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("source_dir and output_dir must differ")
    for required in ("metadata.json", "manifest.jsonl", "contracts", "questions"):
        if not (source_dir / required).exists():
            raise FileNotFoundError(source_dir / required)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def render_variant(
    source_dir: Path,
    output_dir: Path,
    *,
    image_size: int,
    view_padding_factor: float,
    target_board_canvas_fraction: float | None,
) -> dict[str, Any]:
    validate_inputs(source_dir, output_dir, image_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    for directory_name in COPIED_DIRECTORIES:
        source = source_dir / directory_name
        if source.exists():
            shutil.copytree(source, output_dir / directory_name)
    for file_name in COPIED_FILES:
        shutil.copy2(source_dir / file_name, output_dir / file_name)

    style = replace(DEFAULT_RENDER_STYLE, view_padding_factor=view_padding_factor)
    images_dir = output_dir / "images"
    images_dir.mkdir()
    rows = list(iter_jsonl(source_dir / "manifest.jsonl"))
    for row in rows:
        contract = json.loads((source_dir / row["contract_path"]).read_text())
        image = render_contract_image(contract, image_size=image_size, style=style)
        image_path = output_dir / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)

    source_metadata = json.loads((source_dir / "metadata.json").read_text())
    metadata = {
        **source_metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image_size": [image_size, image_size],
        "rendered_images": len(rows),
        "render_variant": {
            "derived_from": str(source_dir),
            "renderer": "data_pipeline.catan_board_bench.render",
            "style": asdict(style),
            "view_padding_factor": view_padding_factor,
            "target_board_canvas_fraction": target_board_canvas_fraction,
        },
    }
    if isinstance(metadata.get("files"), dict):
        metadata["files"] = dict(metadata["files"])
        metadata["files"]["question_dir"] = "questions"
    write_json(output_dir / "metadata.json", metadata)

    canvas_text = (
        f" and approximately {target_board_canvas_fraction:.0%} board coverage"
        if target_board_canvas_fraction is not None
        else ""
    )
    (output_dir / "README.md").write_text(
        "# CatanBoardBench render variant\n\n"
        f"Derived from `{source_dir}` with `{image_size}x{image_size}` images, "
        f"`view_padding_factor={view_padding_factor}`{canvas_text}.\n\n"
        "Contracts, questions, leakage data, and manifest rows are copied without "
        "changing benchmark answers.\n"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", required=True, type=int)
    parser.add_argument("--view-padding-factor", required=True, type=float)
    parser.add_argument("--target-board-canvas-fraction", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = render_variant(
        args.source_dir,
        args.output_dir,
        image_size=args.image_size,
        view_padding_factor=args.view_padding_factor,
        target_board_canvas_fraction=args.target_board_canvas_fraction,
    )
    print(f"rendered_images={metadata['rendered_images']}")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
