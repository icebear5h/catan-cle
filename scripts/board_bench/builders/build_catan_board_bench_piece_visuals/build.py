"""Assemble the isolated-visual dataset and its metadata."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.constants import COLORS
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.isolated import (
    add_node_examples,
    add_port_examples,
    add_road_examples,
    add_robber_examples,
    add_tile_examples,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.patches import (
    add_local_patch_examples,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.writer import DatasetWriter
from scripts.board_bench.shapes import JsonDict, text

__all__ = ["build_dataset"]


def build_dataset(
    *,
    output_dir: Path,
    contract_path: Path,
    image_size: int,
    variants: int,
    prompt_prefix: str,
) -> JsonDict:
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(output_dir, image_size, prompt_prefix)

    for variant in range(variants):
        add_tile_examples(writer, variant)
        add_road_examples(writer, variant)
        add_node_examples(writer, variant)
        add_port_examples(writer, variant)
        add_robber_examples(writer, variant)
    add_local_patch_examples(writer, contract_path)

    counts = Counter(text(row["category"], "category") for row in writer.qa_rows)
    metadata: JsonDict = {
        "schema": "catan_isolated_visual_grounding/v1",
        "name": "isolated_visuals",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_root": "playground/frontend/public/assets",
        "image_size": [image_size, image_size],
        "variants": variants,
        "sample_count": len(writer.manifest_rows),
        "qa_count": len(writer.qa_rows),
        "category_counts": {key: count for key, count in sorted(counts.items())},
        "colors": list(COLORS),
        "files": {
            "manifest": "manifest.jsonl",
            "qa": "questions/qa.jsonl",
            "questions": "questions/questions.jsonl",
            "answer_key": "questions/answer_key.jsonl",
            "messages": "messages.jsonl",
            "images_dir": "images",
        },
    }
    writer.write(metadata)
    return metadata
