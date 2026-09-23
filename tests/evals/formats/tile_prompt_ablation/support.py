"""Shared helpers for tile prompt ablation factors, scoring, and provider contracts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    DEFAULT_QA_PATH,
    load_questions,
)


def materialize_images(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "isolated_visuals"
    rows = [
        json.loads(line)
        for line in DEFAULT_QA_PATH.read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["category"] != "isolated_tile_resource_number":
            continue
        image_path = image_root / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"image:{row['sample_id']}".encode())
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return qa_path, image_root


def load_fixture_questions(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    qa_path, image_root = materialize_images(tmp_path)
    return qa_path, image_root, load_questions(qa_path, image_root)
